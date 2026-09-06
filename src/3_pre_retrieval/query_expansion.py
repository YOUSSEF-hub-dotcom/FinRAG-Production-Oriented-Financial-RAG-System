"""
Pre-Retrieval Stage (Module 3) -- Conditional Financial Query Expansion.

The system does NOT run multi-query expansion blindly (to protect accounting
terminology and save resources). Expansion is triggered only when
`need_expansion` is true OR the standalone query is shorter than
`EXPANSION_MIN_WORDS` words:

  - expanded  -> the standalone query plus 3-4 rich financial synonym
                 variants, all embedded in a SINGLE GPU batch pass
                 (nomic-embed-text-v1.5 via the shared EmbeddingEngine).
  - skipped   -> the standalone query is used as-is, saving ~75% of the
                 retrieval work (spec: "التوسيع المشروط المالي").

The variant generator is deterministic (pure rule-based synonym substitution):
zero hallucination, zero extra LLM latency, and fully unit-testable.
"""

import re
from typing import Callable, Optional

from config.logging_config import get_logger
from config.settings import EXPANSION_MIN_WORDS
from pre_retrieval_schemas import ExpansionResult

logger = get_logger("pre_retrieval.query_expansion")

# Company display names used when expanding a bare ticker into a full subject.
_COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}

# Financial synonym banks (term -> alternative phrasings).
_SYNONYM_BANKS: list[tuple[str, tuple[str, ...]]] = [
    ("revenue", ("top-line revenue", "net sales")),
    ("earnings", ("net earnings", "profitability")),
    ("income", ("net income", "earnings")),
    ("cash flow", ("operating cash flow", "cash from operations")),
    ("profit", ("net profit", "earnings")),
    ("margin", ("profit margin", "operating margin")),
    ("assets", ("total assets", "asset base")),
    ("liabilities", ("total liabilities", "debt obligations")),
    ("equity", ("shareholders' equity", "book value")),
    ("debt", ("total debt", "outstanding borrowings")),
    ("dividend", ("dividend payout", "per-share dividend")),
    ("eps", ("earnings per share", "diluted EPS")),
]

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")

EmbedFn = Callable[[list[str]], list[list[float]]]


class QueryExpander:
    """
    Conditional financial multi-query expansion with a single batch embed.

    Args:
        min_words: Minimum standalone-query word count; shorter queries always
            trigger expansion.
        max_variants: Maximum number of retrieval queries to produce.
    """

    def __init__(
        self,
        min_words: int = EXPANSION_MIN_WORDS,
        max_variants: int = 4,
    ):
        self._min_words = min_words
        self._max_variants = max_variants

    def expand(
        self,
        standalone_query: str,
        need_expansion: bool = False,
        embed_fn: Optional[EmbedFn] = None,
    ) -> ExpansionResult:
        """
        Produce the final retrieval query list for the REWRITE route.

        Args:
            standalone_query: Coreference-resolved query from the intent router.
            need_expansion: Router's expansion decision.
            embed_fn: Batch embedder (list[str] -> list[list[float]]). When
                provided and expansion is triggered, all variants are embedded
                in a single GPU batch pass so the pipeline does not re-embed.

        Returns:
            ExpansionResult with `queries`, optional batch `embeddings`, and
            the `expanded` flag (False => 75% overhead saved).
        """
        text = str(standalone_query).strip()
        if not text:
            return ExpansionResult(queries=[], expanded=False)

        word_count = self._word_count(text)
        should_expand = bool(need_expansion) or word_count < self._min_words

        if not should_expand:
            logger.info(
                "Expansion skipped (words=%d, need=%s): %r",
                word_count,
                need_expansion,
                text[:80],
            )
            return ExpansionResult(queries=[text], expanded=False)

        queries = self._build_variants(text)
        logger.info(
            "Expansion triggered (words=%d, need=%s): %d variants",
            word_count,
            need_expansion,
            len(queries),
        )

        embeddings: list[list[float]] = []
        if embed_fn is not None:
            embeddings = embed_fn(queries)

        return ExpansionResult(
            queries=queries,
            embeddings=embeddings,
            expanded=True,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        return len([w for w in re.split(r"\W+", text) if w])

    def _build_variants(self, text: str) -> list[str]:
        """Deterministically build 3-4 distinct retrieval queries."""
        variants = [text]
        for candidate in self._variant_candidates(text):
            if len(variants) >= self._max_variants:
                break
            if candidate not in variants:
                variants.append(candidate)
        return variants[: self._max_variants]

    def _variant_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        # 1. Financial synonym substitutions (first occurrence only).
        for term, synonyms in _SYNONYM_BANKS:
            lowered = text.lower()
            match = re.search(re.escape(term), lowered)
            if not match:
                continue
            head, tail = text[: match.start()], text[match.end() :]
            for syn in synonyms:
                candidates.append(f"{head}{syn}{tail}")

        # 2. Bare ticker -> full company-name expansion.
        ticker_match = _TICKER_PATTERN.search(text)
        if ticker_match:
            ticker = ticker_match.group(0)
            company = _COMPANY_NAMES.get(ticker)
            if company:
                candidates.append(
                    f"{text[: ticker_match.start()]}{company} ({ticker})"
                    f"{text[ticker_match.end():]}"
                )

        # 3. Fiscal-year rewording variants.
        year_match = _YEAR_PATTERN.search(text)
        if year_match:
            year = year_match.group(0)
            candidates.append(
                f"{text[: year_match.start()]}fiscal year {year}"
                f"{text[year_match.end():]}"
            )
            candidates.append(
                f"{text[: year_match.start()]}FY {year}"
                f"{text[year_match.end():]}"
            )

        # 4. Retrieval-focus phrasing (annual report framing).
        if "10-k" in text.lower() or "10k" in text.lower():
            candidates.append(f"{text} as disclosed in the SEC 10-K annual report")
        else:
            candidates.append(f"{text} as reported in the company's annual report")

        return candidates
