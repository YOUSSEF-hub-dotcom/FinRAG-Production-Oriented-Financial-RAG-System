"""
Synthetic Data Generator for the Module 6 Evaluation pipeline.

Reads raw financial documents from ``data/`` (AAPL, MSFT, NVDA) and uses the
Ragas ``TestsetGenerator`` (wrapped around the Groq async API via LangChain)
to generate financial QA pairs with ground truths.

STRICT LIMIT: synthetic generation is hard-capped to EXACTLY 25 questions
(``test_size=25``) to stay within the Groq free-tier TPM budget and to
guarantee predictable batch execution in ``batch_runner.py``.

The output is written strictly to ``artifacts/test_dataset.csv`` with the
columns ``question`` and ``ground_truth``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Ensure src subdirectories (numbered names, not valid Python packages) are
# importable from the evaluation package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("1_ingestion", "5_generation"):
    _p = str(_PROJECT_ROOT / "src" / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.logging_config import get_logger
from config.settings import (
    DATA_DIR,
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    SUPPORTED_TICKERS,
    SYNTHETIC_MAX_DOC_CHARS,
    SYNTHETIC_TEST_SIZE,
    SYNTH_FALLBACK_MODEL,
    SYNTH_GENERATOR_MODEL,
    TEST_DATASET_PATH,
)

logger = get_logger("evaluation.synthetic_generator")

# ---------------------------------------------------------------------------
# Document loading (raw SEC filings)
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".html", ".htm", ".sgml"}


def _iter_filing_text_files(ticker_dir: Path, max_doc_chars: int):
    """
    Yield cleaned, truncated raw filing texts for a single ticker directory.

    Walks every subdirectory (accession number or year) inside
    ``<data>/<TICKER>/10-K`` and yields ``Document`` objects for each readable
    text-like file (TXT / HTML / SGML). PDFs and DOCX files are skipped
    because the RAG pipeline is tuned for raw SEC submission text.
    """
    filing_root = ticker_dir / "10-K"
    if not filing_root.exists():
        logger.warning("No 10-K directory found for ticker %s", ticker_dir.name)
        return

    for subdir in sorted(filing_root.iterdir()):
        if not subdir.is_dir():
            continue
        for file in sorted(subdir.iterdir()):
            if not file.is_file() or file.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            try:
                raw = file.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Skipping unreadable filing %s: %s", file, exc)
                continue
            text = raw.strip()
            if len(text) < 200:
                logger.debug("Skipping too-short filing %s (%d chars)", file, len(text))
                continue
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars]
            logger.info(
                "Loaded filing %s (%d chars)", file, len(text)
            )
            yield Document(
                page_content=text,
                metadata={
                    "ticker": ticker_dir.name,
                    "source_file": str(file.relative_to(ticker_dir)),
                },
            )


def load_documents(
    tickers: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
    max_doc_chars: int = SYNTHETIC_MAX_DOC_CHARS,
) -> list[Document]:
    """
    Load raw financial documents for the given tickers.

    Args:
        tickers: List of ticker symbols (default: SUPPORTED_TICKERS).
        data_dir: Root data directory (default: DATA_DIR from settings).
        max_doc_chars: Per-file character cap to keep the knowledge graph
            construction within Groq free-tier TPM limits.

    Returns:
        List of LangChain ``Document`` objects with ticker metadata.
    """
    tickers = tickers or list(SUPPORTED_TICKERS)
    data_dir = data_dir or Path(DATA_DIR)

    documents: list[Document] = []
    for ticker in tickers:
        ticker_dir = data_dir / ticker
        if not ticker_dir.exists():
            logger.warning("Data directory not found for ticker %s: %s", ticker, ticker_dir)
            continue
        documents.extend(_iter_filing_text_files(ticker_dir, max_doc_chars))

    if not documents:
        raise FileNotFoundError(
            f"No raw financial documents found under {data_dir} for tickers {tickers}. "
            "Run the Module 1 ingestion pipeline first or place SEC filings in data/."
        )
    logger.info("Loaded %d raw financial documents for %s", len(documents), tickers)
    return documents


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

_SYNTH_LLM_CONTEXT = (
    "You are generating questions and reference answers for an automated "
    "evaluation of a financial RAG assistant. The assistant answers questions "
    "about SEC 10-K filings for AAPL, MSFT and NVDA: revenue, operating income, "
    "segment performance, risk factors, cash flow and balance sheet items. "
    "Questions must be answerable strictly from the provided document excerpts."
)


class SyntheticDataGenerator:
    """
    Generates a STRICT 25-question synthetic QA dataset via Ragas.

    Wraps ``ChatGroq`` (primary -> fallback generation models, preserved from
    Module 5) with ragas' ``LangchainLLMWrapper`` so the TestsetGenerator
    drives the Groq async API under the hood. The 25 generated samples are
    validated and persisted to ``artifacts/test_dataset.csv``.
    """

    def __init__(
        self,
        generator_model: str = SYNTH_GENERATOR_MODEL,
        fallback_model: str = SYNTH_FALLBACK_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int = LLM_SEED,
        test_size: int = SYNTHETIC_TEST_SIZE,
        data_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        max_doc_chars: int = SYNTHETIC_MAX_DOC_CHARS,
    ):
        """
        Args:
            generator_model: Groq primary generation model for the testset.
            fallback_model: Groq fallback generation model.
            temperature: Sampling temperature (deterministic generation uses 0.0).
            max_tokens: Max output tokens per generation call.
            seed: Random seed for reproducibility.
            test_size: EXACT number of QA pairs to generate (default 25).
            data_dir: Root data directory for raw filings.
            output_path: Destination CSV path (default TEST_DATASET_PATH).
            max_doc_chars: Per-document character cap for knowledge-graph build.
        """
        if test_size != 25:
            raise ValueError(
                f"SYNTHETIC_TEST_SIZE must be exactly 25 to respect the Groq "
                f"free-tier TPM budget (got {test_size})."
            )
        self._generator_model = generator_model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._test_size = test_size
        self._data_dir = data_dir
        self._output_path = output_path or Path(TEST_DATASET_PATH)
        self._max_doc_chars = max_doc_chars

    # -- Internal helpers ----------------------------------------------------

    def _build_chat_groq(self, model_name: str) -> ChatGroq:
        """Build a ChatGroq client for the given model (json_object mode)."""
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            model_kwargs={
                "seed": self._seed,
                "response_format": {"type": "json_object"},
            },
        )

    def _build_testset_generator(self):
        """
        Lazily build the ragas TestsetGenerator around Groq.

        Imports are deferred so the evaluation module can be unit-tested
        without the (heavy) ragas dependency installed.
        """
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.testset import TestsetGenerator

        # Reuse the project embedding engine (nomic-embed-text-v1.5, CUDA)
        # for the knowledge-graph node vectors by exposing it as a LangChain
        # embedder.
        from database_indexer import EmbeddingEngine
        from langchain_core.embeddings import Embeddings

        embedding_engine = EmbeddingEngine()

        class _EmbeddingAdapter(Embeddings):
            """Minimal LangChain adapter over the project EmbeddingEngine."""

            def embed_documents(self, texts):
                return embedding_engine.embed(texts)

            def embed_query(self, text):
                return embedding_engine.embed_single(text)

        llm = self._build_chat_groq(self._generator_model)
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(_EmbeddingAdapter())
        return TestsetGenerator.from_langchain(ragas_llm, ragas_embeddings)

    def _build_query_distribution(self, generator):
        """
        Build a weighted QueryDistribution of synthesizer types.

        Balances single-hop factual questions, multi-hop reasoning and
        abstract multi-hop queries so the 25 samples exercise the RAG stack
        across retrieval difficulty levels.
        """
        from ragas.testset.synthesizers import (
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer,
            SingleHopSpecificQuerySynthesizer,
        )

        return [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=generator.llm, llm_context=_SYNTH_LLM_CONTEXT
                ),
                0.5,
            ),
            (
                MultiHopSpecificQuerySynthesizer(
                    llm=generator.llm, llm_context=_SYNTH_LLM_CONTEXT
                ),
                0.3,
            ),
            (
                MultiHopAbstractQuerySynthesizer(
                    llm=generator.llm, llm_context=_SYNTH_LLM_CONTEXT
                ),
                0.2,
            ),
        ]

    # -- Public API ----------------------------------------------------------

    def generate(
        self,
        documents: Optional[list[Document]] = None,
        test_size: Optional[int] = None,
    ):
        """
        Generate the synthetic QA dataset and persist it to CSV.

        Args:
            documents: Optional pre-loaded documents (default: load from data/).
            test_size: Number of QA pairs (must equal 25).

        Returns:
            pandas.DataFrame with columns ``question`` and ``ground_truth``.
        """
        import pandas as pd

        size = test_size or self._test_size
        if size != 25:
            raise ValueError(
                "Synthetic generation is strictly limited to 25 questions "
                f"for the Groq free-tier TPM budget (got {size})."
            )

        if documents is None:
            documents = load_documents(
                data_dir=self._data_dir, max_doc_chars=self._max_doc_chars
            )

        from ragas.run_config import RunConfig

        generator = self._build_testset_generator()
        query_distribution = self._build_query_distribution(generator)
        run_config = RunConfig(
            timeout=180,
            max_retries=6,
            max_wait=60,
            max_workers=2,
            seed=self._seed,
            log_tenacity=True,
        )

        logger.info(
            "Generating synthetic testset: test_size=%d documents=%d",
            size,
            len(documents),
        )
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=size,
            query_distribution=query_distribution,
            run_config=run_config,
        )

        # Always the primary model; ragas retries internally. Fallback model
        # is preserved for reference and used by the online pipeline only.
        df = self._convert_testset(testset)
        self._save(df)
        return df

    @staticmethod
    def _convert_testset(testset) -> "pd.DataFrame":
        """
        Convert a ragas Testset (or any object exposing ``to_pandas``) into
        the canonical ``question`` / ``ground_truth`` DataFrame.

        Accepts either a ragas ``Testset`` instance or a pandas DataFrame that
        already exposes ``user_input`` and ``reference`` columns, so the
        schema logic stays unit-testable without a live ragas install.
        """
        import pandas as pd

        if hasattr(testset, "to_pandas"):
            raw = testset.to_pandas()
        else:
            raw = testset

        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                "testset must expose to_pandas() or be a pandas.DataFrame"
            )

        if "user_input" not in raw.columns or "reference" not in raw.columns:
            raise ValueError(
                "Ragas testset is missing required columns user_input/reference. "
                f"Got columns: {list(raw.columns)}"
            )

        rows = raw[["user_input", "reference"]].copy()
        rows.columns = ["question", "ground_truth"]
        rows = rows.dropna(subset=["question", "ground_truth"])
        rows = rows.reset_index(drop=True)
        rows["question"] = rows["question"].astype(str)
        rows["ground_truth"] = rows["ground_truth"].astype(str)
        return rows

    def _save(self, df) -> None:
        """Persist the dataset strictly to artifacts/test_dataset.csv."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self._output_path, index=False)
        logger.info(
            "Saved %d synthetic QA pairs to %s",
            len(df),
            self._output_path,
        )
