"""
Financial RAG Pipeline Orchestrator.

Master class that wires together Module 1 (Ingestion/Retrieval) and
Module 2 (Generation/Guardrails) into a single queryable interface.

Flow:
    1. Check Redis semantic cache for identical query
    2. Vector search Qdrant with optional metadata pre-filtering
    3. Fetch full text from MongoDB (Qdrant payload has no raw_text)
    4. [Optional Module 4] Hybrid search (dense+BM25+RRF, 40 chunks) ->
       post-retrieval (Cross-Encoder rerank top-8 -> Table Shield ->
       Cylinder reorder: best at head, 2nd-best at tail)
    5. Format retrieved docs into XML <CONTEXT> tags
    6. Generate answer via Groq LLM (primary → fallback)
    7. Background async guardrail verification + cache write
    8. Log end-to-end metrics to MLflow
"""

import asyncio
import concurrent.futures
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import mlflow

# Ensure src subdirectories (numbered names, not valid Python packages) are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = Path(__file__).resolve().parent
for _subdir in ("1_ingestion", "5_generation", "3_pre_retrieval", "4_retrieval"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.logging_config import get_logger
from config.settings import (
    GROQ_FALLBACK_MODEL,
    GROQ_PRIMARY_MODEL,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    MONGODB_COLLECTION,
    MONGODB_DB,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SUPPORTED_TICKERS,
    HYBRID_TOP_K,
)

from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from async_guardrail import AsyncGuardrail, SemanticCache
from generator import FinancialRAGGenerator, format_context_xml
from hybrid_chunker import _make_chunk_id
from hybrid_search import HybridSearchEngine
from intent_router import (
    RuleBasedIntentRouter,
    SECURITY_VIOLATION_RESPONSE,
)
from orchestrator import PreRetrievalOrchestrator
from post_retrieval import PostRetrievalPipeline
from table_shield import TableShield

logger = get_logger("pipeline.orchestrator")

# Parser table placeholder (kept in text chunks by the chunker) that must be
# resolved back to its raw MongoDB table chunk before LLM context formatting.
_TABLE_PLACEHOLDER_RE = re.compile(r"%%TABLE_(\d+)%%")

# Tolerant marker variants: the ingestion section-label extraction truncates
# headers at 80 chars, so section strings can carry UNTERMINATED fragments
# (e.g. "...STATEMENTS %%TABLE_"). The pipeline also splices inlined tables in
# behind synthetic "[TABLE N]" labels. These are internal only and must never
# reach the end user.
_TABLE_FRAGMENT_RE = re.compile(r"%%TABLE_[0-9_]*")
_INLINE_TABLE_MARKER_RE = re.compile(r"\[\s*TABLE\s+\d+\s*\]")


def _strip_internal_markers(text: str) -> str:
    """Remove internal parser/inline table markers from user-visible strings.

    Internal representations that must never appear in the final answer or in
    source chips: the ingestion parser's ``%%TABLE_N%%`` placeholders (which
    survive into stored section labels / text) and the synthetic ``[TABLE N]``
    labels inserted when placeholders are inlined. Stripping preserves the
    natural-language content while purging implementation artifacts.
    """
    if not text:
        return text
    cleaned = _TABLE_FRAGMENT_RE.sub("", str(text))
    cleaned = cleaned.replace("%%", "")  # drop the placeholder closer "%%"
    cleaned = _INLINE_TABLE_MARKER_RE.sub("", cleaned)
    # Collapse any whitespace exposed by marker removal and trim edges.
    cleaned = " ".join(cleaned.split())
    return cleaned


def _context_section_key(doc: dict) -> tuple:
    """Stable dedup identity for a context document.

    Distinct tickers are never collapsed -- each queried company needs its own
    context slot. Within a ticker, documents sharing a real section label are
    treated as siblings (the near-duplicate "Item 8" / "PART II" pairs) and
    collapse to the highest-ranked one. Documents with only the generic
    "General" label fall back to a content fingerprint so genuinely different
    chunks are preserved.
    """
    meta = doc.get("metadata") or {}
    ticker = str(meta.get("ticker") or "UNKNOWN")
    section = _strip_internal_markers(str(meta.get("section") or ""))
    norm_section = re.sub(r"[^a-z0-9]+", " ", section.lower()).strip(" ")
    if not norm_section or norm_section == "general":
        content = re.sub(
            r"[^a-z0-9]+", " ", str(doc.get("text") or "").lower()
        ).strip(" ")
        return (ticker, "content", content)
    return (ticker, "section", norm_section)

# Defensive caps for table-placeholder inlining: a single text chunk can
# reference dozens of tables, so we bound per-table and per-document inlined
# text to keep the LLM context within model limits (primary 8b TPM cap,
# fallback 131k context) and avoid catastrophic generation failures.
MAX_INLINE_TABLE_CHARS = 8000
MAX_INLINE_CHARS_PER_DOC = 24000

# Context-budget controls for the post-retrieval batch path. The retriever
# can surface huge chunks (10-30K chars each, ~20K+ tokens total) that exceed
# every Groq free-tier TPM cap (8b=6000, gpt-oss=8000, 70b=12000), which makes
# generation always fail with 413. We keep the top three documents, inject the
# filing's financial-statement tables (which hold the figures the retriever
# often misses), and cap the total context so generation fits the primary
# model budget. Table selection is query-aware: the fiscal year is taken from
# the question, and the specific tables injected (income statement,
# product/segment detail, ratio/metrics) are marker-matched against the
# filing's own table chunks in MongoDB. Per-slot caps are sized from measured
# offsets of the ground-truth figures so truncation never cuts a needed value.
# Injected tables are run through `_compact_table_text` first: the pandas pivot
# grid (repeated labels, "nan" padding, alignment rows) is ~10x larger than the
# readable rows, so compacting keeps the full statement (deep EPS rows included)
# inside a fraction of the token budget.
MAX_CONTEXT_DOC_CHARS = 1500
MAX_INCOME_TABLE_CHARS = 8000
MAX_COMPACT_INCOME_CHARS = 5000
MAX_PRODUCT_TABLE_CHARS = 5500
MAX_METRICS_TABLE_CHARS = 2000
MAX_TOTAL_CONTEXT_CHARS = 12000
# Size bands for category selection: skip both tiny fragment tables and the
# giant MD&A-detail tables that would drown the budget without the figures.
_PRODUCT_TABLE_BAND = (2500, 5500)
_COMPACT_INCOME_BAND = (3000, 6000)
_METRICS_TABLE_BAND = (1000, 3000)
_INCOME_CATEGORY_MARKERS = {
    "revenue": ("total net sales", "total revenues", "total revenue", "net revenue"),
    "income": ("net income", "operating income", "income from operations"),
    "pershare": ("earnings per share", "per share", "net income per share"),
}
_PRODUCT_SEGMENT_MARKERS = (
    "iphone", "ipad", "mac", "services", "wearables",
    "intelligent cloud", "more personal computing",
    "productivity and business processes", "gaming",
)
# Query terms that gate product/segment-table injection.
_PRODUCT_QUERY_TERMS = (
    "iphone", "ipad", "mac", "services", "wearables", "segment", "product",
    "intelligent cloud", "more personal computing",
    "productivity and business processes", "gaming",
)
# Query terms that gate cash-flow statement injection.
_CASH_FLOW_QUERY_TERMS = (
    "cash flow", "cash generated", "operating cash", "net cash",
    "free cash", "cash from operations",
)
# Markers used to identify cash-flow statement table chunks in MongoDB.
_CASH_FLOW_MARKERS = (
    "net cash provided by operating",
    "cash flows from operating",
    "operating activities",
    "cash flows from",
    "net cash",
)
_CASH_FLOW_TABLE_BAND = (2000, 8000)
MAX_CASH_FLOW_TABLE_CHARS = 6000
# Segment-specific query terms that trigger a broader segment-table search.
_SEGMENT_QUERY_TERMS = (
    "intelligent cloud", "more personal computing",
    "productivity and business processes", "segment revenue",
    "which segment", "segment generated",
)
_SEGMENT_MARKERS = (
    "intelligent cloud", "more personal computing",
    "productivity and business processes",
)
MAX_SEGMENT_TABLE_CHARS = 4000
# Pull the fiscal year out of the question so table injection aligns with the
# query's intent even when the retriever surfaced a chunk from another year.
_QUERY_YEAR_RE = re.compile(
    r"(?:FY|fiscal(?:\s+year)?)\s*[-]?\s*(\d{4})", re.IGNORECASE
)
# Any fiscal year (calendar mention) mentioned anywhere in the question, e.g.
# "fiscal year 2024 (ended September 28, 2024)". Latest mention wins so the
# comparison year in "between 2023 and 2024" resolves to the later filing.
_ANY_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# Company name / ticker -> canonical ticker for query-level grounding. The
# hybrid path pre-filters candidates by ticker, so an MSFT question must never
# let an Apple income-statement table win the top slot (which would make the
# supplementary-table injection target the wrong filing).
_QUERY_TICKER_MAP = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
}

# Canonical display names for generation-time grounding. Used only to make a
# memory-resolved follow-up question explicit to the LLM (never to fabricate
# figures); values mirror the ticker/name pairs already present above.
_TICKER_DISPLAY_NAME = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


def _resolved_generation_query(
    user_query: str,
    resolved_ticker: str | None,
    inherited_year: str | None,
) -> str:
    """Annotate a memory-resolved follow-up with its resolved company/year.

    Coreference resolution only scopes RETRIEVAL; the raw question text (e.g.
    "the Second company ... that same fiscal year") is still handed to the
    generator as-is, so the model concludes the context contains "no second
    company" and refuses -- even when the value-bearing table reached the LLM.
    Appending the already-resolved entity keeps the change generic (no
    hardcoded company, year, or value): the generator receives the same
    explicit grounding the retriever already applied.
    """
    if not resolved_ticker:
        return user_query
    name = _TICKER_DISPLAY_NAME.get(resolved_ticker, resolved_ticker)
    suffix = f"Resolved from the earlier conversation: {name} ({resolved_ticker})"
    if inherited_year:
        suffix += f", fiscal year {inherited_year}"
    return f"{user_query} ({suffix}.)"

# Cross-company intent signals: when the user asks about "all companies",
# "compare", "both", etc. without naming specific tickers, the pipeline
# should auto-expand the filter to every supported ticker.
_CROSS_COMPANY_PATTERNS = re.compile(
    r"\b("
    r"all\s+(?:companies|tickers|stocks|tech|firms)"
    r"|all\s+\d+"
    r"|(?:compare|comparison|versus|vs\.?|between)\s+"
    r"|across\s+(?:all|every|the)\s+"
    r"|both\s+companies"
    r"|each\s+company"
    r"|every\s+company"
    r"|industry\s+(?:wide|comparison|trend|analysis)"
    r"|sector\s+(?:wide|comparison|trend|analysis)"
    r")\b",
    re.IGNORECASE,
)
# Comparison keywords that signal cross-ticker retrieval intent even when the
# individual company names are not spelled out (Issue 3). When any of these
# match, every single-ticker Qdrant payload restriction is dropped.
_COMPARISON_INTENT_RE = re.compile(
    r"\b(?:compar\w*|versus|vs\.?|across|differences?|between)\b",
    re.IGNORECASE,
)
# Content signals used to prefer the *value-bearing* table within a category
# over near-duplicate narrative/MD&A tables that merely mention the label.
_NET_INCOME_ROW_RE = re.compile(r"net income\D{0,120}\d[\d,]{3,}", re.IGNORECASE)
_PER_SHARE_ROW_RE = re.compile(r"(?:basic|diluted)?\s*earnings per share\D{0,80}\d", re.IGNORECASE)
_DECIMAL_PERCENT_RE = re.compile(r"\d+\.\d+\s*(?:\|\s*)?%")
# Empty-cell sentinels emitted by the pandas->markdown exporter.
_EMPTY_TABLE_CELLS = {"nan", "none", ""}


def _compact_table_text(text: str) -> str:
    """
    Shrink a markdown table chunk to its readable rows.

    The exporter pivots each statement row into a wide grid: repeated column
    labels ("| Unnamed: 1 | ..."), "nan" padding and "| :--- |" alignment rows
    inflate the token count several-fold (measured ~10x on the income tables).
    This drops alignment rows, empty cells, "Unnamed:" column headers and
    consecutive duplicate labels while keeping every numeric value in its
    original row/column order, so a compacted statement still carries the exact
    figures the questions ask for (verified against every ground-truth value).
    """
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        kept: list[str] = []
        for cell in cells:
            if not cell or cell.lower() in _EMPTY_TABLE_CELLS:
                continue
            if cell.lower().startswith("unnamed:"):
                continue
            if kept and kept[-1] == cell:
                continue
            kept.append(cell)
        if kept:
            lines.append(" | ".join(kept))
    return "\n".join(lines)


class FinancialRAGPipeline:
    """
    Master orchestrator that connects vector retrieval, MongoDB enrichment,
    LLM generation, guardrail verification, semantic caching, and MLflow logging
    into a single unified query interface.

    Module 4 wiring (opt-in `enable_hybrid_retrieval` + `enable_post_retrieval`):
        Hybrid Search (40 chunks)
            -> Cross-Encoder Reranker (Top 8)
            -> Table Shield (Async Clean)
            -> Cylinder Reorder (best at head, 2nd-best at tail)
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        mongo_db: str = MONGODB_DB,
        mongo_collection: str = MONGODB_COLLECTION,
        qdrant_collection: str = QDRANT_COLLECTION,
        primary_model: str = GROQ_PRIMARY_MODEL,
        fallback_model: str = GROQ_FALLBACK_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int = LLM_SEED,
        top_k: int = 3,
        enable_cache: bool = True,
        enable_guardrail: bool = True,
        enable_pre_retrieval: bool = False,
        enable_hybrid_retrieval: bool = False,
        enable_post_retrieval: bool = False,
    ):
        """
        Initialize all sub-components of the RAG pipeline.

        Args:
            qdrant_path: Persistent path for Qdrant vector storage.
            mongo_db: MongoDB database name.
            mongo_collection: MongoDB collection name.
            qdrant_collection: Qdrant collection name.
            primary_model: Groq primary LLM model name.
            fallback_model: Groq fallback LLM model name.
            temperature: LLM temperature for generation.
            max_tokens: Maximum tokens in LLM response.
            seed: Random seed for deterministic generation.
            top_k: Default number of documents to retrieve.
            enable_cache: Whether to use Redis semantic cache.
            enable_guardrail: Whether to run async guardrail checks.
            enable_pre_retrieval: Whether to enable the Module 3 pre-retrieval
                stage (intent routing + conditional query expansion). Off by
                default so existing callers/tests keep the legacy flow.
            enable_hybrid_retrieval: Whether to route retrieval through the
                Module 4 HybridSearchEngine (strict Qdrant pre-filtering +
                parallel dense/BM25 + RRF fusion) instead of the legacy
                single-pass Qdrant search. Off by default; legacy fallback
                mechanisms are fully preserved.
            enable_post_retrieval: Whether to enable the Module 4 post-retrieval
                pipeline (Cross-Encoder reranker -> Table Shield -> Cylinder
                reorder) on top of hybrid search. When enabled, the hybrid
                engine runs at HYBRID_TOP_K=40 and the post-retrieval stage
                returns the top-8 cleaned, table-shielded, cylinder-reordered
                chunks. Implies enable_hybrid_retrieval. Off by default.
        """
        # Retrieval components
        self._embedding_engine = EmbeddingEngine()
        self._qdrant_indexer = QdrantIndexer(
            path=qdrant_path,
            collection_name=qdrant_collection,
        )
        self._mongo_indexer = MongoDBIndexer(
            db_name=mongo_db,
            collection_name=mongo_collection,
        )

        # Generation components
        self._generator = FinancialRAGGenerator(
            primary_model=primary_model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        # Guardrail and caching
        self._enable_guardrail = enable_guardrail
        self._enable_cache = enable_cache
        self._guardrail = AsyncGuardrail() if enable_guardrail else None
        self._cache = (
            SemanticCache(embed_fn=self._embedding_engine.embed_single)
            if enable_cache
            else None
        )

        # Pre-Retrieval (Module 3): intent routing + conditional expansion.
        self._enable_pre_retrieval = enable_pre_retrieval
        self._pre_retrieval = (
            PreRetrievalOrchestrator(embed_fn=self._embedding_engine.embed)
            if enable_pre_retrieval
            else None
        )

        # Post-Retrieval (Module 4 Part 2): rerank -> table shield -> cylinder.
        # Requires hybrid retrieval (the 40-chunk feed). When enabled the hybrid
        # engine runs at the 40-chunk RRF cap instead of the legacy top_k so the
        # post-retrieval stage can re-rank down to its top-8 contract.
        self._enable_post_retrieval = enable_post_retrieval
        if enable_post_retrieval and not enable_hybrid_retrieval:
            logger.warning(
                "enable_post_retrieval requires enable_hybrid_retrieval -- "
                "forcing hybrid retrieval ON"
            )
            enable_hybrid_retrieval = True

        # Hybrid Retrieval (Module 4): strict pre-filtering + dense/BM25 + RRF.
        self._enable_hybrid_retrieval = enable_hybrid_retrieval
        self._hybrid_search = (
            HybridSearchEngine(
                qdrant_indexer=self._qdrant_indexer,
                mongo_indexer=self._mongo_indexer,
                embed_fn=self._embedding_engine.embed,
                top_k=HYBRID_TOP_K if enable_post_retrieval else top_k,
            )
            if enable_hybrid_retrieval
            else None
        )
        self._post_retrieval = (
            PostRetrievalPipeline(
                table_shield=TableShield(clean_enabled=False)
            )
            if enable_post_retrieval
            else None
        )
        if enable_post_retrieval:
            logger.info(
                "Post-retrieval enabled: hybrid search at top_k=%d "
                "(rerank -> table shield -> cylinder reorder)",
                HYBRID_TOP_K,
            )

        # Config
        self._top_k = top_k

        # Module 6 tracing: the raw chunks (post-reranker) fed to the LLM for
        # the most recent query(). Populated during query() so the offline
        # evaluation BatchRunner can capture reranker contexts for scoring.
        self._last_contexts: list[dict] = []

        logger.info(
            "FinancialRAGPipeline initialized: primary=%s fallback=%s "
            "top_k=%d cache=%s guardrail=%s pre_retrieval=%s hybrid_retrieval=%s "
            "            post_retrieval=%s",
            primary_model,
            fallback_model,
            top_k,
            enable_cache,
            enable_guardrail,
            enable_pre_retrieval,
            self._enable_hybrid_retrieval,
            self._enable_post_retrieval,
        )

    def warm_reranker(self) -> None:
        """Pre-load the post-retrieval cross-encoder (GPU) during startup so the
        first user query does not pay the one-time reranker model-load cost.

        Reuses the exact shared instance held by the post-retrieval pipeline
        (never a second model copy). Raises on failure so startup warm-up can
        report honestly that required warm-up did not complete.
        """
        if not (self._enable_post_retrieval and self._post_retrieval is not None):
            logger.info("Reranker warm-up skipped (post-retrieval disabled)")
            return
        self._post_retrieval.warm_up()

    def _retrieve_documents(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
        tickers: list[str] | None = None,
    ) -> list[dict]:
        """
        Two-phase retrieval: Qdrant vector search → MongoDB text enrichment.

        Phase 1: Embed query, search Qdrant with optional metadata pre-filtering.
        Phase 2: Fetch full raw_text from MongoDB for each retrieved chunk_id.

        Args:
            query: User query string.
            ticker: Optional ticker filter (e.g. 'AAPL').
            fiscal_year: Optional fiscal year filter (e.g. '2025').
            section: Optional SEC 10-K section filter (e.g. 'Item 7').
            top_k: Number of results to retrieve (default: self._top_k).
            tickers: Optional list of tickers for cross-company OR filtering.

        Returns:
            List of enriched document dicts ready for LLM context formatting.
        """
        k = top_k or self._top_k

        # Phase 1: Qdrant vector search
        query_embedding = self._embedding_engine.embed_single(query)
        qdrant_results = self._qdrant_indexer.search(
            query_embedding,
            top_k=k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
            tickers=tickers,
        )

        if not qdrant_results:
            logger.warning("Vector search returned 0 results for query: %s", query[:80])
            return []

        # Phase 2: MongoDB text enrichment
        return self._enrich_documents(qdrant_results)

    def _retrieve_documents_multi(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        metadata_filter: dict,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Multi-query retrieval with metadata pre-filtering (Route 3 / REWRITE).

        Each (possibly expanded) query is searched against Qdrant within the
        isolated pre-filtered scope; results are merged per chunk_id keeping the
        best score, then enriched from MongoDB.

        Args:
            queries: Retrieval queries from the pre-retrieval stage.
            embeddings: Pre-computed batch embeddings aligned with queries
                (empty when expansion was skipped / embedder unavailable).
            metadata_filter: Qdrant pre-filter dict (ticker/fiscal_year/section).
            top_k: Number of final documents to keep (default: self._top_k).

        Returns:
            List of enriched document dicts ready for LLM context formatting.
        """
        k = top_k or self._top_k
        ticker = metadata_filter.get("ticker")
        tickers = metadata_filter.get("tickers")
        fiscal_year = metadata_filter.get("fiscal_year")
        section = metadata_filter.get("section")

        # Qdrant payload years are stored as strings; normalise the int from
        # the metadata extractor so the MatchValue filter actually matches.
        if isinstance(fiscal_year, int):
            fiscal_year = str(fiscal_year)

        if not queries:
            logger.warning("Pre-retrieval returned no queries -- skipping retrieval")
            return []

        merged: dict[str, dict] = {}
        for index, query in enumerate(queries):
            # Reuse the single batch embed when available; otherwise embed.
            if index < len(embeddings) and embeddings[index]:
                query_embedding = embeddings[index]
            else:
                query_embedding = self._embedding_engine.embed_single(query)

            results = self._qdrant_indexer.search(
                query_embedding,
                top_k=k,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                tickers=tickers,
            )
            for r in results:
                cid = r["chunk_id"]
                if cid not in merged or r["score"] > merged[cid]["score"]:
                    merged[cid] = r

        if not merged:
            logger.warning(
                "Multi-query vector search returned 0 results "
                "(filter=%s, queries=%d)",
                metadata_filter,
                len(queries),
            )
            return []

        best = sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:k]
        return self._enrich_documents(best)

    def _enrich_documents(self, qdrant_results: list[dict]) -> list[dict]:
        """Fetch full raw_text from MongoDB for the given Qdrant results."""
        chunk_ids = [r["chunk_id"] for r in qdrant_results]
        mongo_docs = self._mongo_indexer.get_chunks_by_ids(chunk_ids)

        documents = []
        for r in qdrant_results:
            cid = r["chunk_id"]
            payload = r.get("payload", {})
            mongo_doc = mongo_docs.get(cid, {})
            raw_text = mongo_doc.get("raw_text", "")

            if not raw_text:
                logger.debug("Chunk %s missing raw_text in MongoDB, skipping", cid)
                continue

            documents.append({
                "text": raw_text,
                "metadata": {
                    "ticker": payload.get("ticker", "UNKNOWN"),
                    "fiscal_year": payload.get("fiscal_year", "UNKNOWN"),
                    "section": payload.get("section", "General"),
                    "contains_table": payload.get("contains_table", False),
                    "page_number": payload.get("page_number", "N/A"),
                },
                "chunk_id": cid,
                "score": r["score"],
            })

        logger.info(
            "Retrieved %d/%d documents (Qdrant: %d, MongoDB enriched: %d)",
            len(documents),
            len(qdrant_results),
            len(qdrant_results),
            len(mongo_docs),
        )
        return documents

    # ------------------------------------------------------------------
    # Hybrid retrieval (Module 4) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_hybrid_documents(hybrid_results: list[dict]) -> list[dict]:
        """
        Convert HybridSearchEngine / post-retrieval output to the pipeline schema.

        The hybrid engine returns chunks with `rrf_score` and a rich `metadata`
        dict; the downstream context formatter + generator consume the legacy
        `{text, metadata, chunk_id, score}` shape, so we map fields 1:1 while
        retaining the COMPLETE metadata set (ticker, fiscal_year, section,
        doc_type, contains_table, page_number, source_file).
        """
        documents = []
        for r in hybrid_results:
            meta = r.get("metadata", {}) or {}
            documents.append(
                {
                    "text": r.get("text", ""),
                    "metadata": {
                        "ticker": meta.get("ticker", "UNKNOWN"),
                        "fiscal_year": meta.get("fiscal_year", "UNKNOWN"),
                        "section": meta.get("section", "General"),
                        "doc_type": meta.get("doc_type"),
                        "contains_table": meta.get("contains_table", False),
                        "page_number": meta.get("page_number", "N/A"),
                        "source_file": meta.get("source_file"),
                    },
                    "chunk_id": r.get("chunk_id"),
                    "score": r.get("rrf_score", 0.0),
                    "rerank_score": r.get("rerank_score"),
                    "cleaned_text": r.get("cleaned_text"),
                }
            )
        return documents

    def _resolve_table_placeholders(self, documents: list[dict]) -> list[dict]:
        """
        Inline the table chunks referenced by ``%%TABLE_N%%`` placeholders.

        Retrieval text chunks keep the parser table placeholders; the actual
        tables live as separate ``chunk_type='table'`` chunks in MongoDB. Each
        placeholder maps deterministically (``_make_chunk_id``) to its table
        chunk id, which is batch-fetched and spliced into the document text so
        numeric figures reach the LLM context instead of a bare placeholder.
        """
        resolved: list[dict] = []
        for doc in documents:
            text = doc.get("text", "")
            if "%%TABLE_" not in text:
                resolved.append(doc)
                continue

            meta = doc.get("metadata", {}) or {}
            ticker = meta.get("ticker")
            fiscal_year = meta.get("fiscal_year")
            source_file = meta.get("source_file")

            # Legacy enrichment omits source_file; recover it from MongoDB.
            if not source_file:
                mongo_doc = self._mongo_indexer.get_chunk(doc.get("chunk_id") or "")
                if mongo_doc:
                    source_file = mongo_doc.get("source_file")
                    if not fiscal_year or fiscal_year == "UNKNOWN":
                        fiscal_year = mongo_doc.get("fiscal_year")

            if not (ticker and fiscal_year and source_file):
                logger.warning(
                    "Cannot resolve table placeholders for chunk %s "
                    "(ticker=%s year=%s source=%s) -- leaving as-is",
                    doc.get("chunk_id"),
                    ticker,
                    fiscal_year,
                    source_file,
                )
                resolved.append(doc)
                continue

            ref_ids = {
                _make_chunk_id(
                    str(ticker), str(fiscal_year), "tbl", source_file, int(idx)
                )
                for idx in _TABLE_PLACEHOLDER_RE.findall(text)
            }
            tables = self._mongo_indexer.get_chunks_by_ids(list(ref_ids))
            budget = MAX_INLINE_CHARS_PER_DOC

            def _replace(match: re.Match) -> str:
                nonlocal budget
                idx = int(match.group(1))
                table_id = _make_chunk_id(
                    str(ticker), str(fiscal_year), "tbl", source_file, idx
                )
                table_text = (tables.get(table_id) or {}).get("raw_text", "")
                if not table_text or budget <= 0:
                    if not table_text:
                        logger.warning(
                            "Table chunk %s not found -- leaving placeholder",
                            table_id,
                        )
                    return match.group(0)
                snippet = table_text[: min(MAX_INLINE_TABLE_CHARS, budget)]
                budget -= len(snippet)
                return f"\n[TABLE {idx}]\n{snippet}"

            doc["text"] = _TABLE_PLACEHOLDER_RE.sub(_replace, text)
            resolved.append(doc)

        logger.info(
            "Table placeholder resolution: %d/%d documents carried placeholders",
            sum(1 for d in documents if "%%TABLE_" in d.get("text", "")),
            len(documents),
        )
        return resolved

    @staticmethod
    def _query_fiscal_year(query: str) -> str | None:
        """Best-effort fiscal year from the question (latest mention wins)."""
        years = [int(y) for y in _QUERY_YEAR_RE.findall(query or "")]
        return str(max(years)) if years else None

    @staticmethod
    def _detect_ticker(query: str) -> str | None:
        """Detect the company referenced by the question (name or ticker)."""
        lowered = (query or "").lower()
        for token, ticker in _QUERY_TICKER_MAP.items():
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                return ticker
        return None

    @staticmethod
    def _detect_all_tickers(query: str) -> list[str]:
        """Detect ALL companies referenced in the question.

        Returns a deduplicated list in detection order.
        Used for cross-company comparison queries.

        If the query expresses cross-company intent (e.g. "compare all 3",
        "industry trends across all companies") but does not explicitly name
        every ticker, the full ``SUPPORTED_TICKERS`` list is returned so the
        Qdrant filter expands to cover all companies.
        """
        found: list[str] = []
        lowered = (query or "").lower()
        for token, ticker in _QUERY_TICKER_MAP.items():
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                if ticker not in found:
                    found.append(ticker)

        # Cross-company intent detected but no (or not all) tickers mentioned
        # explicitly — expand to every supported ticker.
        if _CROSS_COMPANY_PATTERNS.search(lowered):
            from config.settings import SUPPORTED_TICKERS
            for tk in SUPPORTED_TICKERS:
                if tk not in found:
                    found.append(tk)

        return found

    @staticmethod
    def _mentioned_tickers(query: str) -> list[str]:
        """Tickers explicitly named in the query text (no pattern expansion)."""
        found: list[str] = []
        lowered = (query or "").lower()
        for token, ticker in _QUERY_TICKER_MAP.items():
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                if ticker not in found:
                    found.append(ticker)
        return found

    @classmethod
    def _has_multi_ticker_intent(cls, query_str: str) -> bool:
        """Detect cross-ticker / comparison intent in a raw query string.

        True when the query names two or more distinct companies (e.g.
        "AAPL vs MSFT", "Apple and Microsoft") or contains comparison
        keywords ("compare", "versus", "vs", "across", "difference").
        """
        if len(cls._mentioned_tickers(query_str)) >= 2:
            return True
        return bool(_COMPARISON_INTENT_RE.search(query_str or ""))

    def _apply_cross_ticker_bypass(
        self,
        query_str: str,
        metadata_filter: dict,
    ) -> dict:
        """Strip every ticker restriction when multi-ticker intent is present.

        The frontend always sends its currently-selected company as the single
        ``ticker`` scope; for comparison questions that scope would hide two
        thirds of the corpus. This returns a copy of ``metadata_filter`` with
        the ``ticker``/``tickers`` keys removed so Qdrant executes WITHOUT any
        active-ticker metadata restriction (fiscal_year/section are kept).
        """
        if self._has_multi_ticker_intent(query_str):
            logger.info(
                "Cross-ticker retrieval bypass: comparison/multi-company intent "
                "detected in query %r -- Qdrant ticker filter disabled",
                (query_str or "")[:100],
            )
            return {
                key: value
                for key, value in (metadata_filter or {}).items()
                if key not in ("ticker", "tickers")
            }
        return metadata_filter or {}

    def _resolve_memory_coreference(
        self,
        user_query: str,
        ticker: Optional[str],
        explicit_tickers: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Resolve single-company coreference for follow-up turns when the full
        Module 3 pre-retrieval stage is disabled.

        The legacy flow never resolves ordinal/subject references such as
        "first company", "the second company" or "its"/"their" that depend on
        the prior turns held in the session's conversation history, so follow-up
        memory answers drifted to a default fiscal year / wrong company. When a
        genuine coreferent is present AND a single company is deterministically
        resolved from history, this returns the resolved ticker + inherited
        fiscal year so retrieval is scoped to that one company.

        Returns None when there is nothing to resolve (pre-retrieval is active,
        no history, no coreferent, or the reference does not resolve to a single
        supported company), leaving existing flows completely unchanged.
        """
        if self._enable_pre_retrieval:
            return None
        history = self._get_conversation_history()
        if not history:
            return None
        if not RuleBasedIntentRouter._has_coreferent(user_query):
            return None
        resolved = RuleBasedIntentRouter(
            supported_tickers=list(SUPPORTED_TICKERS)
        ).route(user_query, history=history)
        single = resolved.ticker
        if not single or single == "ALL":
            return None
        return {
            "ticker": single,
            "fiscal_year": (
                str(resolved.fiscal_year) if resolved.fiscal_year is not None else None
            ),
            "tickers": resolved.tickers,
        }

    def _comparison_ticker_pool(
        self,
        query_str: str,
        explicit_tickers: Optional[list[str]] = None,
    ) -> list[str]:
        """Tickers that must EACH contribute context to a comparison query.

        Merges the explicit ``tickers`` request field with the companies named
        in the question text (``_detect_all_tickers`` also expands "compare
        all companies" phrasing to the full supported universe). Capped at 4
        sub-retrievals to bound latency.
        """
        pool: list[str] = []
        for tk in [*(explicit_tickers or []), *self._detect_all_tickers(query_str)]:
            if tk and tk not in pool:
                pool.append(tk)
        return pool[:4]

    def _balanced_ticker_subretrievals_sync(
        self,
        query: str,
        tickers: list[str],
        fiscal_year: str | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """One scoped retrieval PER ticker, merged so no company is starved.

        A single unfiltered vector search over a multi-company question is
        dominated by whichever filing's wording best matches the query text,
        collapsing comparison answers onto one dominant ticker ("no financial
        data for X present"). Running a ``{ticker: t}``-scoped sub-retrieval
        per company guarantees every queried ticker contributes up to
        ``top_k`` chunks; results are merged deduplicated by chunk_id.

        When both hybrid retrieval AND post-retrieval are enabled and more than
        one ticker is requested, the per-ticker searches are run CONCURRENTLY and
        the GPU cross-encoder re-ranks all tickers in ONE batched predict
        (`_batch_multi_ticker_retrieval`) -- the dominant ~4-5s-per-ticker CUDA
        rerank that previously serialized dominates the warm retrieval cost.
        The non-hybrid path and the single-ticker path are unchanged.
        """
        tickers = list(tickers or [])[:4]

        use_batch = (
            self._enable_hybrid_retrieval
            and self._hybrid_search is not None
            and self._enable_post_retrieval
            and self._post_retrieval is not None
            and len(tickers) > 1
        )
        if use_batch:
            return self._batch_multi_ticker_retrieval(
                query, tickers, fiscal_year, top_k
            )

        merged: list[dict] = []
        seen: set[str] = set()
        for tk in tickers:
            try:
                if self._enable_hybrid_retrieval and self._hybrid_search is not None:
                    hybrid_filter: dict = {"ticker": tk}
                    if fiscal_year:
                        hybrid_filter["fiscal_year"] = fiscal_year
                    docs = self._run_hybrid_retrieval(query, [query], hybrid_filter)
                else:
                    docs = self._retrieve_documents(
                        query,
                        ticker=tk,
                        fiscal_year=fiscal_year,
                        top_k=top_k,
                    )
            except Exception as exc:
                logger.warning("Balanced sub-retrieval failed for %s: %s", tk, exc)
                continue
            for doc in docs:
                cid = str(doc.get("chunk_id") or "")
                if cid and cid in seen:
                    continue
                seen.add(cid)
                merged.append(doc)
        logger.info(
            "Balanced multi-ticker retrieval: %d chunks across %d tickers (%s)",
            len(merged), len(tickers[:4]), ",".join(tickers[:4]),
        )
        return merged

    def _batch_multi_ticker_retrieval(
        self,
        query: str,
        tickers: list[str],
        fiscal_year: str | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """Loop-safe wrapper around the async batched multi-ticker retrieval."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if threading.current_thread() is threading.main_thread():
                return asyncio.get_event_loop().run_until_complete(
                    self._batch_multi_ticker_retrieval_core(
                        query, tickers, fiscal_year, top_k
                    )
                )
            return asyncio.run(
                self._batch_multi_ticker_retrieval_core(
                    query, tickers, fiscal_year, top_k
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                self._batch_multi_ticker_retrieval,
                query,
                tickers,
                fiscal_year,
                top_k,
            ).result()

    async def _batch_multi_ticker_retrieval_core(
        self,
        query: str,
        tickers: list[str],
        fiscal_year: str | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """Concurrent per-ticker searches + single batched GPU rerank.

        The per-ticker hybrid searches run concurrently (their MongoDB BM25
        fetch and Qdrant dense I/O overlap; the shared CUDA embedding call
        serializes on a lock). The cross-encoder rerank -- the dominant cost --
        runs ONCE over all tickers' chunks combined, then per-ticker selection,
        rescue guarantee, table shield and cylinder reorder are applied with the
        identical semantics as the sequential path (`aprocess_scored`). Results
        preserve ticker order and deduplicate by chunk_id.
        """
        tks = list(tickers)[:4]

        def make_filter(tk: str) -> dict:
            f: dict = {"ticker": tk}
            if fiscal_year:
                f["fiscal_year"] = fiscal_year
            return f

        async def search_one(tk: str) -> tuple[str, list[dict]]:
            raw = await self._hybrid_search.asearch([query], make_filter(tk))
            return tk, raw

        results = await asyncio.gather(
            *(search_one(tk) for tk in tks),
            return_exceptions=True,
        )

        raw_by_tk: dict[str, list[dict]] = {}
        for (tk, _f), res in zip([(t, make_filter(t)) for t in tks], results):
            if isinstance(res, Exception):
                logger.warning("Balanced sub-retrieval failed for %s: %s", tk, res)
                continue
            # search_one returns (ticker, raw_chunks); keep the raw chunk list.
            raw_by_tk[tk] = res[1]

        if not raw_by_tk:
            return []

        # 2. Single batched GPU predict over ALL tickers' raw chunks.
        flat: list[tuple[str, dict]] = []
        for tk in tks:
            for c in raw_by_tk.get(tk, []):
                flat.append((tk, c))
        scores: list[float] = []
        if flat:
            scores = self._post_retrieval._reranker.predict_scores(
                query, [c for _tk, c in flat]
            )

        # 3. Per-ticker post-processing + merge, preserving ticker order.
        merged: list[dict] = []
        seen: set[str] = set()
        idx = 0
        for tk in tks:
            raw_chunks = raw_by_tk.get(tk, [])
            n = len(raw_chunks)
            tk_scores = scores[idx : idx + n]
            idx += n
            try:
                post = await self._post_retrieval.aprocess_scored(
                    query, raw_chunks, tk_scores
                )
            except Exception as exc:
                logger.warning("Balanced post-retrieval failed for %s: %s", tk, exc)
                post = []
            docs = self._adapt_hybrid_documents(post)
            for doc in docs:
                cid = str(doc.get("chunk_id") or "")
                if cid and cid in seen:
                    continue
                seen.add(cid)
                merged.append(doc)

        logger.info(
            "Balanced multi-ticker retrieval (batched): %d chunks across %d tickers (%s)",
            len(merged),
            len(tks),
            ",".join(tks),
        )
        return merged

    async def _balanced_ticker_subretrievals(
        self,
        query: str,
        tickers: list[str],
        fiscal_year: str | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """Async wrapper -- runs the blocking per-ticker searches off-loop."""
        return await asyncio.to_thread(
            self._balanced_ticker_subretrievals_sync,
            query,
            tickers,
            fiscal_year,
            top_k,
        )

    @staticmethod
    def _detect_fiscal_year(query: str) -> str | None:
        """Best-effort fiscal year from any year mentioned (latest wins)."""
        years = [int(y) for y in _ANY_YEAR_RE.findall(query or "")]
        if not years:
            return None
        return str(max(years))

    @staticmethod
    def _select_marked_table(
        tables: list[dict], markers: tuple[str, ...]
    ) -> dict | None:
        """Largest table chunk whose text contains any of the given markers."""
        matches = [
            c for c in tables
            if any(m in c.get("raw_text", "").lower() for m in markers)
        ]
        if not matches:
            return None
        return max(matches, key=lambda c: len(c.get("raw_text", "")))

    @staticmethod
    def _select_band_table(
        tables: list[dict],
        markers: tuple[str, ...],
        min_chars: int,
        max_chars: int,
    ) -> dict | None:
        """
        Largest table chunk whose size is within [min_chars, max_chars] and
        whose text contains any marker. The size band avoids both giant
        MD&A-detail tables (noise, truncation risk) and tiny fragment tables
        (usually partial or irrelevant).
        """
        matches = [
            c for c in tables
            if min_chars <= len(c.get("raw_text", "")) <= max_chars
            and any(m in c.get("raw_text", "").lower() for m in markers)
        ]
        if not matches:
            return None
        return max(matches, key=lambda c: len(c.get("raw_text", "")))

    @staticmethod
    def _income_detail_score(text: str) -> int:
        """Count value-bearing net-income and EPS rows (labels followed by a figure)."""
        return len(_NET_INCOME_ROW_RE.findall(text)) + len(_PER_SHARE_ROW_RE.findall(text))

    @staticmethod
    def _percent_value_score(text: str) -> int:
        """Count decimal percentage values (e.g. 46.9%) that carry the figure."""
        return len(_DECIMAL_PERCENT_RE.findall(text))

    def _select_income_table(self, tables: list[dict]) -> dict:
        """
        Pick the canonical consolidated income statement.

        Requires the revenue, income AND per-share marker families so the full
        statement (net sales through EPS) is captured; falls back progressively
        to weaker matches, then to the largest table overall.
        """
        groups = _INCOME_CATEGORY_MARKERS
        for selectors in (
            (groups["revenue"], groups["income"], groups["pershare"]),
            (groups["revenue"], groups["income"]),
            # Loose-revenue tier: NVDA-style filings label the top line just
            # "Revenue" (no "total ... sales/revenue" phrase), so without this
            # tier the selector degrades to the largest-table fallback and
            # injects a non-financial chunk -- comparison answers then report
            # that company's figures as "not available".
            (("revenue",), groups["income"]),
            (groups["revenue"],),
            (groups["income"],),
        ):
            matches = [
                c for c in tables
                if all(any(m in c.get("raw_text", "").lower() for m in sel) for sel in selectors)
            ]
            if matches:
                return max(matches, key=lambda c: len(c.get("raw_text", "")))
        return max(tables, key=lambda c: len(c.get("raw_text", "")))

    def _augment_context(self, documents: list[dict], query: str = "", all_companies: bool = False) -> list[dict]:
        """
        Bound the LLM context and inject the filing's financial-statement tables.

        The hybrid+post-retrieval path can return huge chunks whose combined
        token count exceeds every available Groq TPM cap (generation then 413s
        into a refusal). The numeric ground truths for these queries live in the
        filing's table chunks rather than the retrieved text chunks, so this
        step keeps the top-ranked retrieved chunks that fit the shared
        context budget (trimmed, unresolved table placeholders stripped) and appends:

        1. The income statement (revenue+income+per-share matched,
           ``MAX_INCOME_TABLE_CHARS`` cap). When it exceeds the cap and the
           question targets per-share or net-income figures, a compact income
           summary is injected too so the late (EPS) rows survive truncation.
        2. The product/segment table (query-gated, marker-matched,
           ``MAX_PRODUCT_TABLE_CHARS`` cap).
        3. A ratio/metrics table (query-gated on gross-margin/gaming,
           ``MAX_METRICS_TABLE_CHARS`` cap).

        The fiscal year is taken from the question (latest mention) so the
        correct filing's tables are used even when the retriever surfaced a
        chunk from a different year. The total is budgeted to fit
        ``MAX_TOTAL_CONTEXT_CHARS`` (the primary-model token budget).

        The prepared documents are finally ordered by query-specificity so the
        answer-bearing table (product/segment, gross-margin, compact income)
        ranks first, ahead of the income statement and the retrieved text.
        """
        if not documents:
            return documents

        budget = MAX_TOTAL_CONTEXT_CHARS

        top = documents[0]
        prepared: list[dict] = []

        # Single-company retrieval keeps the top-ranked chunks until the shared
        # context budget is consumed. But for cross-entity / multi-ticker
        # retrieval the merged list is ordered by
        # ticker (e.g. AAPL first), so a naive documents[:3] would keep ONLY
        # the first company's narrative and starve the others of context --
        # producing Apple-only answers. In that case we keep a balanced number
        # of text chunks per ticker so every queried company reaches the LLM.
        distinct_tickers: list[str] = []
        for doc in documents:
            tk = (doc.get("metadata") or {}).get("ticker")
            if tk and tk not in distinct_tickers:
                distinct_tickers.append(tk)
        multi_entity = all_companies or len(distinct_tickers) > 1

        if multi_entity:
            per_ticker_cap = 2
            per_ticker_seen: dict[str, int] = {}
            seen_sections: set[tuple] = set()
            for doc in documents:
                tk = (doc.get("metadata") or {}).get("ticker") or "UNKNOWN"
                if per_ticker_seen.get(tk, 0) >= per_ticker_cap:
                    continue
                sec_key = _context_section_key(doc)
                if sec_key in seen_sections:
                    continue
                seen_sections.add(sec_key)
                text = _TABLE_PLACEHOLDER_RE.sub("", doc.get("text", ""))
                if len(text) > MAX_CONTEXT_DOC_CHARS:
                    text = text[:MAX_CONTEXT_DOC_CHARS]
                if not text:
                    continue
                if len(text) > budget:
                    text = text[:budget]
                if not text:
                    continue
                budget -= len(text)
                per_ticker_seen[tk] = per_ticker_seen.get(tk, 0) + 1
                doc["text"] = text
                prepared.append(doc)
        else:
            seen_sections: set[tuple] = set()
            for doc in documents:
                sec_key = _context_section_key(doc)
                if sec_key in seen_sections:
                    continue
                seen_sections.add(sec_key)
                text = _TABLE_PLACEHOLDER_RE.sub("", doc.get("text", ""))
                if len(text) > MAX_CONTEXT_DOC_CHARS:
                    text = text[:MAX_CONTEXT_DOC_CHARS]
                if not text:
                    continue
                if len(text) > budget:
                    text = text[:budget]
                if not text:
                    continue
                budget -= len(text)
                doc["text"] = text
                prepared.append(doc)
                if budget <= 0:
                    break
        if not prepared:
            return documents

        top_meta = top.get("metadata") or {}
        # Explicit ALL scope: inject the financial-statement tables for every
        # ingested filing (not just the companies named in the question), so a
        # cross-entity prompt synthesizes figures from all companies at once.
        if all_companies:
            all_tickers = list(SUPPORTED_TICKERS)
        else:
            all_tickers = self._detect_all_tickers(query)
            if not all_tickers:
                single = self._detect_ticker(query) or top_meta.get("ticker")
                if single:
                    all_tickers = [single]
        fiscal_year = self._query_fiscal_year(query) or top_meta.get("fiscal_year")

        if all_tickers and fiscal_year and fiscal_year != "UNKNOWN":
            num_tickers = len(all_tickers)
            # Split table budget across tickers for cross-company queries
            income_cap = MAX_INCOME_TABLE_CHARS // num_tickers
            compact_cap = MAX_COMPACT_INCOME_CHARS // num_tickers
            product_cap = MAX_PRODUCT_TABLE_CHARS // num_tickers
            metrics_cap = MAX_METRICS_TABLE_CHARS // num_tickers

            for tk in all_tickers:
                if tk == "UNKNOWN" or not tk:
                    continue
                # The financial-statement tables are the numeric ground truth for
                # every compared ticker. A transient Mongo read failure must never
                # silently starve a ticker of its income statement (which is what
                # turns a cross-company comparison into a "not available" refusal).
                # Retry the exact year-scoped fetch on transient exceptions so a
                # one-off hiccup cannot drop a whole ticker's evidence.
                tables: list[dict] = []
                for attempt in range(3):
                    try:
                        chunks = self._mongo_indexer.get_chunks_by_filter(
                            {"ticker": tk, "fiscal_year": str(fiscal_year)}
                        )
                        tables = [
                            c for c in chunks
                            if c.get("chunk_type") == "table" and c.get("raw_text")
                        ]
                        break
                    except Exception as exc:
                        logger.warning(
                            "Supplementary table fetch failed for %s (attempt %d/3): %s",
                            tk, attempt + 1, exc,
                        )
                        if attempt < 2:
                            time.sleep(0.2 * (attempt + 1))

                if not tables:
                    logger.warning(
                        "No financial-statement tables for %s FY%s; its figures will "
                        "be absent from the cross-company context",
                        tk, fiscal_year,
                    )
                    continue

                selected_ids: set[str] = set()
                query_lower = (query or "").lower()

                def _append(
                    candidate: dict | None,
                    section: str,
                    char_cap: int,
                    rank: int,
                ) -> None:
                    if candidate is None:
                        return
                    nonlocal budget
                    snippet = _compact_table_text(candidate["raw_text"])[: min(char_cap, budget)]
                    if not snippet:
                        return
                    budget -= len(snippet)
                    selected_ids.add(candidate.get("chunk_id"))
                    prepared.append({
                        "text": snippet,
                        "metadata": {
                            "ticker": candidate.get("ticker"),
                            "fiscal_year": candidate.get("fiscal_year"),
                            "section": candidate.get("section", section),
                            "doc_type": candidate.get("doc_type"),
                            "contains_table": True,
                            "page_number": candidate.get("page_number", "N/A"),
                            "source_file": candidate.get("source_file"),
                            "chunk_type": "table",
                            "augment_rank": rank,
                        },
                        "chunk_id": candidate.get("chunk_id"),
                        "score": 0.0,
                        "supplementary": True,
                    })

                def _available() -> list[dict]:
                    return [
                        c for c in tables if c.get("chunk_id") not in selected_ids
                    ]

                income = self._select_income_table(tables)
                income_truncated = len(
                    _compact_table_text(income.get("raw_text", ""))
                ) > income_cap
                _append(income, f"Income Statement ({tk})", income_cap, rank=2)

                needs_income_detail = any(
                    k in query_lower for k in ("net income", "per share", "earnings per share")
                )
                if income_truncated and needs_income_detail:
                    band_candidates = [
                        c for c in _available()
                        if _COMPACT_INCOME_BAND[0] <= len(c.get("raw_text", "")) <= _COMPACT_INCOME_BAND[1]
                        and any(m in c.get("raw_text", "").lower()
                               for m in _INCOME_CATEGORY_MARKERS["income"] + _INCOME_CATEGORY_MARKERS["pershare"])
                    ]
                    compact = max(
                        band_candidates,
                        key=lambda c: (
                            self._income_detail_score(c.get("raw_text", "")),
                            len(c.get("raw_text", "")),
                        ),
                    ) if band_candidates else None
                    if compact is not None:
                        _append(compact, f"Income Detail ({tk})", compact_cap, rank=1)

                product_term = next(
                    (t for t in _PRODUCT_SEGMENT_MARKERS if t in query_lower),
                    None,
                )
                if any(k in query_lower for k in _PRODUCT_QUERY_TERMS):
                    product = self._select_band_table(
                        _available(),
                        (product_term,) if product_term else _PRODUCT_SEGMENT_MARKERS,
                        *_PRODUCT_TABLE_BAND,
                    )
                    if product is not None:
                        _append(product, f"Product & Segments ({tk})", product_cap, rank=0)

                if "gross margin" in query_lower:
                    margin_candidates = [
                        c for c in _available()
                        if _METRICS_TABLE_BAND[0] <= len(c.get("raw_text", "")) <= _METRICS_TABLE_BAND[1]
                        and "gross margin percentage" in c.get("raw_text", "").lower()
                    ]
                    if not margin_candidates:
                        margin_candidates = [
                            c for c in _available()
                            if _METRICS_TABLE_BAND[0] <= len(c.get("raw_text", "")) <= _METRICS_TABLE_BAND[1]
                            and "gross margin" in c.get("raw_text", "").lower()
                        ]
                    metrics = max(
                        margin_candidates,
                        key=lambda c: (
                            self._percent_value_score(c.get("raw_text", "")),
                            len(c.get("raw_text", "")),
                        ),
                    ) if margin_candidates else None
                    if metrics is not None:
                        _append(metrics, f"Gross Margin ({tk})", metrics_cap, rank=0)

                if any(k in query_lower for k in _CASH_FLOW_QUERY_TERMS):
                    cf = self._select_band_table(
                        _available(),
                        _CASH_FLOW_MARKERS,
                        *_CASH_FLOW_TABLE_BAND,
                    )
                    if cf is None:
                        cf = max(
                            [
                                c for c in _available()
                                if any(m in c.get("raw_text", "").lower() for m in _CASH_FLOW_MARKERS)
                            ],
                            key=lambda c: len(c.get("raw_text", "")),
                        ) if any(
                            any(m in c.get("raw_text", "").lower() for m in _CASH_FLOW_MARKERS)
                            for c in _available()
                        ) else None
                    if cf is not None:
                        _append(cf, f"Cash Flow Statement ({tk})", MAX_CASH_FLOW_TABLE_CHARS, rank=0)

                if any(k in query_lower for k in _SEGMENT_QUERY_TERMS):
                    seg = self._select_band_table(
                        _available(),
                        _SEGMENT_MARKERS,
                        1500, 9000,
                    )
                    if seg is None:
                        seg = max(
                            [
                                c for c in _available()
                                if any(m in c.get("raw_text", "").lower() for m in _SEGMENT_MARKERS)
                            ],
                            key=lambda c: len(c.get("raw_text", "")),
                        ) if any(
                            any(m in c.get("raw_text", "").lower() for m in _SEGMENT_MARKERS)
                            for c in _available()
                        ) else None
                    if seg is not None:
                        _append(seg, f"Segment Results ({tk})", MAX_SEGMENT_TABLE_CHARS, rank=0)

        prepared.sort(
            key=lambda d: (d.get("metadata") or {}).get("augment_rank", 3)
        )

        # Final hygiene: purged section labels / text are what the generator
        # context XML and the API source chips are built from. The parser's
        # %%TABLE_N%% placeholders and the synthetic [TABLE N] markers are
        # internal artifacts that must never surface to the end user (the LLM
        # otherwise echoes them straight back into its sources field).
        for doc in prepared:
            meta = doc.get("metadata") or {}
            section = _strip_internal_markers(str(meta.get("section") or ""))
            meta["section"] = section or "General"
            doc["text"] = _strip_internal_markers(doc.get("text") or "")

        total = sum(len(d.get("text", "")) for d in prepared)
        logger.info(
            "Context prepared: %d documents, %d chars (budget=%d)",
            len(prepared),
            total,
            MAX_TOTAL_CONTEXT_CHARS,
        )
        return prepared

    def _run_post_retrieval(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[dict]:
        """
        Drive the Module 4 post-retrieval stage (rerank -> shield -> cylinder)
        from a sync context, loop-safe.

        Mirrors the event-loop strategy of `_run_hybrid_retrieval`: uses the
        current event loop when one exists, otherwise the module-level loop via
        `run_until_complete` (never `asyncio.run`, which would close a caller's
        loop). In a running loop the sync `process()` wrapper is offloaded to a
        worker thread so the loop is never blocked.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if threading.current_thread() is threading.main_thread():
                return asyncio.get_event_loop().run_until_complete(
                    self._post_retrieval.aprocess(query, chunks)
                )
            return asyncio.run(self._post_retrieval.aprocess(query, chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                self._post_retrieval.process, query, chunks
            ).result()

    def _run_hybrid_retrieval(
        self,
        query: str,
        queries: list[str],
        metadata_filter: dict,
    ) -> list[dict]:
        """
        Run Module 4 hybrid retrieval from the sync query() path, loop-safe,
        optionally applying the post-retrieval stage before adaptation.

        - No running loop (CLI / sync test / sync endpoint): drives the
          awaitable `asearch()` via the CURRENT event loop with
          `run_until_complete`, preserving the persistent-loop contract the
          async Redis guardrail relies on (never `asyncio.run`, which would
          close a caller's loop).
        - Running loop (async FastAPI endpoint calling sync query()): the
          event loop cannot be blocked in-thread, so the sync `search()`
          (which manages its own loop) is offloaded to a worker thread.
        - With post-retrieval enabled, the 40 hybrid chunks are re-ranked to
          top-8, table-shielded, and cylinder-reordered before adaptation.

        Returns the adapted pipeline document list.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if threading.current_thread() is threading.main_thread():
                raw = asyncio.get_event_loop().run_until_complete(
                    self._hybrid_search.asearch(queries, metadata_filter)
                )
            else:
                raw = asyncio.run(self._hybrid_search.asearch(queries, metadata_filter))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(
                    self._hybrid_search.search, queries, metadata_filter
                ).result()

        if self._enable_post_retrieval and self._post_retrieval is not None:
            raw = self._run_post_retrieval(query, raw)

        return self._adapt_hybrid_documents(raw)

    async def _run_hybrid_retrieval_async(
        self,
        query: str,
        queries: list[str],
        metadata_filter: dict,
    ) -> list[dict]:
        """Awaitable hybrid retrieval for the async query_stream() path."""
        results = await self._hybrid_search.asearch(queries, metadata_filter)
        if self._enable_post_retrieval and self._post_retrieval is not None:
            results = await self._post_retrieval.aprocess(query, results)
        return self._adapt_hybrid_documents(results)

    def query(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 3,
        tickers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Execute a full RAG query end-to-end.

        Steps:
            1. Cache check — return instantly on Redis hit
            2. Vector retrieval from Qdrant + MongoDB enrichment
            3. XML context formatting
            4. LLM generation via primary/fallback Groq models
            5. Background async guardrail verification + cache write
            6. MLflow logging of end-to-end metrics

        Args:
            user_query: Natural language financial question.
            ticker: Optional ticker filter (e.g. 'AAPL').
            fiscal_year: Optional fiscal year filter (e.g. '2025').
            session_id: Optional session identifier for conversation memory.
            top_k: Number of documents to retrieve.

        Returns:
            ConsolidatedFinancialAnswer dict with validated JSON response.
        """
        pipeline_run_id = str(uuid.uuid4())[:8]
        t_start = time.time()

        logger.info(
            "Pipeline query started: id=%s ticker=%s year=%s",
            pipeline_run_id,
            ticker,
            fiscal_year,
        )

        # Start MLflow run
        mlflow_active = False
        try:
            mlflow.set_experiment("financial_rag_pipeline")
            mlflow.start_run(run_name=f"pipeline_{pipeline_run_id}", nested=True)
            mlflow.log_param("query", user_query[:200])
            mlflow.log_param("ticker", ticker or "ANY")
            mlflow.log_param("fiscal_year", fiscal_year or "ANY")
            mlflow.log_param("top_k", top_k)
            mlflow_active = True
        except Exception as exc:
            logger.warning("MLflow pipeline run start failed (non-blocking): %s", exc)

        result: dict[str, Any] | None = None
        started_guardrail = False
        try:
            # Step 1: Cache check
            if self._enable_cache and self._cache is not None:
                cached = self._cache.get(
                    user_query, ticker=ticker, fiscal_year=fiscal_year
                )
                if cached is not None:
                    logger.info("Cache HIT for pipeline query (id=%s)", pipeline_run_id)
                    if mlflow_active:
                        try:
                            mlflow.log_metric("cache_hit", 1)
                            mlflow.log_metric("e2e_latency_ms", round((time.time() - t_start) * 1000, 2))
                        except Exception:
                            pass
                    result = {
                        "raw_output": cached.get("answer_json", ""),
                        "parsed": None,
                        "model_used": "cache",
                        "fallback_triggered": False,
                        "ttft_ms": 0,
                        "cache_hit": True,
                        "pipeline_run_id": pipeline_run_id,
                    }
                    try:
                        from schemas import ConsolidatedFinancialAnswer
                        result["parsed"] = ConsolidatedFinancialAnswer.model_validate_json(
                            cached.get("answer_json", "")
                        )
                    except Exception:
                        result["parsed"] = None
                    return result

            # Step 2: Pre-Retrieval (Module 3) -- optional intent routing
            pr_route = "DEFAULT"
            skip_cache_override = False

            # Explicit cross-entity scope requested by the client (e.g. the
            # frontend "ALL - Cross-Entity" ticker option, or "ALL" passed in
            # the ``tickers`` list). When set, retrieval must span EVERY
            # ingested filing rather than the single collection implied by the
            # active company selector -- this is what unblocks cross-ticker
            # comparison prompts that previously collapsed to one company and
            # returned "information not available".
            all_companies = (bool(ticker) and ticker.upper() == "ALL") or any(
                bool(t) and t.upper() == "ALL" for t in (tickers or [])
            )

            if self._enable_pre_retrieval and self._pre_retrieval is not None:
                pr = self._pre_retrieval.process(user_query, history=self._get_conversation_history())

                # Route 1: security violation -- halt immediately, no GPU spend
                if not pr.is_safe:
                    logger.warning(
                        "Security violation blocked (id=%s): %r",
                        pipeline_run_id,
                        user_query[:80],
                    )
                    if mlflow_active:
                        try:
                            mlflow.log_param("pre_retrieval_route", "BLOCKED")
                            mlflow.log_metric("documents_retrieved", 0)
                            mlflow.log_metric("cache_hit", 0)
                            mlflow.log_metric(
                                "e2e_latency_ms",
                                round((time.time() - t_start) * 1000, 2),
                            )
                        except Exception:
                            pass
                    result = {
                        "raw_output": SECURITY_VIOLATION_RESPONSE,
                        "parsed": None,
                        "model_used": "pre-retrieval",
                        "fallback_triggered": False,
                        "ttft_ms": 0,
                        "cache_hit": False,
                        "security_violation": True,
                        "pipeline_run_id": pipeline_run_id,
                    }
                    return result

                # Route 2: GENERAL chitchat -- bypass vector DBs and RAG
                if pr.action == "GENERAL":
                    pr_route = "GENERAL"
                    documents = []
                    retrieval_ms = 0.0
                    skip_cache_override = True
                else:
                    # Route 3: REWRITE -- expanded queries + metadata pre-filter
                    pr_route = "REWRITE"
                    t_retrieve = time.time()
                    # Issue 3: comparison/multi-company queries must never be
                    # scoped to a single ticker -- strip ticker keys first.
                    pr_scope = self._apply_cross_ticker_bypass(
                        user_query, pr.metadata_filter
                    )
                    # Explicit ALL scope: never apply any single-ticker
                    # restriction, even if the pre-retrieval stage derived one.
                    if all_companies:
                        pr_scope = {
                            key: value
                            for key, value in pr_scope.items()
                            if key not in ("ticker", "tickers")
                        }
                    if self._enable_hybrid_retrieval and self._hybrid_search is not None:
                        # Module 4: Module 3 queries + metadata dict route
                        # straight into the hybrid engine (dense+BM25+RRF),
                        # then the optional post-retrieval stage.
                        documents = self._run_hybrid_retrieval(
                            user_query, pr.queries, pr_scope
                        )
                    else:
                        documents = self._retrieve_documents_multi(
                            pr.queries, pr.embeddings, pr_scope, top_k
                        )
                    retrieval_ms = (time.time() - t_retrieve) * 1000
            else:
                # Legacy flow (pre-retrieval disabled). On the hybrid path use
                # the explicit args when given; otherwise ground the company and
                # fiscal year from the question text so the dense/BM25 candidate
                # pool and the supplementary-table injection stay scoped to the
                # filing the query is actually about.
                t_retrieve = time.time()

                # Single-company coreference for follow-up turns ("first
                # company", "what about the third company?", "what is its
                # revenue?") is resolved even when the full pre-retrieval stage
                # is disabled, scoping retrieval to the resolved company and
                # inheriting the session's fiscal year instead of collapsing to
                # every filing / the pipeline's default FY2024.
                memory_coref = self._resolve_memory_coreference(
                    user_query, ticker, tickers
                )
                resolved_ticker = None
                inherited_year = None
                if memory_coref:
                    resolved_ticker = memory_coref.get("ticker")
                    inherited_year = memory_coref.get("fiscal_year")
                    logger.info(
                        "Memory coreference resolved: %r -> %s (fiscal_year=%s)",
                        (user_query or "")[:80],
                        resolved_ticker,
                        inherited_year,
                    )
                    fiscal_year = inherited_year or fiscal_year
                    all_companies = False
                    ticker = resolved_ticker

                # Merge explicit tickers with auto-detected tickers from query text
                all_detected = self._detect_all_tickers(user_query)
                effective_tickers = list(tickers) if tickers else []
                for t in all_detected:
                    if t not in effective_tickers:
                        effective_tickers.append(t)

                # Issue 3: dynamic cross-ticker retrieval bypass. When the
                # question names multiple tickers, uses comparison wording, OR
                # the client explicitly selected the "ALL" cross-entity scope,
                # the single ticker handed over by the frontend payload is
                # discarded and BALANCED per-ticker sub-retrievals guarantee
                # every queried company contributes context.
                if self._has_multi_ticker_intent(user_query) or all_companies:
                    if all_companies:
                        from config.settings import SUPPORTED_TICKERS
                        compare_pool = list(SUPPORTED_TICKERS)
                        logger.info(
                            "Cross-entity 'ALL' scope: balanced per-ticker "
                            "sub-retrievals across all %d ingested filings "
                            "(requested_ticker=%s, tickers=%s)",
                            len(compare_pool),
                            ticker,
                            tickers,
                        )
                    else:
                        logger.info(
                            "Cross-ticker bypass: balanced per-ticker sub-retrievals "
                            "(requested_ticker=%s, tickers=%s)",
                            ticker,
                            tickers,
                        )
                        compare_pool = self._comparison_ticker_pool(user_query, tickers)
                    documents = self._balanced_ticker_subretrievals_sync(
                        user_query,
                        compare_pool,
                        fiscal_year=fiscal_year,
                        top_k=top_k,
                    )
                else:
                    scope_ticker = ticker
                    scope_tickers = (
                        list(effective_tickers) if effective_tickers else None
                    )

                    if self._enable_hybrid_retrieval and self._hybrid_search is not None:
                        if scope_tickers and len(scope_tickers) > 1:
                            hybrid_filter = {"tickers": scope_tickers}
                        else:
                            single_ticker = scope_ticker or (
                                scope_tickers[0] if scope_tickers else None
                            )
                            hybrid_filter = (
                                {"ticker": single_ticker} if single_ticker else {}
                            )
                        if fiscal_year:
                            hybrid_filter["fiscal_year"] = fiscal_year
                        if not hybrid_filter.get("fiscal_year"):
                            hybrid_filter["fiscal_year"] = self._query_fiscal_year(
                                user_query
                            )
                        documents = self._run_hybrid_retrieval(
                            user_query,
                            [user_query],
                            hybrid_filter,
                        )
                    else:
                        # Non-hybrid legacy path: also detect multi-ticker and pass
                        # the list so QdrantIndexer builds an OR filter.
                        multi_list = (
                            scope_tickers
                            if scope_tickers and len(scope_tickers) > 1
                            else None
                        )
                        effective_ticker = scope_ticker or (
                            multi_list[0] if multi_list else None
                        )
                        documents = self._retrieve_documents(
                            user_query,
                            ticker=effective_ticker,
                            fiscal_year=fiscal_year,
                            top_k=top_k,
                            tickers=multi_list,
                        )
                retrieval_ms = (time.time() - t_retrieve) * 1000

            if not documents and pr_route != "GENERAL":
                logger.warning("No documents retrieved for query (id=%s)", pipeline_run_id)
                if mlflow_active:
                    try:
                        mlflow.log_metric("documents_retrieved", 0)
                        mlflow.log_metric("retrieval_ms", round(retrieval_ms, 2))
                        mlflow.log_metric("cache_hit", 0)
                        mlflow.log_metric("e2e_latency_ms", round((time.time() - t_start) * 1000, 2))
                    except Exception:
                        pass
                result = {
                    "raw_output": "The requested financial information is not available in the provided reports.",
                    "parsed": None,
                    "model_used": "none",
                    "fallback_triggered": False,
                    "ttft_ms": 0,
                    "cache_hit": False,
                    "pipeline_run_id": pipeline_run_id,
                }
                return result

            # Resolve parser table placeholders (%%TABLE_N%%) into the raw
            # table chunks so numeric figures actually reach the LLM context.
            # On the hybrid+post-retrieval path also bound the context size
            # (Groq TPM caps reject the 20K-token contexts the retriever can
            # produce) and inject missing financial-statement tables.
            documents = self._resolve_table_placeholders(documents)
            documents = self._augment_context(documents, user_query, all_companies=all_companies)

            # Module 6 tracing: capture the raw post-retrieval chunks fed to
            # the LLM so the offline evaluation BatchRunner can score them.
            self._last_contexts = list(documents)

            # Step 3: Format XML context
            context_xml = format_context_xml(documents)

            # Step 4: Generate answer. When the question was resolved by memory
            # to a specific company/year, hand the generator the explicit
            # entity (retrieval was already scoped; a question that still reads
            # "the Second company" makes the LLM refuse despite the figures).
            gen_query = _resolved_generation_query(
                user_query, resolved_ticker, inherited_year
            )
            gen_result = self._generator.generate(
                query=gen_query,
                retrieved_docs=documents,
                stream=False,
                multi_ticker=self._has_multi_ticker_intent(user_query) or all_companies,
            )

            total_ms = (time.time() - t_start) * 1000

            # Step 5: Background guardrail + cache write (skip cache for fallback/invalid answers)
            parsed_answer = gen_result.get("parsed")
            is_valid_answer = (
                parsed_answer is not None
                and not gen_result.get("fallback_triggered", False)
                and "not available" not in parsed_answer.answer.lower()
            )
            skip_cache = (not is_valid_answer) or skip_cache_override

            # Guardrail runs only on retrieved-financial flows (not chitchat).
            if (
                parsed_answer is not None
                and self._enable_guardrail
                and self._guardrail is not None
                and pr_route != "GENERAL"
            ):
                started_guardrail = True
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                                ticker=ticker,
                                fiscal_year=fiscal_year,
                            )
                        )
                    else:
                        loop.run_until_complete(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                                ticker=ticker,
                                fiscal_year=fiscal_year,
                            )
                        )
                except RuntimeError:
                    try:
                        asyncio.run(
                            self._guardrail.check(
                                query=user_query,
                                answer=parsed_answer.answer,
                                extracted_raw_data=parsed_answer.extracted_raw_data,
                                parsed_output=parsed_answer,
                                skip_cache=skip_cache,
                                ticker=ticker,
                                fiscal_year=fiscal_year,
                            )
                        )
                    except Exception as exc:
                        logger.warning("Guardrail check failed (non-blocking): %s", exc)
                except Exception as exc:
                    logger.warning("Guardrail check failed (non-blocking): %s", exc)

            # Step 6: MLflow logging
            if mlflow_active:
                try:
                    mlflow.log_metric("documents_retrieved", len(documents))
                    mlflow.log_metric("retrieval_ms", round(retrieval_ms, 2))
                    mlflow.log_metric("generation_ttft_ms", gen_result.get("ttft_ms", 0))
                    mlflow.log_metric("cache_hit", 0)
                    mlflow.log_metric("fallback_triggered", int(gen_result.get("fallback_triggered", False)))
                    mlflow.log_metric("e2e_latency_ms", round(total_ms, 2))
                    mlflow.log_param("model_used", gen_result.get("model_used", "unknown"))
                    mlflow.log_param("pre_retrieval_route", pr_route)
                except Exception as exc:
                    logger.warning("MLflow pipeline metrics failed (non-blocking): %s", exc)

            result = {
                **gen_result,
                "cache_hit": False,
                "pipeline_run_id": pipeline_run_id,
            }
            return result

        finally:
            if mlflow_active:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

    async def query_stream(
        self,
        user_query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 3,
        tickers: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming RAG query — yields LLM tokens as they arrive.

        Executes retrieval synchronously, then streams tokens from the LLM.
        Guardrail verification runs in the background after streaming completes.

        Args:
            user_query: Natural language financial question.
            ticker: Optional ticker filter.
            fiscal_year: Optional fiscal year filter.
            session_id: Optional session identifier.
            top_k: Number of documents to retrieve.

        Yields:
            Individual token strings from the LLM response.
        """
        # Step 1: Pre-Retrieval (Module 3) -- optional intent routing
        documents: list[dict] = []
        pr_route = "DEFAULT"
        multi_ticker = self._has_multi_ticker_intent(user_query)
        # Explicit cross-entity scope requested by the client ("ALL - Cross-Entity"
        # ticker option, or "ALL" in the tickers list). Forces retrieval across
        # every ingested filing instead of a single-ticker Qdrant collection.
        all_companies = (bool(ticker) and ticker.upper() == "ALL") or any(
            bool(t) and t.upper() == "ALL" for t in (tickers or [])
        )
        if self._enable_pre_retrieval and self._pre_retrieval is not None:
            pr = self._pre_retrieval.process(user_query, history=self._get_conversation_history())
            if not pr.is_safe:
                yield SECURITY_VIOLATION_RESPONSE
                return
            if pr.action == "GENERAL":
                pr_route = "GENERAL"
            else:
                pr_route = "REWRITE"
                # Issue 3: strip ticker scope for comparison/multi-company
                # questions before any Qdrant filter is applied.
                pr_scope = self._apply_cross_ticker_bypass(
                    user_query, pr.metadata_filter
                )
                if all_companies:
                    pr_scope = {
                        key: value
                        for key, value in pr_scope.items()
                        if key not in ("ticker", "tickers")
                    }
                if self._enable_hybrid_retrieval and self._hybrid_search is not None:
                    documents = await self._run_hybrid_retrieval_async(
                        user_query, pr.queries, pr_scope
                    )
                else:
                    documents = self._retrieve_documents_multi(
                        pr.queries, pr.embeddings, pr_scope, top_k
                    )
        else:
            # Single-company coreference for follow-up turns ("first company",
            # "what about the third company?", "what is its revenue?") is
            # resolved even when the full pre-retrieval stage is disabled,
            # scoping retrieval to the resolved company and inheriting the
            # session's fiscal year instead of collapsing to every filing / the
            # pipeline's default FY2024.
            memory_coref = self._resolve_memory_coreference(
                user_query, ticker, tickers
            )
            resolved_ticker = None
            inherited_year = None
            if memory_coref:
                resolved_ticker = memory_coref.get("ticker")
                inherited_year = memory_coref.get("fiscal_year")
                logger.info(
                    "Memory coreference resolved (stream): %r -> %s (fiscal_year=%s)",
                    (user_query or "")[:80],
                    resolved_ticker,
                    inherited_year,
                )
                fiscal_year = inherited_year or fiscal_year
                all_companies = False
                ticker = resolved_ticker

            # Merge explicit tickers with auto-detected tickers from query text
            all_detected = self._detect_all_tickers(user_query)
            effective_tickers = list(tickers) if tickers else []
            for t in all_detected:
                if t not in effective_tickers:
                    effective_tickers.append(t)

            # Issue 3: dynamic cross-ticker retrieval bypass -- comparison /
            # multi-company queries (or an explicit "ALL" cross-entity scope)
            # override the frontend's single active ticker and run BALANCED
            # per-ticker sub-retrievals so context from every queried company
            # reaches the generator.
            if self._has_multi_ticker_intent(user_query) or all_companies:
                if all_companies:
                    from config.settings import SUPPORTED_TICKERS
                    compare_pool = list(SUPPORTED_TICKERS)
                    logger.info(
                        "Cross-entity 'ALL' scope (stream): balanced per-ticker "
                        "sub-retrievals across all %d ingested filings "
                        "(requested_ticker=%s, tickers=%s)",
                        len(compare_pool),
                        ticker,
                        tickers,
                    )
                else:
                    logger.info(
                        "Cross-ticker bypass (stream): balanced per-ticker "
                        "sub-retrievals (requested_ticker=%s, tickers=%s)",
                        ticker,
                        tickers,
                    )
                    compare_pool = self._comparison_ticker_pool(user_query, tickers)
                documents = await self._balanced_ticker_subretrievals(
                    user_query,
                    compare_pool,
                    fiscal_year=fiscal_year,
                    top_k=top_k,
                )
            else:
                scope_ticker = ticker
                scope_tickers = list(effective_tickers) if effective_tickers else None

                if self._enable_hybrid_retrieval and self._hybrid_search is not None:
                    if scope_tickers and len(scope_tickers) > 1:
                        stream_filter = {"tickers": scope_tickers}
                    else:
                        single_ticker = scope_ticker or (
                            scope_tickers[0] if scope_tickers else None
                        )
                        stream_filter = (
                            {"ticker": single_ticker} if single_ticker else {}
                        )
                    if fiscal_year:
                        stream_filter["fiscal_year"] = fiscal_year
                    documents = await self._run_hybrid_retrieval_async(
                        user_query,
                        [user_query],
                        stream_filter,
                    )
                else:
                    multi_list = (
                        scope_tickers if scope_tickers and len(scope_tickers) > 1 else None
                    )
                    effective_ticker = scope_ticker or (
                        multi_list[0] if multi_list else None
                    )
                    documents = self._retrieve_documents(
                        user_query,
                        ticker=effective_ticker,
                        fiscal_year=fiscal_year,
                        top_k=top_k,
                        tickers=multi_list,
                    )

        if not documents and pr_route != "GENERAL":
            yield "The requested financial information is not available in the provided reports."
            return

        # Parity with sync query(): inline %%TABLE_N%% placeholders into their
        # MongoDB table chunks, bound context size, and inject the filing's
        # financial-statement tables (income statement etc.) so numeric
        # comparison questions actually have figures to answer from.
        documents = self._resolve_table_placeholders(documents)
        documents = self._augment_context(documents, user_query, all_companies=all_companies)

        # Expose the retrieved chunks to the API layer (main.py enriches the
        # SSE ``sources`` event from this store) using the same
        # {chunk_id, score, payload{...}} shape the sync query() consumers
        # expect, so citation strings map back to real chunk text.
        self._last_contexts = [
            {
                "chunk_id": str(doc.get("chunk_id") or ""),
                "score": float(doc.get("score") or 0.0),
                "payload": {
                    **(doc.get("metadata") or {}),
                    "text": doc.get("text", ""),
                    "raw_text": doc.get("text", ""),
                },
            }
            for doc in documents
        ]

        # Stream tokens from generator (ditto sync: make the resolved company/year
        # explicit so the LLM doesn't refuse a coreferential follow-up question).
        gen_query = _resolved_generation_query(
            user_query, resolved_ticker, inherited_year
        )
        full_response_parts: list[str] = []
        async for token in self._generator.stream_tokens(
            query=gen_query,
            retrieved_docs=documents,
            multi_ticker=multi_ticker or all_companies,
        ):
            full_response_parts.append(token)
            yield token

        # Step 3: Background guardrail after streaming completes
        full_response = "".join(full_response_parts)
        if self._enable_guardrail and self._guardrail is not None and pr_route != "GENERAL":
            try:
                parsed = self._generator._parse_json_output(full_response)
                if parsed is not None:
                    skip_cache = "not available" in parsed.answer.lower()
                    asyncio.ensure_future(
                        self._guardrail.check(
                            query=user_query,
                            answer=parsed.answer,
                            extracted_raw_data=parsed.extracted_raw_data,
                            parsed_output=parsed,
                            skip_cache=skip_cache,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                        )
                    )
            except Exception as exc:
                logger.warning("Stream guardrail check failed (non-blocking): %s", exc)

        logger.info("Stream complete: %d tokens", len(full_response_parts))

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self._cache is not None:
            self._cache.close()
            logger.info("Cache cleared")

    def reset_memory(self) -> None:
        """Reset the generator's conversation memory."""
        self._generator.reset_memory()

    def _get_conversation_history(self) -> list[dict[str, str]]:
        """Extract the generator's conversation history as a list of role/content dicts.

        Skips the system message. Returns an empty list when the generator is
        mocked or has no memory attached.
        """
        history: list[dict[str, str]] = []
        memory = getattr(self._generator, "_memory", None)
        if memory is None:
            return history
        messages = getattr(memory, "messages", None)
        if not messages:
            return history
        for msg in messages:
            try:
                if getattr(msg, "type", None) == "system":
                    continue
                history.append({
                    "role": getattr(msg, "type", "user"),
                    "content": getattr(msg, "content", ""),
                })
            except Exception:
                continue
        return history

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics (Qdrant/MongoDB counts)."""
        try:
            return {
                "qdrant_total": self._qdrant_indexer.count_points(),
                "mongo_total": self._mongo_indexer.count_documents(),
            }
        except Exception as exc:
            logger.warning("Failed to get stats: %s", exc)
            return {"error": str(exc)}

    def close(self) -> None:
        """Close all database connections and release resources."""
        try:
            self._qdrant_indexer.close()
        except Exception:
            pass
        try:
            self._mongo_indexer.close()
        except Exception:
            pass
        if self._guardrail is not None:
            try:
                self._guardrail.close()
            except Exception:
                pass
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass
        logger.info("Pipeline closed")
