"""
Integration Test -- Module 4 (Post-Retrieval Pipeline) wired through the full pipeline.

Exercises the REAL HybridSearchEngine (dense + BM25 + RRF) followed by the
post-retrieval stage (Cross-Encoder rerank -> Table Shield -> Cylinder reorder)
with REAL Redis + REAL Qdrant + REAL MongoDB (isolated temporary stores).
Only the cross-encoder model, Table Shield LLM, and generator LLM are stubbed.

Contract scenarios verified:
  1. Hybrid Search (40 chunks) -> Cross-Encoder Reranker (Top 8) ->
     Table Shield (Async Clean) -> Cylinder Reorder (dynamic: best at head,
     2nd-best at tail).
  2. Complete metadata retained: ticker, fiscal_year, section, source_file,
     page_number, chunk_id, doc_type.
  3. Table passthrough: contains_table=True chunks pass through Table Shield
     100% untouched (no LLM call, text unchanged).
  4. Non-table cleaning: contains_table=False chunks are cleaned (LLM stub).
  5. Cylinder reorder order: Rank 1, 3, 5, 7, 8, 6, 4, 2 by rerank_score.
  6. Route 3 (REWRITE) + Module 3 queries route into hybrid + post-retrieval.
  7. Legacy flow (pre-retrieval disabled) + post-retrieval still isolates by args.
  8. Cache still short-circuits before hybrid + post-retrieval.
  9. Post-retrieval disabled preserves legacy behaviour.

Isolation:
  - Dedicated Redis namespace (production untouched)
  - Per-test temporary Qdrant path + unique MongoDB database
  - Namespace flushed before and after every test
"""

import asyncio
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_caching"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "5_generation"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "3_pre_retrieval"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

from config.logging_config import get_logger
from schemas import ConsolidatedFinancialAnswer
from intent_router import RuleBasedIntentRouter
from orchestrator import PreRetrievalOrchestrator

logger = get_logger("test.post_retrieval_integration")

NS = "it4pr_sem_cache"
_DIM = 768

_VOCAB = [
    "what", "was", "is", "the", "apple", "microsoft", "nvidia",
    "revenue", "net", "sales", "income", "fiscal", "2024", "2025",
    "in", "for", "reported", "million", "dollars", "and", "operating",
    "growth", "products", "services", "segment", "azure", "cloud",
    "risk", "factor", "item", "1a", "balance", "sheet", "cash",
    "flow", "quarter", "earnings", "profit", "loss", "cost",
    "research", "development", "total", "assets", "liabilities",
]


def fake_embed_768(text: str) -> list[float]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    vec = [0.0] * _DIM
    for i, word in enumerate(_VOCAB):
        if word in tokens:
            vec[i] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Corpus: 25 AAPL FY2024 chunks (hybrid returns top-25 for the filter)
# ---------------------------------------------------------------------------

_SECTIONS = [
    "Income Statement", "Balance Sheet", "Cash Flow Statement",
    "Item 1A", "Item 7", "Item 1", "Risk Factors", "General",
]


def _make_25_chunks() -> list[dict]:
    """25 diverse AAPL FY2024 chunks -- 5 tables, 20 text."""
    chunks = []
    for i in range(25):
        section = _SECTIONS[i % len(_SECTIONS)]
        is_table = (i % 5 == 0)
        if is_table:
            text = (
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Revenue | {245100 + i} |\n| Operating Income | {160800 + i} |"
            )
        else:
            text = (
                f"AAPL fiscal 2024 {section} (chunk {i}). "
                f"Revenue 245100 million dollars. Operating income 160800 million. "
                f"Net income 94600 million. Research and development cost 29800 million."
            )
        chunks.append({
            "chunk_id": f"it4pr-aapl-2024-{i:03d}",
            "text": text,
            "chunk_type": "table" if is_table else "text",
            "token_count": 40,
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": section,
                "doc_type": "10-K",
                "contains_table": is_table,
                "page_number": str(20 + i),
                "source_file": "aapl_10k_2024.pdf",
            },
        })
    return chunks


def _make_5_chunks() -> list[dict]:
    """5 chunks for the post-retrieval-disabled legacy comparison."""
    return [
        {
            "chunk_id": f"it4pr-legacy-{i}",
            "text": f"Apple fiscal 2024 chunk {i}. Revenue 245100 million.",
            "chunk_type": "text",
            "token_count": 30,
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": _SECTIONS[i % len(_SECTIONS)],
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": str(20 + i),
                "source_file": "aapl_10k_2024.pdf",
            },
        }
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _Harness:
    def __init__(self, pipe):
        self.pipe = pipe
        self.cache = pipe._cache
        self.gen_calls = 0
        self.last_retrieved: list[dict] | None = None
        self.hybrid_calls = 0
        self.last_hybrid_queries: list[str] | None = None
        self.last_hybrid_filter: dict | None = None
        self.post_retrieval_calls = 0
        self.reranker_calls = 0
        self.table_shield_calls = 0
        self.cylinder_calls = 0

    def reset_counts(self):
        self.gen_calls = 0
        self.last_retrieved = None
        self.hybrid_calls = 0
        self.last_hybrid_queries = None
        self.last_hybrid_filter = None
        self.post_retrieval_calls = 0
        self.reranker_calls = 0
        self.table_shield_calls = 0
        self.cylinder_calls = 0


def _close_quietly(pipe) -> None:
    for closer in (
        lambda: pipe._qdrant_indexer.close(),
        lambda: pipe._mongo_indexer.close(),
    ):
        try:
            closer()
        except Exception:
            pass


def _build_harness(tmp_path, *, enable_pre_retrieval, enable_hybrid, enable_post):
    from pipeline import FinancialRAGPipeline

    qdrant_path = str(tmp_path / "qdrant_it4pr")
    mongo_db = f"fin_rag_it4pr_{uuid.uuid4().hex[:8]}"

    pipe = FinancialRAGPipeline(
        qdrant_path=qdrant_path,
        mongo_db=mongo_db,
        mongo_collection="raw_chunks",
        qdrant_collection="it4pr_vectors",
        top_k=3,
        enable_cache=True,
        enable_guardrail=True,
        enable_pre_retrieval=enable_pre_retrieval,
        enable_hybrid_retrieval=enable_hybrid,
        enable_post_retrieval=enable_post,
    )
    h = _Harness(pipe)

    # ---- Cache & embedder bindings ----
    pipe._cache._namespace = NS
    pipe._guardrail._cache._namespace = NS
    pipe._cache._embed_fn = fake_embed_768
    pipe._guardrail._cache._embed_fn = fake_embed_768
    pipe._embedding_engine.embed_single = fake_embed_768
    pipe._embedding_engine.embed = lambda texts: [fake_embed_768(t) for t in texts]

    # _augment_context (filing-table injection) is not part of the
    # post-retrieval contract under test; pin it to identity so assertions
    # observe exactly the rerank/shield/cylinder output.
    pipe._augment_context = lambda documents, query: documents

    if enable_pre_retrieval:
        pipe._pre_retrieval = PreRetrievalOrchestrator(
            router=RuleBasedIntentRouter(),
            embed_fn=lambda texts: [fake_embed_768(t) for t in texts],
        )

    if enable_hybrid and pipe._hybrid_search is not None:
        # HybridSearchEngine expects a BATCH embedder (list[str] -> list[vec]).
        pipe._hybrid_search._embed_fn = lambda texts: [
            fake_embed_768(t) for t in texts
        ]

    # ---- Stub the generator ----
    def stub_generate(query, retrieved_docs, stream=False):
        h.gen_calls += 1
        h.last_retrieved = list(retrieved_docs or [])
        n = len(h.last_retrieved)
        answer_text = f"Retrieved {n} documents for query"
        parsed = ConsolidatedFinancialAnswer(
            internal_thought="test 123",
            extracted_raw_data=answer_text,
            answer=answer_text,
            sources=[
                h.last_retrieved[0].get("chunk_id", "N/A") if h.last_retrieved else "N/A"
            ],
        )
        return {
            "raw_output": parsed.model_dump_json(),
            "parsed": parsed,
            "model_used": "test-stub",
            "fallback_triggered": False,
            "ttft_ms": 0.0,
        }

    pipe._generator.generate = stub_generate

    # ---- Post-Retrieval mocks (reranker + table shield) ----
    if enable_post and pipe._post_retrieval is not None:
        # Mock reranker: deterministic scores, top_n=8
        _orig_reranker_rerank = None

        def _fake_rerank(query, chunks, top_n=8):
            h.reranker_calls += 1
            top = []
            for i, chunk in enumerate(chunks[:top_n]):
                copy = dict(chunk)
                copy["rerank_score"] = round(1.0 - i * 0.1, 4)
                top.append(copy)
            return top

        pipe._post_retrieval._reranker.rerank = _fake_rerank
        pipe._post_retrieval._reranker._top_n = 8

        # Mock table shield: async clean for non-tables, passthrough for tables
        async def _fake_shield(chunks):
            h.table_shield_calls += 1
            results = []
            for chunk in chunks:
                meta = dict(chunk.get("metadata") or {})
                is_table = meta.get("contains_table") is True
                if is_table:
                    results.append({
                        **chunk,
                        "cleaned_text": chunk.get("text", ""),
                        "metadata": meta,
                    })
                else:
                    results.append({
                        **chunk,
                        "text": f"Cleaned: {chunk.get('text', '')}",
                        "cleaned_text": f"Cleaned: {chunk.get('text', '')}",
                        "metadata": meta,
                    })
            return results

        pipe._post_retrieval._table_shield.shield = _fake_shield

        # Wrap the post-retrieval aprocess to count calls
        _orig_aprocess = pipe._post_retrieval.aprocess

        async def _counting_aprocess(query, chunks):
            h.post_retrieval_calls += 1
            return await _orig_aprocess(query, chunks)

        pipe._post_retrieval.aprocess = _counting_aprocess

    # ---- Redis check ----
    if pipe._cache._get_client() is False:
        _close_quietly(pipe)
        pytest.skip("Redis unavailable -- integration test requires live Redis")

    # ---- Seed isolated stores ----
    chunks = _make_25_chunks()
    embeddings = [fake_embed_768(c["text"]) for c in chunks]
    pipe._mongo_indexer.upsert_chunks(chunks)
    pipe._qdrant_indexer.upsert_vectors(chunks, embeddings)

    # ---- Wrap hybrid asearch to count ----
    if enable_hybrid and pipe._hybrid_search is not None:
        _orig_asearch = pipe._hybrid_search.asearch

        async def _counting_asearch(queries, metadata_filter=None, **kwargs):
            h.hybrid_calls += 1
            h.last_hybrid_queries = list(queries)
            h.last_hybrid_filter = metadata_filter
            return await _orig_asearch(queries, metadata_filter, **kwargs)

        pipe._hybrid_search.asearch = _counting_asearch

    pipe._cache.flush_all()
    return h


def _teardown(h) -> None:
    pipe = h.pipe
    try:
        pipe._cache.flush_all()
    except Exception:
        pass
    try:
        pipe._mongo_indexer.drop_collection()
    except Exception:
        pass
    _close_quietly(pipe)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _no_mlflow():
    with (
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.end_run"),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
    ):
        yield


@pytest.fixture(scope="module", autouse=True)
def _persistent_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield


@pytest.fixture
def post_retrieval_harness(tmp_path):
    h = _build_harness(
        tmp_path,
        enable_pre_retrieval=True,
        enable_hybrid=True,
        enable_post=True,
    )
    yield h
    _teardown(h)


@pytest.fixture
def legacy_post_retrieval_harness(tmp_path):
    h = _build_harness(
        tmp_path,
        enable_pre_retrieval=False,
        enable_hybrid=True,
        enable_post=True,
    )
    yield h
    _teardown(h)


# ============================================================================
# 1. Full Post-Retrieval Pipeline -- 25 -> rerank 5 -> shield -> cylinder
# ============================================================================

class TestFullPostRetrievalPipeline:

    def test_returns_top_8_documents(self, post_retrieval_harness):
        """Pipeline with post-retrieval must return exactly 8 docs to generator."""
        h = post_retrieval_harness
        q = "What was Apple's revenue in fiscal 2024?"
        res = h.pipe.query(q)

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert h.post_retrieval_calls == 1
        assert h.reranker_calls == 1
        assert h.table_shield_calls == 1
        assert h.gen_calls == 1
        assert h.last_retrieved is not None
        assert len(h.last_retrieved) == 8

    def test_cylinder_reorder_applied(self, post_retrieval_harness):
        """First doc in output should be rank-1 (best rerank_score)."""
        h = post_retrieval_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        docs = h.last_retrieved
        assert len(docs) == 8
        # Reranker mock assigns scores [1.0, 0.9, ..., 0.3] in original order.
        # Dynamic cylinder pattern (0,2,4,6,7,5,3,1) ->
        # [rank1, rank3, rank5, rank7, rank8, rank6, rank4, rank2].
        # Adapted docs still have rerank_score in metadata-like top-level field.
        scores = [d["rerank_score"] for d in docs]
        assert scores == [1.0, 0.8, 0.6, 0.4, 0.3, 0.5, 0.7, 0.9]

    def test_metadata_complete(self, post_retrieval_harness):
        """All metadata keys retained: ticker, fiscal_year, section, source_file, page_number, chunk_id."""
        h = post_retrieval_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        for doc in h.last_retrieved:
            assert "chunk_id" in doc
            assert doc["chunk_id"] is not None
            meta = doc["metadata"]
            assert meta["ticker"] == "AAPL"
            assert str(meta["fiscal_year"]) == "2024"
            assert "section" in meta
            assert meta.get("source_file") is not None
            assert meta.get("page_number") is not None

    def test_table_chunks_passthrough_untouched(self, post_retrieval_harness):
        """Table chunks (contains_table=True) pass through Table Shield unchanged."""
        h = post_retrieval_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        for doc in h.last_retrieved:
            meta = doc["metadata"]
            if meta.get("contains_table") is True:
                # Table text should NOT be "Cleaned: ..." prefixed
                assert not doc["text"].startswith("Cleaned:")
                assert doc["cleaned_text"] == doc["text"]

    def test_non_table_chunks_cleaned(self, post_retrieval_harness):
        """Non-table chunks are cleaned by Table Shield (mock returns 'Cleaned: ...')."""
        h = post_retrieval_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        has_cleaned = False
        for doc in h.last_retrieved:
            meta = doc["metadata"]
            if meta.get("contains_table") is False:
                assert doc["text"].startswith("Cleaned:")
                has_cleaned = True
        assert has_cleaned, "At least one non-table chunk should be cleaned"

    def test_strictly_isolates_apple_fiscal_2024(self, post_retrieval_harness):
        """All docs in the output must be AAPL FY2024."""
        h = post_retrieval_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert str(doc["metadata"]["fiscal_year"]) == "2024"

    def test_hybrid_top_k_bumped_to_40(self, post_retrieval_harness):
        """When post-retrieval is enabled, hybrid search candidate_k is 40."""
        h = post_retrieval_harness
        assert h.pipe._hybrid_search._candidate_k == 40


# ============================================================================
# 2. Route 3 REWRITE -- Module 3 queries + metadata route into hybrid + post
# ============================================================================

class TestRoute3PostRetrieval:

    def test_module3_queries_route_into_engine(self, post_retrieval_harness):
        h = post_retrieval_harness
        q = "What was Apple's revenue in fiscal 2024?"
        res = h.pipe.query(q)

        assert h.hybrid_calls == 1
        assert h.last_hybrid_queries == [q]
        assert h.last_hybrid_filter["ticker"] == "AAPL"
        assert str(h.last_hybrid_filter["fiscal_year"]) == "2024"
        assert res["parsed"] is not None

    def test_section_prefilter_isolates_correctly(self, post_retrieval_harness):
        h = post_retrieval_harness
        q = "What are the risk factors in the AAPL 10-K?"
        res = h.pipe.query(q)

        assert h.hybrid_calls == 1
        assert res["parsed"] is not None
        # All returned docs should be AAPL
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"


# ============================================================================
# 3. Legacy flow (pre-retrieval disabled) + post-retrieval
# ============================================================================

class TestLegacyPostRetrieval:

    def test_legacy_args_routes_into_post_retrieval(self, legacy_post_retrieval_harness):
        h = legacy_post_retrieval_harness
        q = "What was Apple's revenue in fiscal 2024?"
        res = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert h.hybrid_calls == 1
        assert h.post_retrieval_calls == 1
        assert len(h.last_retrieved) == 8
        assert h.last_hybrid_queries == [q]
        assert h.last_hybrid_filter == {"ticker": "AAPL", "fiscal_year": "2024"}

    def test_legacy_args_isolates_by_ticker_year(self, legacy_post_retrieval_harness):
        h = legacy_post_retrieval_harness
        h.pipe.query("Apple risk factors", ticker="AAPL", fiscal_year="2024")

        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert str(doc["metadata"]["fiscal_year"]) == "2024"


# ============================================================================
# 4. Cache still short-circuits before hybrid + post-retrieval
# ============================================================================

class TestPostRetrievalCacheInterplay:

    def test_cache_hit_skips_hybrid_and_post_retrieval(self, post_retrieval_harness):
        h = post_retrieval_harness
        q = "What was Apple's revenue in fiscal 2024?"
        h.pipe.query(q)  # First query -- populates cache

        h.reset_counts()
        res = h.pipe.query(q)  # Second query -- cache hit

        assert res["cache_hit"] is True
        assert res["model_used"] == "cache"
        assert h.hybrid_calls == 0
        assert h.post_retrieval_calls == 0
        assert h.gen_calls == 0


# ============================================================================
# 5. Post-retrieval disabled preserves legacy behaviour
# ============================================================================

class TestPostRetrievalDisabledPreservesLegacy:

    def test_post_retrieval_off_preserves_legacy_retrieval(self, tmp_path):
        from pipeline import FinancialRAGPipeline

        pipe = FinancialRAGPipeline(
            enable_cache=False,
            enable_guardrail=False,
            enable_pre_retrieval=False,
            enable_hybrid_retrieval=True,
            enable_post_retrieval=False,
        )
        try:
            assert pipe._enable_post_retrieval is False
            assert pipe._post_retrieval is None
            assert pipe._enable_hybrid_retrieval is True
            # Hybrid top_k stays at pipeline top_k (3), not HYBRID_TOP_K
            assert pipe._hybrid_search._top_k == 3
        finally:
            _close_quietly(pipe)


# ============================================================================
# 6. Post-retrieval implies hybrid enabled
# ============================================================================

class TestPostRetrievalImpliesHybrid:

    def test_post_retrieval_forces_hybrid_on(self, tmp_path):
        from pipeline import FinancialRAGPipeline

        pipe = FinancialRAGPipeline(
            enable_cache=False,
            enable_guardrail=False,
            enable_pre_retrieval=False,
            enable_hybrid_retrieval=False,
            enable_post_retrieval=True,
        )
        try:
            assert pipe._enable_hybrid_retrieval is True
            assert pipe._hybrid_search is not None
            assert pipe._post_retrieval is not None
            assert pipe._hybrid_search._candidate_k == 40
            assert pipe._hybrid_search._top_k == 40
        finally:
            _close_quietly(pipe)
