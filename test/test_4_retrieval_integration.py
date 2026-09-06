"""
Integration Test -- Module 4 (Hybrid Retrieval) wired through the full pipeline.

Exercises the REAL HybridSearchEngine (Qdrant dense + BM25 sparse + RRF) with
REAL Redis + REAL Qdrant + REAL MongoDB (isolated temporary stores) end to end.
Only the LLM generation step is stubbed, so the flow is deterministic and needs
no network or API key. The intent router is the deterministic
RuleBasedIntentRouter.

Contract scenarios verified:
  1. Route 3 (REWRITE) -> Module 3 queries + metadata_filter route STRAIGHT into
     HybridSearchEngine.asearch(); strict pre-filtering isolates the correct
     company / year / section chunks (dense + sparse share the same scope).
  2. Conditional expansion -> a short query produces 3-4 variants that reach the
     engine and are embedded in a SINGLE batch pass, then fused with RRF.
  3. Legacy flow (pre-retrieval disabled) -> single user query + ticker/year
     args route into the engine with a legacy-shaped filter.
  4. Module 2 interplay -> cache hits still short-circuit before hybrid search.
  5. Hybrid disabled by default -> legacy retrieval remains the fallback.

Isolation:
  - Dedicated Redis namespace `it4_sem_cache` (production untouched)
  - Per-test temporary Qdrant path + unique MongoDB database
  - Namespace flushed before and after every test
"""

import asyncio
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

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

logger = get_logger("test.hybrid_retrieval_integration")

NS = "it4_sem_cache"
_DIM = 768

# Deterministic 768-dim bag-of-words embedder (replaces nomic on the GPU).
_VOCAB = [
    "what", "was", "is", "the", "apple", "microsoft", "nvidia",
    "revenue", "net", "sales", "income", "fiscal", "2024", "2025",
    "in", "for", "reported", "million", "dollars", "and", "operating",
    "growth", "products", "services", "segment", "azure", "cloud",
    "risk", "factor", "item", "1a",
]


def fake_embed_768(text: str) -> list[float]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    vec = [0.0] * _DIM
    for i, word in enumerate(_VOCAB):
        if word in tokens:
            vec[i] = 1.0
    return vec


def _make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "it4-aapl-2024-inc",
            "text": (
                "Apple revenue in fiscal 2024 reported net sales 245100 million dollars. "
                "Products and Services segments drove growth."
            ),
            "chunk_type": "text",
            "token_count": 40,
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Income Statement",
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": "20",
            },
        },
        {
            "chunk_id": "it4-aapl-2025-inc",
            "text": (
                "Apple revenue in fiscal 2025 reported net sales 260000 million dollars. "
                "Strong growth across hardware and services segments."
            ),
            "chunk_type": "text",
            "token_count": 40,
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2025",
                "section": "Income Statement",
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": "45",
            },
        },
        {
            "chunk_id": "it4-aapl-2024-item1a",
            "text": (
                "Apple risk factors under Item 1A of the 10-K discuss "
                "competition in smartphones and cybersecurity threats."
            ),
            "chunk_type": "text",
            "token_count": 40,
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Item 1A",
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": "60",
            },
        },
        {
            "chunk_id": "it4-msft-2024-inc",
            "text": (
                "Microsoft revenue in fiscal 2024 reported net sales 160800 million dollars. "
                "Azure and cloud services drove growth."
            ),
            "chunk_type": "text",
            "token_count": 40,
            "metadata": {
                "ticker": "MSFT",
                "fiscal_year": "2024",
                "section": "Income Statement",
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": "25",
            },
        },
        {
            "chunk_id": "it4-msft-2024-item1a",
            "text": (
                "Microsoft risk factors under Item 1A of the 10-K discuss "
                "competition in cloud computing and cybersecurity threats."
            ),
            "chunk_type": "text",
            "token_count": 40,
            "metadata": {
                "ticker": "MSFT",
                "fiscal_year": "2024",
                "section": "Item 1A",
                "doc_type": "10-K",
                "contains_table": False,
                "page_number": "30",
            },
        },
    ]


# Deterministic per-chunk stub facts so answers can be asserted exactly.
_CHUNK_FACTS = {
    "it4-aapl-2024-inc": ("AAPL", "2024", "245100"),
    "it4-aapl-2025-inc": ("AAPL", "2025", "260000"),
    "it4-aapl-2024-item1a": ("AAPL", "2024", "risk factors"),
    "it4-msft-2024-inc": ("MSFT", "2024", "160800"),
    "it4-msft-2024-item1a": ("MSFT", "2024", "risk factors"),
}


class _Harness:
    def __init__(self, pipe):
        self.pipe = pipe
        self.cache = pipe._cache
        self.qdrant_calls = 0
        self.gen_calls = 0
        self.embed_batch_calls = 0
        self.hybrid_calls = 0
        self.hybrid_embed_calls = 0
        self.sparse_corpus_calls = 0
        self.last_hybrid_queries: list[str] | None = None
        self.last_hybrid_filter: dict | None = None
        self.last_retrieved: list[dict] | None = None

    def reset_counts(self):
        self.qdrant_calls = 0
        self.gen_calls = 0
        self.embed_batch_calls = 0
        self.hybrid_calls = 0
        self.hybrid_embed_calls = 0
        self.sparse_corpus_calls = 0
        self.last_hybrid_queries = None
        self.last_hybrid_filter = None
        self.last_retrieved = None

    def assert_no_downstream(self):
        assert self.hybrid_calls == 0, "hybrid retrieval should NOT run"
        assert self.qdrant_calls == 0, "vector search should NOT run"
        assert self.gen_calls == 0, "LLM generation should NOT run"


def _close_quietly(pipe) -> None:
    for closer in (
        lambda: pipe._qdrant_indexer.close(),
        lambda: pipe._mongo_indexer.close(),
    ):
        try:
            closer()
        except Exception:
            pass


def _build_harness(tmp_path, *, enable_pre_retrieval, enable_hybrid_retrieval):
    from pipeline import FinancialRAGPipeline

    qdrant_path = str(tmp_path / "qdrant_it4")
    mongo_db = f"fin_rag_it4_{uuid.uuid4().hex[:8]}"

    pipe = FinancialRAGPipeline(
        qdrant_path=qdrant_path,
        mongo_db=mongo_db,
        mongo_collection="raw_chunks",
        qdrant_collection="it4_vectors",
        top_k=3,
        enable_cache=True,
        enable_guardrail=True,
        enable_pre_retrieval=enable_pre_retrieval,
        enable_hybrid_retrieval=enable_hybrid_retrieval,
    )
    h = _Harness(pipe)

    # Rebind cache instances (pipeline fast-path + guardrail write path).
    pipe._cache._namespace = NS
    pipe._guardrail._cache._namespace = NS
    pipe._cache._embed_fn = fake_embed_768
    pipe._guardrail._cache._embed_fn = fake_embed_768

    # Rebind the pipeline embedder (fallback safety) to the deterministic one.
    pipe._embedding_engine.embed_single = fake_embed_768
    pipe._embedding_engine.embed = lambda texts: [fake_embed_768(t) for t in texts]

    if enable_pre_retrieval:
        # Deterministic rule router + counting expansion embedder.
        def _counting_batch_embed(texts):
            h.embed_batch_calls += 1
            return [fake_embed_768(t) for t in texts]

        pipe._pre_retrieval = PreRetrievalOrchestrator(
            router=RuleBasedIntentRouter(),
            embed_fn=_counting_batch_embed,
        )

    if enable_hybrid_retrieval and pipe._hybrid_search is not None:
        # The engine captured the ORIGINAL bound EmbeddingEngine.embed at
        # construction, so rebind it to the deterministic embedder FIRST, then
        # wrap it with a counting wrapper.
        pipe._hybrid_search._embed_fn = fake_embed_768
        _orig_embed = pipe._hybrid_search._embed_fn

        def _hybrid_embed(texts):
            h.hybrid_embed_calls += 1
            return [_orig_embed(t) for t in texts]

        pipe._hybrid_search._embed_fn = _hybrid_embed

    if pipe._cache._get_client() is False:
        _close_quietly(pipe)
        pytest.skip("Redis unavailable -- integration test requires live Redis")

    # Seed the isolated stores using the real Module 1 indexer primitives.
    chunks = _make_chunks()
    embeddings = [fake_embed_768(c["text"]) for c in chunks]
    pipe._mongo_indexer.upsert_chunks(chunks)
    pipe._qdrant_indexer.upsert_vectors(chunks, embeddings)

    # Downstream call counters (dense searches + sparse corpus fetches).
    _orig_search = pipe._qdrant_indexer.search

    def _search(*args, **kwargs):
        h.qdrant_calls += 1
        return _orig_search(*args, **kwargs)

    pipe._qdrant_indexer.search = _search

    _orig_get_filter = pipe._mongo_indexer.get_chunks_by_filter

    def _get_filter(*args, **kwargs):
        h.sparse_corpus_calls += 1
        return _orig_get_filter(*args, **kwargs)

    pipe._mongo_indexer.get_chunks_by_filter = _get_filter

    if enable_hybrid_retrieval and pipe._hybrid_search is not None:
        # Record exactly what Module 3 routes into the hybrid engine.
        _orig_hybrid = pipe._hybrid_search.asearch

        async def _asearch(queries, metadata_filter=None, **kwargs):
            h.hybrid_calls += 1
            h.last_hybrid_queries = list(queries)
            h.last_hybrid_filter = metadata_filter
            return await _orig_hybrid(queries, metadata_filter, **kwargs)

        pipe._hybrid_search.asearch = _asearch

    # Stub only the LLM call; everything else stays real.
    def stub_generate(query, retrieved_docs, stream=False):
        h.gen_calls += 1
        h.last_retrieved = list(retrieved_docs or [])
        ticker, year, figure = "ANY", "2024", "1000"
        if h.last_retrieved:
            cid = h.last_retrieved[0].get("chunk_id")
            ticker, year, figure = _CHUNK_FACTS.get(cid, ("ANY", "2024", "1000"))
        answer_text = f"{ticker} reported {figure} in fiscal {year}"
        parsed = ConsolidatedFinancialAnswer(
            internal_thought=f"verified {year}",
            extracted_raw_data=answer_text,
            answer=answer_text,
            sources=[f"{ticker} - {year} - {h.last_retrieved[0].get('chunk_id', 'N/A') if h.last_retrieved else 'N/A'}"],
        )
        return {
            "raw_output": parsed.model_dump_json(),
            "parsed": parsed,
            "model_used": "test-stub",
            "fallback_triggered": False,
            "ttft_ms": 0.0,
        }

    pipe._generator.generate = stub_generate

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
# Module fixtures
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
    """One long-lived event loop so the async Redis client never binds a dead loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield


@pytest.fixture
def hybrid_harness(tmp_path):
    h = _build_harness(
        tmp_path,
        enable_pre_retrieval=True,
        enable_hybrid_retrieval=True,
    )
    yield h
    _teardown(h)


@pytest.fixture
def legacy_hybrid_harness(tmp_path):
    h = _build_harness(
        tmp_path,
        enable_pre_retrieval=False,
        enable_hybrid_retrieval=True,
    )
    yield h
    _teardown(h)


# ============================================================================
# 1. Route 3 REWRITE -- Module 3 queries + metadata route into the hybrid engine
# ============================================================================

class TestRoute3HybridRewrite:

    def test_ticker_year_routes_into_engine_and_isolates_apple(self, hybrid_harness):
        h = hybrid_harness
        q = "What was Apple's revenue in fiscal 2024?"
        res = h.pipe.query(q)

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert "245100" in res["parsed"].answer
        # Module 3 -> Module 4 contract: exact query + strict filter reach asearch().
        assert h.hybrid_calls == 1
        assert h.last_hybrid_queries == [q]
        assert h.last_hybrid_filter["ticker"] == "AAPL"
        assert str(h.last_hybrid_filter["fiscal_year"]) == "2024"
        # Single fused search: one dense pass, one sparse corpus fetch, one embed.
        assert h.hybrid_embed_calls == 1
        assert h.sparse_corpus_calls == 2
        assert h.qdrant_calls == 1
        assert h.gen_calls == 1
        # Strict scope: every chunk handed to the generator is AAPL / fiscal 2024.
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert doc["metadata"]["fiscal_year"] == "2024"

    def test_ticker_section_prefilter_isolates_item1a(self, hybrid_harness):
        h = hybrid_harness
        q = "What are the risk factors in Item 1A of the MSFT 10-K?"
        res = h.pipe.query(q)

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert h.hybrid_calls == 1
        assert h.last_hybrid_filter == {"ticker": "MSFT", "section": "Item 1A"}
        assert "risk factors" in res["parsed"].answer
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "MSFT"
            assert doc["metadata"]["section"] == "Item 1A"

    def test_year_prefilter_isolates_fiscal_2025(self, hybrid_harness):
        h = hybrid_harness
        q = "What was Apple's revenue in fiscal 2025?"
        res = h.pipe.query(q)

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert "260000" in res["parsed"].answer
        assert str(h.last_hybrid_filter["fiscal_year"]) == "2025"
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert doc["metadata"]["fiscal_year"] == "2025"

    def test_msft_query_never_served_apple_data(self, hybrid_harness):
        h = hybrid_harness
        q = "What was Microsoft's revenue in fiscal 2024?"
        res = h.pipe.query(q)

        assert res["parsed"] is not None
        assert "160800" in res["parsed"].answer
        assert h.last_hybrid_filter["ticker"] == "MSFT"
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "MSFT"
            assert doc["metadata"]["fiscal_year"] == "2024"


# ============================================================================
# 2. Conditional expansion -- variants reach the engine, single batch embed
# ============================================================================

class TestHybridConditionalExpansion:

    def test_short_query_expands_variants_single_batch_and_fuses(self, hybrid_harness):
        h = hybrid_harness
        res = h.pipe.query("revenue growth")

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert h.hybrid_calls == 1, "expanded query must hit the engine once"
        assert h.last_hybrid_queries is not None
        assert len(h.last_hybrid_queries) >= 3, "M3 expansion must produce >= 3 variants"
        # M3 expansion embeds ALL variants in ONE batch; the engine then embeds
        # the full variant set in a SECOND single batch pass.
        assert h.embed_batch_calls == 1, "expansion must batch-embed once"
        assert h.hybrid_embed_calls == 1, "engine must batch-embed all variants once"
        assert h.qdrant_calls >= 3, "each variant must run a dense search"
        assert h.sparse_corpus_calls == 2, "one sparse corpus call per engine + one augment_context table fetch"
        assert h.gen_calls == 1

    def test_precise_query_skips_expansion(self, hybrid_harness):
        h = hybrid_harness
        h.pipe.query("What was Apple's revenue in fiscal 2024?")

        assert h.embed_batch_calls == 0, "no expansion -> no M3 batch embed"
        assert h.hybrid_embed_calls == 1, "single query still embedded by the engine"


# ============================================================================
# 3. Legacy flow (pre-retrieval disabled) -- hybrid still isolates by args
# ============================================================================

class TestHybridLegacyFallback:

    def test_legacy_args_filter_routes_into_engine(self, legacy_hybrid_harness):
        h = legacy_hybrid_harness
        q = "What was Apple's revenue in fiscal 2024?"
        res = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert "245100" in res["parsed"].answer
        assert h.hybrid_calls == 1
        assert h.last_hybrid_queries == [q]
        assert h.last_hybrid_filter == {"ticker": "AAPL", "fiscal_year": "2024"}
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "AAPL"
            assert doc["metadata"]["fiscal_year"] == "2024"

    def test_legacy_args_filter_isolates_msft(self, legacy_hybrid_harness):
        h = legacy_hybrid_harness
        q = "What was Microsoft's revenue in fiscal 2024?"
        res = h.pipe.query(q, ticker="MSFT", fiscal_year="2024")

        assert res["parsed"] is not None
        assert "160800" in res["parsed"].answer
        assert h.last_hybrid_filter == {"ticker": "MSFT", "fiscal_year": "2024"}
        assert h.last_retrieved
        for doc in h.last_retrieved:
            assert doc["metadata"]["ticker"] == "MSFT"


# ============================================================================
# 4. Module 2 interplay -- cache still short-circuits before hybrid search
# ============================================================================

class TestHybridCacheInterplay:

    def test_rewrite_miss_then_exact_hit_skips_hybrid_retrieval(self, hybrid_harness):
        h = hybrid_harness
        q = "What was Apple's revenue in fiscal 2024?"
        h.pipe.query(q)

        h.reset_counts()
        res = h.pipe.query(q)

        assert res["cache_hit"] is True
        assert res["model_used"] == "cache"
        h.assert_no_downstream()


# ============================================================================
# 5. Default wiring -- hybrid is OFF unless explicitly enabled
# ============================================================================

class TestHybridDisabledByDefault:

    def test_hybrid_flag_off_by_default_preserves_legacy_retrieval(self):
        from pipeline import FinancialRAGPipeline

        pipe = FinancialRAGPipeline(
            enable_cache=False,
            enable_guardrail=False,
            enable_pre_retrieval=False,
            enable_hybrid_retrieval=False,
        )
        try:
            assert pipe._enable_hybrid_retrieval is False
            assert pipe._hybrid_search is None
        finally:
            _close_quietly(pipe)
