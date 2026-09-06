"""
Integration Test -- Module 2 (Semantic Query Caching) wired into the full pipeline.

Exercises the REAL two-tier semantic cache + REAL Redis + REAL Qdrant + REAL
MongoDB (isolated temporary stores) end to end. Only the LLM generation step is
stubbed, so the flow is deterministic and requires no network or API key.

Contract scenarios verified:
  1. Cache miss  -> full pipeline execution -> guardrail PASS -> Redis write
  2. Exact / Tier-A high-similarity hit -> instant return, zero downstream calls
  3. Gray-zone (Tier B) -> reranker decides hit vs bypass (miss)
  4. Multi-tenant metadata integrity: AAPL vs MSFT answers never collide

Isolation:
  - Dedicated Redis namespace `it_sem_cache` (production `sem_cache` untouched)
  - Per-test temporary Qdrant path + unique MongoDB database
  - Namespace flushed before and after every test
"""

import asyncio
import json
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

from config.logging_config import get_logger
from config.settings import CACHE_TTL_STATIC_SECONDS
from schemas import ConsolidatedFinancialAnswer

logger = get_logger("test.caching_integration")

NS = "it_sem_cache"
_DIM = 768

# ---------------------------------------------------------------------------
# Deterministic 768-dim bag-of-words embedder (replaces nomic on the GPU).
# Preserves the Tier A (>= 0.96) / gray-zone (0.60-0.96) / miss (< 0.60)
# boundaries so the two-tier decision logic is exercised exactly.
# ---------------------------------------------------------------------------
_VOCAB = [
    "what", "was", "is", "the", "apple", "microsoft", "nvidia",
    "revenue", "net", "sales", "income", "fiscal", "2024", "2025",
    "in", "for", "reported", "million", "dollars", "and", "operating",
    "growth", "products", "services", "segment", "azure", "cloud",
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
            "chunk_id": "it-aapl-0001",
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
            "chunk_id": "it-msft-0001",
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
    ]


# Stubbed generation output per ticker. Every number in `answer` also appears
# verbatim in `extracted_raw_data`, so the real numerical guardrail PASSES and
# the verified answer is written to Redis.
_ANSWER_FACTS = {
    "AAPL": {
        "internal_thought": "AAPL verified 245100 figure.",
        "extracted_raw_data": "Apple net sales fiscal 2024: 245100 million dollars",
        "answer": "Apple reported net sales of 245100 million dollars in fiscal 2024",
        "sources": ["AAPL - 2024 - Income Statement - 20"],
    },
    "MSFT": {
        "internal_thought": "MSFT verified 160800 figure.",
        "extracted_raw_data": "Microsoft net sales fiscal 2024: 160800 million dollars",
        "answer": "Microsoft reported net sales of 160800 million dollars in fiscal 2024",
        "sources": ["MSFT - 2024 - Income Statement - 25"],
    },
    "ANY": {
        "internal_thought": "General verified 1000 figure.",
        "extracted_raw_data": "General net sales: 1000 million dollars",
        "answer": "General reported net sales of 1000 million dollars",
        "sources": ["ANY - 2024 - General - 1"],
    },
}


class _Harness:
    """Wrapper around a live pipeline plus downstream call counters."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.cache = pipe._cache
        self.qdrant_calls = 0
        self.mongo_calls = 0
        self.gen_calls = 0

    def reset_counts(self):
        self.qdrant_calls = 0
        self.mongo_calls = 0
        self.gen_calls = 0

    def assert_no_downstream(self):
        assert self.qdrant_calls == 0, "vector search should NOT run on a cache hit"
        assert self.mongo_calls == 0, "MongoDB enrichment should NOT run on a cache hit"
        assert self.gen_calls == 0, "LLM generation should NOT run on a cache hit"


def _redis_entry(h: _Harness, query: str, ticker: str, year: str) -> dict | None:
    key = h.cache._key(query, ticker, year)
    raw = h.cache._get_client().get(key)
    return None if raw is None else json.loads(raw)


def _close_quietly(pipe) -> None:
    for closer in (
        lambda: pipe._qdrant_indexer.close(),
        lambda: pipe._mongo_indexer.close(),
    ):
        try:
            closer()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _no_mlflow():
    """Silence MLflow writes so the test session never touches the MLflow store."""
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
    """Keep one long-lived event loop for the session.

    pipeline.query() runs the background guardrail via asyncio.get_event_loop().
    Without a persistent loop it falls back to asyncio.run(), whose loop is
    destroyed immediately and would leave the async Redis client bound to a
    dead loop, silently breaking later cache writes. Never close() this loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield


@pytest.fixture
def harness(tmp_path):
    from pipeline import FinancialRAGPipeline

    qdrant_path = str(tmp_path / "qdrant_it")
    mongo_db = f"fin_rag_it_{uuid.uuid4().hex[:8]}"

    pipe = FinancialRAGPipeline(
        qdrant_path=qdrant_path,
        mongo_db=mongo_db,
        mongo_collection="raw_chunks",
        qdrant_collection="it_vectors",
        top_k=2,
        enable_cache=True,
        enable_guardrail=True,
    )
    h = _Harness(pipe)

    # Rebind both cache instances (pipeline fast-path + guardrail write path)
    # to the dedicated namespace and to the deterministic embedder, so entries
    # written by the guardrail also carry an embedding for semantic screening.
    pipe._cache._namespace = NS
    pipe._guardrail._cache._namespace = NS
    pipe._cache._embed_fn = fake_embed_768
    pipe._guardrail._cache._embed_fn = fake_embed_768
    pipe._embedding_engine.embed_single = fake_embed_768

    if pipe._cache._get_client() is False:
        _close_quietly(pipe)
        pytest.skip("Redis unavailable -- integration test requires live Redis")

    # Seed the isolated stores using the real Module 1 indexer primitives.
    chunks = _make_chunks()
    embeddings = [fake_embed_768(c["text"]) for c in chunks]
    pipe._mongo_indexer.upsert_chunks(chunks)
    pipe._qdrant_indexer.upsert_vectors(chunks, embeddings)

    # Downstream call counters (cache-hit tests must prove zero calls).
    _orig_search = pipe._qdrant_indexer.search

    def _search(*args, **kwargs):
        h.qdrant_calls += 1
        return _orig_search(*args, **kwargs)

    pipe._qdrant_indexer.search = _search

    _orig_mongo = pipe._mongo_indexer.get_chunks_by_ids

    def _mongo(*args, **kwargs):
        h.mongo_calls += 1
        return _orig_mongo(*args, **kwargs)

    pipe._mongo_indexer.get_chunks_by_ids = _mongo

    # Stub only the LLM call; everything else stays real.
    def stub_generate(query, retrieved_docs, stream=False):
        h.gen_calls += 1
        ticker = "ANY"
        if retrieved_docs:
            ticker = retrieved_docs[0]["metadata"].get("ticker", "ANY")
        facts = _ANSWER_FACTS.get(ticker, _ANSWER_FACTS["ANY"])
        parsed = ConsolidatedFinancialAnswer(**facts)
        raw = parsed.model_dump_json()
        return {
            "raw_output": raw,
            "parsed": parsed,
            "model_used": "test-stub",
            "fallback_triggered": False,
            "ttft_ms": 0.0,
        }

    pipe._generator.generate = stub_generate

    pipe._cache.flush_all()

    yield h

    # Teardown: purge the namespace, drop the temp Mongo collection, release locks.
    try:
        pipe._cache.flush_all()
    except Exception:
        pass
    try:
        pipe._mongo_indexer.drop_collection()
    except Exception:
        pass
    _close_quietly(pipe)


# ============================================================================
# 1. Cache miss -> full pipeline execution -> guardrail PASS -> Redis write
# ============================================================================

class TestCacheMissWritesRedis:

    def test_miss_executes_full_pipeline_and_writes_redis(self, harness):
        h = harness
        q = "what was apple revenue in fiscal 2024"

        res = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert "Apple" in res["parsed"].answer
        assert h.qdrant_calls >= 1, "cache miss must reach Qdrant retrieval"
        assert h.mongo_calls >= 1, "cache miss must reach MongoDB enrichment"
        assert h.gen_calls == 1, "cache miss must reach LLM generation"

        entry = _redis_entry(h, q, "AAPL", "2024")
        assert entry is not None, "guardrail PASS must persist the verified answer to Redis"
        assert entry["guardrail_passed"] is True
        assert "Apple" in entry["answer_json"]
        assert entry["ticker"] == "AAPL"
        assert entry["fiscal_year"] == "2024"

        key = h.cache._key(q, "AAPL", "2024")
        assert key.startswith(f"{NS}:AAPL:2024:"), "cache key must be namespace-scoped"
        assert h.cache._get_client().ttl(key) == CACHE_TTL_STATIC_SECONDS, \
            "canonical SEC filing (ticker + year) must use the long static TTL"


# ============================================================================
# 2. Exact-match hit -> instant return, zero downstream calls
# ============================================================================

class TestExactHitSkipsDownstream:

    def test_identical_query_returns_cached_answer_without_downstream(self, harness):
        h = harness
        q = "what was apple revenue in fiscal 2024"
        h.pipe.query(q, ticker="AAPL", fiscal_year="2024")

        h.reset_counts()
        res = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is True
        assert res["model_used"] == "cache"
        assert "Apple" in res["raw_output"]
        h.assert_no_downstream()


# ============================================================================
# 3. Tier A -- high-similarity fast hit (same token set, reordered phrasing)
# ============================================================================

class TestTierAFastHit:

    def test_high_similarity_rephrase_hits_without_downstream(self, harness):
        h = harness
        stored = "what was apple revenue in fiscal 2024"
        rephrase = "apple revenue in fiscal 2024 what was"
        h.pipe.query(stored, ticker="AAPL", fiscal_year="2024")

        h.reset_counts()
        res = h.pipe.query(rephrase, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is True
        assert res["model_used"] == "cache"
        assert "Apple" in res["raw_output"]
        h.assert_no_downstream()


# ============================================================================
# 4. Tier B -- gray-zone reranking decides hit vs bypass (miss)
# ============================================================================

class TestTierBGrayZone:

    def test_gray_zone_rerank_hit(self, harness):
        h = harness
        stored = "apple revenue in fiscal 2024"
        gray = "apple revenue in fiscal 2024 and operating income growth"
        h.pipe.query(stored, ticker="AAPL", fiscal_year="2024")
        h.cache._rerank_fn = lambda _q, _cands: [0.80]

        h.reset_counts()
        res = h.pipe.query(gray, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is True
        assert res["model_used"] == "cache"
        assert "Apple" in res["raw_output"]
        h.assert_no_downstream()

    def test_gray_zone_rerank_miss_bypasses_cache(self, harness):
        h = harness
        stored = "apple revenue in fiscal 2024"
        gray = "apple revenue in fiscal 2024 and operating income growth"
        h.pipe.query(stored, ticker="AAPL", fiscal_year="2024")
        h.cache._rerank_fn = lambda _q, _cands: [0.50]

        h.reset_counts()
        res = h.pipe.query(gray, ticker="AAPL", fiscal_year="2024")

        assert res["cache_hit"] is False
        assert res["parsed"] is not None
        assert h.qdrant_calls >= 1, "Tier B miss must proceed to retrieval"
        assert h.gen_calls == 1, "Tier B miss must proceed to generation"


# ============================================================================
# 5. Multi-tenant metadata integrity -- AAPL vs MSFT never collide
# ============================================================================

class TestMetadataIntegrity:

    def test_aapl_cache_never_served_for_msft(self, harness):
        h = harness
        q = "what was apple revenue in fiscal 2024"

        r_aapl = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")
        assert r_aapl["cache_hit"] is False
        assert "Apple" in r_aapl["parsed"].answer

        assert h.cache.get(q, ticker="MSFT", fiscal_year="2024") is None, \
            "AAPL-namespaced entry must not be visible under the MSFT namespace"

        h.reset_counts()
        r_msft = h.pipe.query(q, ticker="MSFT", fiscal_year="2024")
        assert r_msft["cache_hit"] is False
        assert "Microsoft" in r_msft["parsed"].answer

        r_aapl_again = h.pipe.query(q, ticker="AAPL", fiscal_year="2024")
        assert r_aapl_again["cache_hit"] is True
        assert "Apple" in r_aapl_again["raw_output"]

        msft_entry = _redis_entry(h, q, "MSFT", "2024")
        assert msft_entry is not None
        assert "Microsoft" in msft_entry["answer_json"]
        assert msft_entry["ticker"] == "MSFT"
        assert msft_entry["fiscal_year"] == "2024"
