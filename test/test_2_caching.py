"""
Test Suite for Module 2: Semantic Query Caching (two-tier cache).

Covers:
  - Exact-match put/get/miss/invalidate with Redis
  - Ticker + fiscal_year namespacing (no cross-company / cross-year hits)
  - Tier A: fast embedding screening (>= 0.96 hit, < 0.60 miss)
  - Tier B: gray-zone reranking (>= 0.75 hit)
  - TTL policy (long TTL for SEC static filings vs session TTL)
  - Graceful fallback when Redis is unavailable
"""

import asyncio
import re
import sys
from pathlib import Path

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "2_caching"))

from config.logging_config import get_logger
from config.settings import (
    CACHE_FAST_HIT_THRESHOLD,
    CACHE_FAST_MISS_THRESHOLD,
    CACHE_RERANK_THRESHOLD,
    CACHE_TTL_SECONDS,
    CACHE_TTL_STATIC_SECONDS,
)
from semantic_cache import SemanticCache, _cosine, _hash_query

logger = get_logger("test.caching")

# Fixed vocabulary used by the fake embedder so vectors are comparable.
_VOCAB = [
    "what", "is", "the", "revenue", "apple", "microsoft",
    "fiscal", "report", "of", "2024", "2025",
]


def fake_embed(text: str) -> list[float]:
    """Deterministic bag-of-words embedding over a fixed vocabulary."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return [1.0 if w in tokens else 0.0 for w in _VOCAB]


def _redis_ok(cache: SemanticCache) -> bool:
    return cache._get_client() is not False


# ============================================================================
# TEST 1: Exact-match put / get / invalidate
# ============================================================================

def test_exact_roundtrip_and_invalidate():
    cache = SemanticCache(ttl_seconds=60)
    try:
        query = "What was Apple's revenue in 2025?"
        ok = cache.put(query, '{"answer":"$394.3B"}', guardrail_passed=True)
        if not ok:
            assert cache.get(query) is None
            logger.info("Exact roundtrip: Redis unavailable (graceful fallback OK)")
            return
        assert _redis_ok(cache)

        entry = cache.get(query)
        assert entry is not None
        assert entry["guardrail_passed"] is True
        assert "$394.3B" in entry["answer_json"]

        assert cache.get("totally different query") is None

        cache.invalidate(query)
        assert cache.get(query) is None
        logger.info("Exact roundtrip + invalidate: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 2: Ticker + fiscal_year namespacing
# ============================================================================

def test_namespacing_prevents_cross_company_hits():
    cache = SemanticCache(ttl_seconds=60)
    try:
        query = "What was revenue in fiscal 2025?"
        ok = cache.put(
            query, '{"answer":"AAPL data"}', guardrail_passed=True,
            ticker="AAPL", fiscal_year="2025",
        )
        if not ok:
            logger.info("Namespacing: Redis unavailable (graceful fallback OK)")
            return

        # Same text, different company / year -> must miss.
        assert cache.get(query, ticker="MSFT", fiscal_year="2025") is None
        assert cache.get(query, ticker="AAPL", fiscal_year="2024") is None
        assert cache.get(query, ticker="AAPL", fiscal_year=None) is None

        # Same text + same namespace -> hit.
        entry = cache.get(query, ticker="AAPL", fiscal_year="2025")
        assert entry is not None
        assert "AAPL data" in entry["answer_json"]
        logger.info("Namespacing: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 3: Tier A — fast embedding screening
# ============================================================================

def test_tier_a_fast_hit_and_fast_miss():
    cache = SemanticCache(
        namespace="test_tier_a",
        embed_fn=fake_embed,
    )
    try:
        stored = "What is apple revenue"
        ok = cache.put(
            stored, '{"answer":"tier A"}', guardrail_passed=True,
            ticker="AAPL", fiscal_year="2024",
        )
        if not ok:
            logger.info("Tier A: Redis unavailable (graceful fallback OK)")
            return

        # Near-identical phrasing (same token set) -> cosine 1.0 >= 0.96 -> hit.
        entry = cache.get(
            "What is apple revenue?", ticker="AAPL", fiscal_year="2024"
        )
        assert entry is not None, "expected Tier A fast hit"
        assert "tier A" in entry["answer_json"]

        # Disjoint token set -> cosine 0.0 < 0.60 -> fast miss.
        assert cache.get(
            "Fiscal report of microsoft", ticker="AAPL", fiscal_year="2024"
        ) is None
        logger.info("Tier A fast hit + fast miss: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 4: Tier B — gray-zone reranking
# ============================================================================

def test_tier_b_gray_zone_hit_and_miss():
    def rerank_hit(_query, _candidates):
        return [0.80]

    def rerank_miss(_query, _candidates):
        return [0.50]

    cache = SemanticCache(
        namespace="test_tier_b",
        embed_fn=fake_embed,
        rerank_fn=rerank_hit,
    )
    try:
        stored = "What is apple revenue"
        ok = cache.put(
            stored, '{"answer":"tier B"}', guardrail_passed=True,
            ticker="AAPL", fiscal_year="2024",
        )
        if not ok:
            logger.info("Tier B: Redis unavailable (graceful fallback OK)")
            return

        # "What is the revenue" overlaps 3/4 tokens -> cosine 0.75 (gray zone).
        entry = cache.get(
            "What is the revenue", ticker="AAPL", fiscal_year="2024"
        )
        assert entry is not None, "expected Tier B rerank hit (score 0.80 >= 0.75)"
        assert "tier B" in entry["answer_json"]

        # Now force rerank below threshold -> miss.
        cache._rerank_fn = rerank_miss
        assert cache.get(
            "What is the revenue", ticker="AAPL", fiscal_year="2024"
        ) is None
        logger.info("Tier B gray zone hit + miss: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 5: TTL policy — static SEC filings vs session queries
# ============================================================================

def test_ttl_policy_resolution():
    cache = SemanticCache()
    try:
        # Canonical SEC filing with ticker + fiscal_year -> long TTL.
        assert cache._resolve_ttl("AAPL", "2025") == CACHE_TTL_STATIC_SECONDS
        assert cache._resolve_ttl("MSFT", "2024") == CACHE_TTL_STATIC_SECONDS
        # Missing fiscal_year or unsupported ticker -> session TTL.
        assert cache._resolve_ttl("AAPL", None) == CACHE_TTL_SECONDS
        assert cache._resolve_ttl(None, None) == CACHE_TTL_SECONDS

        if not _redis_ok(cache):
            logger.info("TTL policy: Redis unavailable (logic check only OK)")
            return

        # Verify the long TTL actually lands on the stored key.
        key = cache._key("What was AAPL revenue 2025?", "AAPL", "2025")
        ok = cache.put(
            "What was AAPL revenue 2025?", '{"answer":"x"}',
            guardrail_passed=True, ticker="AAPL", fiscal_year="2025",
        )
        if not ok:
            logger.info("TTL policy: Redis unavailable (graceful fallback OK)")
            return
        client = cache._get_client()
        ttl = client.ttl(key)
        assert ttl == CACHE_TTL_STATIC_SECONDS, f"expected long TTL, got {ttl}"
        cache.invalidate("What was AAPL revenue 2025?", ticker="AAPL", fiscal_year="2025")
        logger.info("TTL policy: OK")
    finally:
        cache.close()


# ============================================================================
# TEST 6: flush_all — namespace-scoped purge
# ============================================================================

def test_flush_all_scoped_to_namespace():
    cache = SemanticCache(namespace="test_flush_ns", ttl_seconds=60)
    try:
        ok1 = cache.put("flush me", '{"answer":"1"}', guardrail_passed=True)
        ok2 = cache.put("flush me too", '{"answer":"2"}', guardrail_passed=True)
        if not (ok1 and ok2):
            logger.info("flush_all: Redis unavailable (graceful fallback OK)")
            return

        assert cache.get("flush me") is not None
        assert cache.get("flush me too") is not None

        removed = cache.flush_all()
        assert removed == 2
        assert cache.get("flush me") is None
        assert cache.get("flush me too") is None
        logger.info("flush_all (namespace-scoped): OK")
    finally:
        cache.close()


# ============================================================================
# TEST 7: Graceful fallback when Redis is unavailable
# ============================================================================

def test_graceful_fallback_redis_down():
    cache = SemanticCache(
        redis_url="redis://127.0.0.1:1",
        socket_connect_timeout=1,
        embed_fn=fake_embed,
    )
    try:
        assert cache.get("anything") is None
        assert cache.put("anything", '{"answer":"x"}', guardrail_passed=True) is False
        assert cache.invalidate("anything") is False
        assert cache.flush_all() == -1
        logger.info("Graceful fallback (Redis down): OK")
    finally:
        cache.close()


# ============================================================================
# TEST 8: Unit helpers
# ============================================================================

def test_helpers():
    assert _hash_query("Hello World") == _hash_query("hello world  ")
    assert _hash_query("alpha") != _hash_query("beta")
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(_cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    logger.info("Unit helpers: OK")


# ============================================================================
# TEST 9: Asynchronous API (aget / aput / ainvalidate / aflush_all)
# ============================================================================

def _run(coro):
    """Run an async cache coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


async def _aredis_ok(cache: SemanticCache) -> bool:
    return (await cache._aget_client()) is not False


def test_async_exact_roundtrip_and_invalidate():
    async def scenario():
        cache = SemanticCache(ttl_seconds=60)
        try:
            if not await _aredis_ok(cache):
                logger.info("Async roundtrip: Redis unavailable (graceful fallback OK)")
                return
            query = "What was Microsoft revenue in fiscal 2024?"
            ok = await cache.aput(query, '{"answer":"$245.1B"}', guardrail_passed=True)
            assert ok is True
            entry = await cache.aget(query)
            assert entry is not None
            assert entry["guardrail_passed"] is True
            assert "$245.1B" in entry["answer_json"]
            assert await cache.aget("some other question") is None
            assert await cache.ainvalidate(query) is True
            assert await cache.aget(query) is None
            logger.info("Async roundtrip + invalidate: OK")
        finally:
            cache.close()

    _run(scenario())


def test_async_namespacing_prevents_cross_company_hits():
    async def scenario():
        cache = SemanticCache(ttl_seconds=60)
        try:
            if not await _aredis_ok(cache):
                logger.info("Async namespacing: Redis unavailable (graceful fallback OK)")
                return
            query = "What was revenue in fiscal 2025?"
            await cache.aput(
                query, '{"answer":"AAPL data"}', guardrail_passed=True,
                ticker="AAPL", fiscal_year="2025",
            )
            assert await cache.aget(query, ticker="MSFT", fiscal_year="2025") is None
            assert await cache.aget(query, ticker="AAPL", fiscal_year="2024") is None
            entry = await cache.aget(query, ticker="AAPL", fiscal_year="2025")
            assert entry is not None
            assert "AAPL data" in entry["answer_json"]
            logger.info("Async namespacing: OK")
        finally:
            cache.close()

    _run(scenario())


def test_async_tier_a_fast_hit_and_miss():
    async def scenario():
        cache = SemanticCache(
            namespace="test_async_tier_a",
            embed_fn=fake_embed,
        )
        try:
            if not await _aredis_ok(cache):
                logger.info("Async Tier A: Redis unavailable (graceful fallback OK)")
                return
            await cache.aput(
                "What is apple revenue", '{"answer":"tier A"}',
                guardrail_passed=True, ticker="AAPL", fiscal_year="2024",
            )
            entry = await cache.aget(
                "What is apple revenue?", ticker="AAPL", fiscal_year="2024"
            )
            assert entry is not None, "expected async Tier A fast hit"
            assert "tier A" in entry["answer_json"]
            assert await cache.aget(
                "Fiscal report of microsoft", ticker="AAPL", fiscal_year="2024"
            ) is None
            logger.info("Async Tier A fast hit + fast miss: OK")
        finally:
            cache.close()

    _run(scenario())


def test_async_tier_b_gray_zone_hit_and_miss():
    def rerank_hit(_query, _candidates):
        return [0.80]

    def rerank_miss(_query, _candidates):
        return [0.50]

    async def scenario():
        cache = SemanticCache(
            namespace="test_async_tier_b",
            embed_fn=fake_embed,
            rerank_fn=rerank_hit,
        )
        try:
            if not await _aredis_ok(cache):
                logger.info("Async Tier B: Redis unavailable (graceful fallback OK)")
                return
            await cache.aput(
                "What is apple revenue", '{"answer":"tier B"}',
                guardrail_passed=True, ticker="AAPL", fiscal_year="2024",
            )
            entry = await cache.aget(
                "What is the revenue", ticker="AAPL", fiscal_year="2024"
            )
            assert entry is not None, "expected async Tier B rerank hit"
            assert "tier B" in entry["answer_json"]
            cache._rerank_fn = rerank_miss
            assert await cache.aget(
                "What is the revenue", ticker="AAPL", fiscal_year="2024"
            ) is None
            logger.info("Async Tier B gray zone hit + miss: OK")
        finally:
            cache.close()

    _run(scenario())


def test_async_flush_all_scoped_to_namespace():
    async def scenario():
        cache = SemanticCache(namespace="test_async_flush_ns", ttl_seconds=60)
        try:
            if not await _aredis_ok(cache):
                logger.info("Async flush: Redis unavailable (graceful fallback OK)")
                return
            assert await cache.aput("flush me", '{"answer":"1"}', guardrail_passed=True)
            assert await cache.aput("flush me too", '{"answer":"2"}', guardrail_passed=True)
            assert await cache.aget("flush me") is not None
            removed = await cache.aflush_all()
            assert removed == 2
            assert await cache.aget("flush me") is None
            assert await cache.aget("flush me too") is None
            logger.info("Async flush_all (namespace-scoped): OK")
        finally:
            cache.close()

    _run(scenario())


def test_async_graceful_fallback_redis_down():
    async def scenario():
        cache = SemanticCache(
            redis_url="redis://127.0.0.1:1",
            socket_connect_timeout=1,
            embed_fn=fake_embed,
        )
        try:
            assert await cache.aget("anything") is None
            assert await cache.aput("anything", '{"answer":"x"}', guardrail_passed=True) is False
            assert await cache.ainvalidate("anything") is False
            assert await cache.aflush_all() == -1
            logger.info("Async graceful fallback (Redis down): OK")
        finally:
            cache.close()

    _run(scenario())
