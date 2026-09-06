"""
Two-Tier Semantic Query Cache (Module 2).

GPTCache-style caching with Redis backing and two screening tiers:

  Tier A — Fast embedding screening (nomic-embed-text-v1.5 cosine similarity):
             score < 0.60   -> fast miss (proceed to retrieval)
             score >= 0.96  -> fast hit (return cached response immediately)
  Tier B — Gray zone (0.60 <= score < 0.96): bge-reranker-large cross-encoder;
             score >= 0.75  -> hit
             score <  0.75  -> miss

Metadata namespacing: keys are scoped as
  {namespace}:{TICKER}:{fiscal_year}:{sha256(query)}
so cross-company / cross-year queries can never collide.

TTL policy:
  - Long TTL (default 7 days) for canonical SEC 10-K static filings
    (ticker in SUPPORTED_TICKERS AND a fiscal_year is present).
  - Session TTL for ad-hoc / generic queries.

Both a synchronous and an asynchronous API are provided:
  - sync  (pipeline fast-path read, API warmup/flush): get / put / invalidate /
    flush_all / _get_client
  - async (background guardrail write, non-blocking): aget / aput / ainvalidate /
    aflush_all / _aget_client

The cache supports two modes:
  - Exact mode (default): plain SHA-256 lookup, no embeddings required.
  - Semantic mode: pass `embed_fn` (nomic embedder) to enable Tier A; pass
    `rerank_fn` (bge-reranker-large) or rely on the built-in lazy cross-encoder
    to enable Tier B. If an embedder is not provided, semantic screening is
    skipped entirely (safe fallback to exact-match only).
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable

from config.logging_config import get_logger
from config.settings import (
    CACHE_FAST_HIT_THRESHOLD,
    CACHE_FAST_MISS_THRESHOLD,
    CACHE_NAMESPACE,
    CACHE_RERANKER_MODEL,
    CACHE_RERANK_THRESHOLD,
    CACHE_TTL_SECONDS,
    CACHE_TTL_STATIC_SECONDS,
    REDIS_URL,
    SUPPORTED_TICKERS,
)
from redis_client import RedisClient

logger = get_logger("caching.semantic")

EmbedFn = Callable[[str], list[float]]
RerankFn = Callable[[str, list[str]], list[float]]


def _hash_query(query: str) -> str:
    """Deterministic short hash for cache keys (same convention as before)."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """
    Redis-backed two-tier semantic cache for verified RAG outputs.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        ttl_static_seconds: int = CACHE_TTL_STATIC_SECONDS,
        embed_fn: EmbedFn | None = None,
        rerank_fn: RerankFn | None = None,
        fast_hit_threshold: float = CACHE_FAST_HIT_THRESHOLD,
        fast_miss_threshold: float = CACHE_FAST_MISS_THRESHOLD,
        rerank_threshold: float = CACHE_RERANK_THRESHOLD,
        namespace: str = CACHE_NAMESPACE,
        socket_connect_timeout: int = 3,
    ):
        self._ttl = ttl_seconds
        self._ttl_static = ttl_static_seconds
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._fast_hit = fast_hit_threshold
        self._fast_miss = fast_miss_threshold
        self._rerank_threshold = rerank_threshold
        self._namespace = namespace
        self._redis = RedisClient(redis_url, socket_connect_timeout=socket_connect_timeout)
        self._reranker: Any = None

    # ------------------------------------------------------------------
    # Internal helpers (shared by sync + async paths)
    # ------------------------------------------------------------------

    def _key(self, query: str, ticker: str | None, fiscal_year: str | None) -> str:
        ticker = (ticker or "ANY").upper()
        year = fiscal_year or "ANY"
        return f"{self._namespace}:{ticker}:{year}:{_hash_query(query)}"

    def _resolve_ttl(self, ticker: str | None, fiscal_year: str | None) -> int:
        """Long TTL for canonical static SEC filings; session TTL otherwise."""
        if ticker and fiscal_year and ticker.upper() in SUPPORTED_TICKERS:
            return self._ttl_static
        return self._ttl

    def _build_entry(
        self,
        query: str,
        answer_json: str,
        guardrail_passed: bool,
        ticker: str | None,
        fiscal_year: str | None,
    ) -> dict[str, Any]:
        """Build the JSON-serialisable cache entry (embedding included when set)."""
        entry: dict[str, Any] = {
            "query": query,
            "query_hash": _hash_query(query),
            "answer_json": answer_json,
            "guardrail_passed": guardrail_passed,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "ticker": (ticker or "ANY").upper(),
            "fiscal_year": fiscal_year or "ANY",
        }
        if self._embed_fn is not None:
            try:
                entry["embedding"] = self._embed_fn(query)
            except Exception as exc:
                logger.warning("Cache entry embedding failed (storing exact-only): %s", exc)
        return entry

    def _embed_for_scan(self, query: str) -> list[float] | None:
        """Embed the query for semantic screening. None disables screening."""
        if self._embed_fn is None:
            return None
        try:
            return self._embed_fn(query)
        except Exception as exc:
            logger.warning("Query embedding failed (semantic screening skipped): %s", exc)
            return None

    async def _embed_for_scan_async(self, query: str) -> list[float] | None:
        """Embed the query off the event loop. None disables screening."""
        if self._embed_fn is None:
            return None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._embed_fn, query)
        except Exception as exc:
            logger.warning("Query embedding failed (semantic screening skipped): %s", exc)
            return None

    def _default_rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Lazily load bge-reranker-large (sigmoid-scaled 0..1) on first use."""
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._reranker = CrossEncoder(
                CACHE_RERANKER_MODEL,
                device=device,
                activation_fn=torch.nn.Sigmoid(),
            )
            logger.info("Loaded %s on %s (Tier B gray-zone reranking)", CACHE_RERANKER_MODEL, device)
        return self._reranker.predict([(query, c) for c in candidates]).tolist()

    def _rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Rerank candidate queries. Returns [] when reranking is unavailable."""
        fn = self._rerank_fn or self._default_rerank
        try:
            return fn(query, candidates)
        except Exception as exc:
            logger.warning("Reranking failed (gray zone treated as miss): %s", exc)
            return []

    @staticmethod
    def _parse_entry(raw: str | None) -> dict | None:
        """Parse a raw JSON cache value into an entry dict, or None if unusable."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _scan_namespace(
        self, client: Any, query_emb: list[float], ticker: str | None, fiscal_year: str | None
    ) -> list[tuple[float, dict]]:
        """Scan one namespace prefix (sync) and return (cosine, entry) pairs."""
        prefix = f"{self._namespace}:{(ticker or 'ANY').upper()}:{(fiscal_year or 'ANY').upper()}:"
        results: list[tuple[float, dict]] = []
        for key in client.scan_iter(match=prefix + "*", count=1000):
            entry = self._parse_entry(client.get(key))
            if entry is None:
                continue
            emb = entry.get("embedding")
            if not emb:
                continue
            results.append((_cosine(query_emb, emb), entry))
        return results

    async def _scan_namespace_async(
        self, client: Any, query_emb: list[float], ticker: str | None, fiscal_year: str | None
    ) -> list[tuple[float, dict]]:
        """Scan one namespace prefix (async) and return (cosine, entry) pairs."""
        prefix = f"{self._namespace}:{(ticker or 'ANY').upper()}:{(fiscal_year or 'ANY').upper()}:"
        results: list[tuple[float, dict]] = []
        async for key in client.scan_iter(match=prefix + "*", count=1000):
            entry = self._parse_entry(await client.get(key))
            if entry is None:
                continue
            emb = entry.get("embedding")
            if not emb:
                continue
            results.append((_cosine(query_emb, emb), entry))
        return results

    def _decide_semantic(self, query: str, scanned: list[tuple[float, dict]]) -> dict | None:
        """Tier A + Tier B decision over scored candidate entries."""
        if not scanned:
            return None
        scanned.sort(key=lambda item: item[0], reverse=True)

        best_score, best_entry = scanned[0]
        if best_score >= self._fast_hit:
            logger.info("Cache semantic HIT (Tier A score=%.3f)", best_score)
            return best_entry
        if best_score < self._fast_miss:
            logger.info("Cache semantic miss (Tier A score=%.3f < %.2f)", best_score, self._fast_miss)
            return None

        # Tier B — gray zone (0.60 <= best < 0.96)
        gray = [(s, e) for s, e in scanned if s >= self._fast_miss]
        rerank_scores = self._rerank(query, [e["query"] for _, e in gray])
        if rerank_scores:
            top = max(rerank_scores)
            if top >= self._rerank_threshold:
                idx = rerank_scores.index(top)
                logger.info("Cache semantic HIT (Tier B rerank=%.3f)", top)
                return gray[idx][1]
            logger.info("Cache semantic miss (Tier B rerank=%.3f < %.2f)", top, self._rerank_threshold)
            return None

        logger.info("Cache semantic miss (no rerank scores in gray zone)")
        return None

    # ------------------------------------------------------------------
    # Synchronous public API
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> dict | None:
        """
        Retrieve a cached answer (synchronous, pipeline fast-path).

        Returns the cached entry dict (with 'answer_json' and
        'guardrail_passed') or None on miss.
        """
        client = self._redis.get()
        if not client:
            return None

        key = self._key(query, ticker, fiscal_year)
        try:
            exact = self._parse_entry(client.get(key))
        except Exception as exc:
            logger.warning("Cache GET failed: %s", exc)
            return None
        if exact is not None:
            logger.info("Cache exact HIT (ns=%s)", key)
            return exact

        query_emb = self._embed_for_scan(query)
        if query_emb is None:
            return None

        try:
            scanned = self._scan_namespace(client, query_emb, ticker, fiscal_year)
        except Exception as exc:
            logger.warning("Semantic scan failed (treated as miss): %s", exc)
            return None
        return self._decide_semantic(query, scanned)

    def put(
        self,
        query: str,
        answer_json: str,
        guardrail_passed: bool = True,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> bool:
        """
        Write an answer to the cache (embedding included when an embedder is set).

        Returns True on success, False on failure / Redis unavailable.
        """
        client = self._redis.get()
        if not client:
            return False

        entry = self._build_entry(query, answer_json, guardrail_passed, ticker, fiscal_year)
        ttl = self._resolve_ttl(ticker, fiscal_year)
        try:
            client.set(self._key(query, ticker, fiscal_year), json.dumps(entry), ex=ttl)
            logger.info("Cache SET (ns=%s, ttl=%ds)", self._key(query, ticker, fiscal_year), ttl)
            return True
        except Exception as exc:
            logger.warning("Cache SET failed: %s", exc)
            return False

    def invalidate(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> bool:
        """Remove the exact cache entry for a query."""
        client = self._redis.get()
        if not client:
            return False
        try:
            return bool(client.delete(self._key(query, ticker, fiscal_year)))
        except Exception as exc:
            logger.warning("Cache DELETE failed: %s", exc)
            return False

    def flush_all(self) -> int:
        """Delete all cache keys (namespace-prefixed) from Redis.

        Returns the number of keys removed, or -1 on error / Redis unavailable.
        """
        client = self._redis.get()
        if not client:
            return -1
        try:
            keys = list(client.scan_iter(match=f"{self._namespace}:*", count=1000))
            if not keys:
                return 0
            count = client.delete(*keys)
            logger.info("Cache flushed: %d keys removed", count)
            return count
        except Exception as exc:
            logger.warning("Cache flush failed: %s", exc)
            return -1

    def _get_client(self) -> Any:
        """Expose the underlying sync Redis client (used by API warm-up)."""
        return self._redis.get()

    # ------------------------------------------------------------------
    # Asynchronous public API (non-blocking, background guardrail path)
    # ------------------------------------------------------------------

    async def aget(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> dict | None:
        """
        Retrieve a cached answer asynchronously.

        Same two-tier semantics as `get`, but all Redis I/O is non-blocking
        and the query embedding is computed off the event loop.
        """
        client = await self._redis.aget()
        if not client:
            return None

        key = self._key(query, ticker, fiscal_year)
        try:
            exact = self._parse_entry(await client.get(key))
        except Exception as exc:
            logger.warning("Cache GET failed (async): %s", exc)
            return None
        if exact is not None:
            logger.info("Cache exact HIT (ns=%s, async)", key)
            return exact

        query_emb = await self._embed_for_scan_async(query)
        if query_emb is None:
            return None

        try:
            scanned = await self._scan_namespace_async(client, query_emb, ticker, fiscal_year)
        except Exception as exc:
            logger.warning("Semantic scan failed (treated as miss, async): %s", exc)
            return None
        return self._decide_semantic(query, scanned)

    async def aput(
        self,
        query: str,
        answer_json: str,
        guardrail_passed: bool = True,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> bool:
        """
        Write an answer to the cache asynchronously.

        Returns True on success, False on failure / Redis unavailable.
        """
        client = await self._redis.aget()
        if not client:
            return False

        entry = await self._abuild_entry(
            query, answer_json, guardrail_passed, ticker, fiscal_year
        )
        ttl = self._resolve_ttl(ticker, fiscal_year)
        try:
            await client.set(self._key(query, ticker, fiscal_year), json.dumps(entry), ex=ttl)
            logger.info("Cache SET (ns=%s, ttl=%ds, async)", self._key(query, ticker, fiscal_year), ttl)
            return True
        except Exception as exc:
            logger.warning("Cache SET failed (async): %s", exc)
            return False

    async def _abuild_entry(
        self,
        query: str,
        answer_json: str,
        guardrail_passed: bool,
        ticker: str | None,
        fiscal_year: str | None,
    ) -> dict[str, Any]:
        """Build a cache entry with the embedding computed off the event loop."""
        entry: dict[str, Any] = {
            "query": query,
            "query_hash": _hash_query(query),
            "answer_json": answer_json,
            "guardrail_passed": guardrail_passed,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "ticker": (ticker or "ANY").upper(),
            "fiscal_year": fiscal_year or "ANY",
        }
        if self._embed_fn is not None:
            try:
                loop = asyncio.get_running_loop()
                entry["embedding"] = await loop.run_in_executor(None, self._embed_fn, query)
            except Exception as exc:
                logger.warning("Cache entry embedding failed (storing exact-only, async): %s", exc)
        return entry

    async def ainvalidate(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> bool:
        """Remove the exact cache entry for a query asynchronously."""
        client = await self._redis.aget()
        if not client:
            return False
        try:
            return bool(await client.delete(self._key(query, ticker, fiscal_year)))
        except Exception as exc:
            logger.warning("Cache DELETE failed (async): %s", exc)
            return False

    async def aflush_all(self) -> int:
        """Delete all cache keys (namespace-prefixed) from Redis asynchronously.

        Returns the number of keys removed, or -1 on error / Redis unavailable.
        """
        client = await self._redis.aget()
        if not client:
            return -1
        try:
            keys = [key async for key in client.scan_iter(match=f"{self._namespace}:*", count=1000)]
            if not keys:
                return 0
            count = await client.delete(*keys)
            logger.info("Cache flushed: %d keys removed (async)", count)
            return count
        except Exception as exc:
            logger.warning("Cache flush failed (async): %s", exc)
            return -1

    async def _aget_client(self) -> Any:
        """Expose the underlying async Redis client."""
        return await self._redis.aget()

    def close(self) -> None:
        """Close both sync and async Redis connections (sync context)."""
        self._redis.close()

    async def aclose(self) -> None:
        """Close the async Redis connection from an async context."""
        await self._redis.aclose()
