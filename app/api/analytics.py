"""
Analytics aggregation service for the Financial RAG dashboard.

Reads raw audit logs from MongoDB (``rag_audit_logs``) and the latest
evaluation scores from ``artifacts/evaluation_scores.json`` to produce a
summary suitable for the System Analytics tab.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pymongo import AsyncMongoClient

from config.logging_config import get_logger
from config.settings import (
    EVALUATION_SCORES_PATH,
    MONGODB_DB,
    MONGODB_URI,
    SUPPORTED_TICKERS,
)
from app.api.db_logger import AUDIT_COLLECTION_NAME

logger = get_logger("api.analytics")

# ---------------------------------------------------------------------------
# MongoDB helper (lazy singleton, async client)
# ---------------------------------------------------------------------------

_analytics_client: Optional[AsyncMongoClient] = None


async def _get_collection():
    """Return the ``rag_audit_logs`` collection (lazy-connect)."""
    global _analytics_client
    if _analytics_client is None:
        _analytics_client = AsyncMongoClient(
            MONGODB_URI, serverSelectionTimeoutMS=5000
        )
    db = _analytics_client[MONGODB_DB]
    return db[AUDIT_COLLECTION_NAME]


async def _get_raw_chunks_collection():
    """Return the ``raw_chunks`` collection (same client)."""
    global _analytics_client
    if _analytics_client is None:
        _analytics_client = AsyncMongoClient(
            MONGODB_URI, serverSelectionTimeoutMS=5000
        )
    db = _analytics_client[MONGODB_DB]
    return db["raw_chunks"]


# ---------------------------------------------------------------------------
# Evaluation scores reader (filesystem, cached after first read)
# ---------------------------------------------------------------------------

_eval_scores_cache: Optional[dict[str, Any]] = None


def _read_evaluation_scores() -> dict[str, Any]:
    """Read the latest evaluation scores JSON from the artifacts directory."""
    global _eval_scores_cache
    if _eval_scores_cache is not None:
        return _eval_scores_cache

    fallback = {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "sample_count": 0,
        "judge_model": "unknown",
    }

    scores_path = EVALUATION_SCORES_PATH
    if not scores_path.exists():
        logger.warning("Evaluation scores file not found at %s", scores_path)
        return fallback

    try:
        raw = json.loads(scores_path.read_text(encoding="utf-8"))
        aggregates = raw.get("aggregate_scores", raw)
        fallback = {
            "faithfulness": float(aggregates.get("faithfulness", 0.0)),
            "answer_relevance": float(aggregates.get("answer_relevance", 0.0)),
            "context_precision": float(aggregates.get("context_precision", 0.0)),
            "context_recall": float(aggregates.get("context_recall", 0.0)),
            "sample_count": int(raw.get("sample_count", 0)),
            "judge_model": raw.get("judge_model", "unknown"),
        }
        _eval_scores_cache = fallback
    except Exception as exc:
        logger.error("Failed to parse evaluation scores: %s", exc)

    return fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: int) -> float:
    """Compute the p-th percentile from a sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

async def get_analytics_summary(days: int = 30) -> dict[str, Any]:
    """Aggregate analytics from MongoDB audit logs + evaluation scores.

    All MongoDB operations are properly awaited.  On any DB failure a
    structured fallback dict is returned so the endpoint always responds
    with valid JSON.
    """
    _empty = {
        "performance": {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "avg_ttft_ms": 0.0,
            "cache_hit_rate": 0.0,
            "guardrail_pass_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        },
        "eval_scores": _read_evaluation_scores(),
        "context_quality": {"high": 0.0, "mid": 0.0, "low": 0.0, "total": 0},
        "volume": {"total_chunks": 0, "by_ticker": {}},
    }

    try:
        collection = await _get_collection()
    except Exception as exc:
        logger.warning("Analytics MongoDB connect failed: %s", exc)
        return _empty

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # --- Performance aggregation (async!) ------------------------------------
    perf_doc = _empty["performance"]
    try:
        # PyMongo >= 4.9 async API: AsyncCollection.aggregate() is a
        # coroutine -- it MUST be awaited to obtain the AsyncCursor,
        # otherwise the coroutine is never awaited (RuntimeWarning) and
        # to_list() raises AttributeError.
        cursor = await collection.aggregate(
            [
                {"$match": {"timestamp": {"$gte": cutoff}}},
                {
                    "$group": {
                        "_id": None,
                        "total_queries": {"$sum": 1},
                        "avg_latency_ms": {"$avg": "$execution_metadata.total_latency_ms"},
                        "avg_ttft_ms": {"$avg": "$execution_metadata.ttft_ms"},
                        "cache_hits": {
                            "$sum": {"$cond": ["$execution_metadata.cache_hit", 1, 0]}
                        },
                        "guardrail_passed": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$execution_metadata.guardrail_status", "passed"]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "all_latencies": {"$push": "$execution_metadata.total_latency_ms"},
                    }
                },
            ],
            allowDiskUse=True,
        )
        results = await cursor.to_list(length=1)
    except Exception as exc:
        logger.error("Performance aggregation failed: %s", exc)
        results = []

    if results:
        agg = results[0]
        total = max(agg["total_queries"], 1)
        perf_doc = {
            "total_queries": agg["total_queries"],
            "avg_latency_ms": round(agg.get("avg_latency_ms") or 0.0, 1),
            "avg_ttft_ms": round(agg.get("avg_ttft_ms") or 0.0, 1),
            "cache_hit_rate": round(agg["cache_hits"] / total * 100, 1),
            "guardrail_pass_rate": round(agg["guardrail_passed"] / total * 100, 1),
        }
        latencies = sorted(v for v in agg.get("all_latencies", []) if v is not None)
        if latencies:
            perf_doc["p50_latency_ms"] = round(_percentile(latencies, 50), 1)
            perf_doc["p95_latency_ms"] = round(_percentile(latencies, 95), 1)
            perf_doc["p99_latency_ms"] = round(_percentile(latencies, 99), 1)
        else:
            perf_doc["p50_latency_ms"] = 0.0
            perf_doc["p95_latency_ms"] = 0.0
            perf_doc["p99_latency_ms"] = 0.0

    # --- Context quality distribution (async!) --------------------------------
    cq_doc = _empty["context_quality"]
    try:
        cq_cursor = await collection.aggregate(
            [
                {"$match": {"timestamp": {"$gte": cutoff}}},
                {"$unwind": "$retrieved_chunks"},
                {"$match": {"retrieved_chunks.score": {"$exists": True, "$ne": None}}},
                {
                    "$group": {
                        "_id": None,
                        "high": {"$sum": {"$cond": [{"$gt": ["$retrieved_chunks.score", 0.8]}, 1, 0]}},
                        "mid": {
                            "$sum": {
                                "$cond": [
                                    {"$and": [
                                        {"$gte": ["$retrieved_chunks.score", 0.5]},
                                        {"$lte": ["$retrieved_chunks.score", 0.8]},
                                    ]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "low": {"$sum": {"$cond": [{"$lt": ["$retrieved_chunks.score", 0.5]}, 1, 0]}},
                        "total": {"$sum": 1},
                    }
                },
            ],
            allowDiskUse=True,
        )
        cq_results = await cq_cursor.to_list(length=1)
        if cq_results:
            agg = cq_results[0]
            total = max(agg["total"], 1)
            cq_doc = {
                "high": round(agg["high"] / total * 100, 1),
                "mid": round(agg["mid"] / total * 100, 1),
                "low": round(agg["low"] / total * 100, 1),
                "total": agg["total"],
            }
    except Exception as exc:
        logger.error("Context quality aggregation failed: %s", exc)

    # --- Volume stats (async!) ------------------------------------------------
    vol_doc = _empty["volume"]
    try:
        raw_chunks = await _get_raw_chunks_collection()
        total_chunks = await raw_chunks.count_documents({})
        by_ticker: dict[str, int] = {}
        for tk in SUPPORTED_TICKERS:
            by_ticker[tk] = await raw_chunks.count_documents({"ticker": tk})
        vol_doc = {"total_chunks": total_chunks, "by_ticker": by_ticker}
    except Exception as exc:
        logger.warning("Volume stats query failed: %s", exc)

    return {
        "performance": perf_doc,
        "eval_scores": _read_evaluation_scores(),
        "context_quality": cq_doc,
        "volume": vol_doc,
    }
