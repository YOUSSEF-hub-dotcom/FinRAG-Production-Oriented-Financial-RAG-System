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

# D3: any latency above this is treated as a clock/anomaly outlier (the wall
# clock produced hour-scale values) and excluded from mean/percentile math.
_MAX_LATENCY_MS = 600_000.0  # 10 minutes

# D4: context-quality bucket cut-offs applied to the NORMALIZED score (the
# [0,1] value produced by ``_normalize_audit_score`` on the write path), not a
# raw cosine. Equal-width tertiles keep the distribution non-degenerate.
_CONTEXT_HIGH_THRESHOLD = 0.66
_CONTEXT_MID_THRESHOLD = 0.33

# D2: only events that actually carry a guardrail verdict are eligible for the
# pass-rate denominator (stream/upload events have no verdict and must not
# dilute it).
_ELIGIBLE_GUARDRAIL_STATUSES = ["passed", "failed"]

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
            "sample_count": int(raw.get("sample_count", len(raw.get("samples", [])))),
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
            "eligible_queries": 0,
            "avg_latency_ms": 0.0,
            "avg_ttft_ms": 0.0,
            "cache_hit_rate": 0.0,
            "guardrail_pass_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "latency_outliers_dropped": 0,
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
                        "guardrail_eligible": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$in": [
                                            "$execution_metadata.guardrail_status",
                                            _ELIGIBLE_GUARDRAIL_STATUSES,
                                        ]
                                    },
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
        eligible = int(agg.get("guardrail_eligible", 0) or 0)

        # D3: drop non-positive, non-numeric, and implausible (clock-step)
        # latencies before computing the mean/percentiles, and surface how many
        # were excluded so the dashboard can flag data quality.
        all_latencies = agg.get("all_latencies", []) or []
        latencies = sorted(
            v
            for v in all_latencies
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and 0 < v <= _MAX_LATENCY_MS
        )
        dropped = sum(1 for v in all_latencies if v is not None) - len(latencies)

        perf_doc = {
            "total_queries": agg["total_queries"],
            "eligible_queries": eligible,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "avg_ttft_ms": round(agg.get("avg_ttft_ms") or 0.0, 1),
            "cache_hit_rate": round(agg["cache_hits"] / total * 100, 1),
            # D2: denominator is the eligible set (events with a real verdict),
            # never all matched events. No eligible events -> 0.0 (no div-by-zero).
            "guardrail_pass_rate": (
                round(agg["guardrail_passed"] / eligible * 100, 1) if eligible else 0.0
            ),
            "latency_outliers_dropped": dropped,
        }
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
                        # D4: buckets are applied to the NORMALIZED score (both
                        # sync and stream paths write ``_normalize_audit_score``
                        # output), using non-overlapping tertile cut-offs.
                        "high": {
                            "$sum": {
                                "$cond": [
                                    {"$gte": ["$retrieved_chunks.score", _CONTEXT_HIGH_THRESHOLD]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "mid": {
                            "$sum": {
                                "$cond": [
                                    {"$and": [
                                        {"$gte": ["$retrieved_chunks.score", _CONTEXT_MID_THRESHOLD]},
                                        {"$lt": ["$retrieved_chunks.score", _CONTEXT_HIGH_THRESHOLD]},
                                    ]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "low": {
                            "$sum": {
                                "$cond": [
                                    {"$lt": ["$retrieved_chunks.score", _CONTEXT_MID_THRESHOLD]},
                                    1,
                                    0,
                                ]
                            }
                        },
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
