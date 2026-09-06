"""
Background Task Worker for post-response processing.

Default path: FastAPI BackgroundTasks run guardrail verification, cache updates,
and metric logging off the main HTTP thread so API responses remain fast.

Production async queue: Arq (Redis-backed) worker. Run with:
    arq app.api.worker.WorkerSettings
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings, create_pool

from config.logging_config import get_logger
from config.settings import ARQ_JOB_TIMEOUT_SECONDS, ARQ_MAX_JOBS, REDIS_URL

# Ensure src subdirectories are importable
_src_root = Path(__file__).resolve().parent.parent.parent / "src"
for _subdir in ("1_ingestion", "5_generation"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_src_root_str = str(_src_root)
_project_root_str = str(_src_root.parent)
for _p in (_src_root_str, _project_root_str):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = get_logger("api.worker")


# ---------------------------------------------------------------------------
# Background guardrail runner
# ---------------------------------------------------------------------------

async def run_guardrail_background(
    query: str,
    answer: str,
    extracted_raw_data: str,
    parsed_output: Any = None,
) -> dict[str, Any]:
    """
    Run deep numerical guardrail verification as a background task.

    Args:
        query: Original user query.
        answer: Generated answer text.
        extracted_raw_data: Verbatim extracted facts from context.
        parsed_output: Optional ConsolidatedFinancialAnswer for caching.

    Returns:
        Dict with guardrail outcome (passed, verified_claims, failed_claims, etc.).
    """
    from async_guardrail import AsyncGuardrail

    t0 = time.time()
    guardrail = AsyncGuardrail()

    try:
        result = await guardrail.check(
            query=query,
            answer=answer,
            extracted_raw_data=extracted_raw_data,
            parsed_output=parsed_output,
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Background guardrail: passed=%s verified=%d failed=%d elapsed=%.1fms",
            result.get("passed"),
            len(result.get("verified_claims", [])),
            len(result.get("failed_claims", [])),
            elapsed_ms,
        )
        return result
    except Exception as exc:
        logger.warning("Background guardrail failed: %s", exc)
        return {
            "passed": False,
            "verified_claims": [],
            "failed_claims": [],
            "final_output": answer,
            "cache_written": False,
            "detail": f"Guardrail error: {exc}",
        }
    finally:
        guardrail.close()


# ---------------------------------------------------------------------------
# Background metric logging
# ---------------------------------------------------------------------------

def log_metrics_background(metrics: dict[str, Any]) -> None:
    """
    Log pipeline metrics as a fire-and-forget background task.

    Args:
        metrics: Dict of metric names to values (int/float/str).
    """
    try:
        import mlflow

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, str):
                mlflow.log_param(key, value)
        logger.debug("Background metrics logged: %s", metrics)
    except Exception as exc:
        logger.warning("Background metric logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Arq worker (Redis-backed async queue) — non-blocking task processing
# ---------------------------------------------------------------------------

async def ingest_document_job(
    ctx: dict[str, Any],
    file_path: str,
    ticker: str,
    fiscal_year: str,
    task_id: str,
) -> dict[str, Any]:
    """
    Arq task: run the full ingestion pipeline for an uploaded document.

    Imported lazily inside the coroutine to avoid a module-level import cycle
    with ``app.api.main`` (which imports ``worker`` at startup). The blocking
    ingestion work is delegated to ``asyncio.to_thread`` so a single worker
    keeps serving other queued jobs while a large filing embeds on CUDA.

    Args:
        ctx: Arq worker context (unused here — jobs are self-contained).
        file_path: Absolute path to the saved uploaded file.
        ticker: Ticker the document belongs to.
        fiscal_year: Fiscal year the document belongs to.
        task_id: Short task identifier used in logs / response body.

    Returns:
        Dict with ingestion outcome (chunks_created, mongo_count, qdrant_count).
    """
    from app.api.main import _ingest_document

    result = await asyncio.to_thread(
        _ingest_document, Path(file_path), ticker, fiscal_year, task_id
    )
    return result or {"status": "no_chunks"}


async def enqueue_ingestion(
    file_path: Path,
    ticker: str,
    fiscal_year: str,
    task_id: str,
) -> bool:
    """
    Enqueue a document-ingestion job on the Arq worker pool.

    Args:
        file_path: Absolute path to the saved uploaded file.
        ticker: Ticker the document belongs to.
        fiscal_year: Fiscal year the document belongs to.
        task_id: Short task identifier used in logs / response body.

    Returns:
        True if the job was enqueued, False on Redis/queue failure.
    """
    try:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            await pool.enqueue_job(
                "ingest_document_job",
                str(file_path),
                ticker,
                fiscal_year,
                task_id,
            )
            logger.info("Arq job enqueued: task=%s file=%s", task_id, file_path.name)
            return True
        finally:
            await pool.aclose()
    except Exception as exc:
        logger.error("Arq enqueue failed for task=%s: %s", task_id, exc)
        return False


# ---------------------------------------------------------------------------
# Arq worker: RAG audit-log persistence (Module 7 — Advanced API Feature #2)
# ---------------------------------------------------------------------------


async def log_audit_event_job(
    ctx: dict[str, Any],
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Arq task: persist a single RAG audit event to MongoDB.

    Runs off the request path on the worker pool so the API process never
    blocks on database writes. The blocking pymongo driver call is delegated
    to a thread so the async worker keeps serving other queued jobs.

    Args:
        ctx: Arq worker context (unused -- jobs are self-contained).
        event_dict: Dynamic audit payload (see ``AuditLogEvent`` schema).

    Returns:
        Dict with the outcome (logged True/False).
    """
    from app.api.db_logger import MongoAuditLogger

    logger_ = MongoAuditLogger()
    try:
        inserted_id = await logger_.log_event(event_dict)
        await logger_.close()
        if inserted_id is None:
            return {"logged": False}
        return {"logged": True, "inserted_id": inserted_id}
    except Exception as exc:
        logger.error("Arq audit log job failed: %s", exc, exc_info=True)
        return {"logged": False, "error": str(exc)}


async def enqueue_audit_log(event_dict: dict[str, Any]) -> bool:
    """
    Enqueue an audit-log persistence job on the Arq worker pool.

    Args:
        event_dict: Dynamic audit payload (see ``AuditLogEvent`` schema).

    Returns:
        True if the job was enqueued, False on Redis/queue failure.
    """
    try:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        try:
            await pool.enqueue_job("log_audit_event_job", event_dict)
            logger.info(
                "Arq audit-log job enqueued (request_id=%s)",
                event_dict.get("request_id"),
            )
            return True
        finally:
            await pool.aclose()
    except Exception as exc:
        logger.error("Arq audit-log enqueue failed: %s", exc)
        return False


class WorkerSettings:
    """Arq worker configuration -- start with ``arq app.api.worker.WorkerSettings``."""

    functions: list[Any] = [ingest_document_job, log_audit_event_job]
    redis_settings: Any = RedisSettings.from_dsn(REDIS_URL)
    max_jobs: int = ARQ_MAX_JOBS
    job_timeout: int = ARQ_JOB_TIMEOUT_SECONDS
