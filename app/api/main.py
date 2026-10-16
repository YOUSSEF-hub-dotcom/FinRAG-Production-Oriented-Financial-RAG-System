"""
FastAPI Production Backend for Financial RAG System.

Endpoints:
  - POST /api/v1/chat         → Main query (sync)
  - POST /api/v1/chat/stream  → SSE streaming
  - POST /api/v1/documents/upload → File upload + ingestion
  - GET  /health              → Service health check

Lifespan manages FinancialRAGPipeline init/close.
Structured JSON logging on every request via middleware.
"""

import asyncio
import json
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from config.logging_config import get_logger
from config.settings import DATA_DIR, GUARDRAIL_ENFORCE, SUPPORTED_TICKERS, USE_ARQ_QUEUE

# Ensure src subdirectories are importable (numbered names not valid packages)
_src_root = Path(__file__).resolve().parent.parent.parent / "src"
for _subdir in ("1_ingestion", "2_caching", "5_generation"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Also add src/ itself and project root
_src_root_str = str(_src_root)
_project_root_str = str(_src_root.parent)
for _p in (_src_root_str, _project_root_str):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from streaming_json import StreamingAnswerExtractor
from app.api.db_logger import get_audit_logger
from app.api.auth import admin_router, auth_router, seed_super_admin
from app.api.auth.dependencies import UserPrincipal, require_roles
from app.api.parsers import APIFileParser
from app.api.rate_limiter import (
    get_current_user,
    limiter,
)
from app.api.schemas import (
    AuditLogEvent,
    CacheFlushResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    DocumentUploadResponse,
    ErrorResponse,
    GuardrailStatus,
    HealthCheckResponse,
    HealthServiceStatus,
    IngestionTaskStatusResponse,
    RateLimitErrorResponse,
    SourceCitation,
    AnalyticsSummaryResponse,
)
from app.api.worker import run_guardrail_background
from model_loader import get_or_create_pipeline
from pipeline import FinancialRAGPipeline

logger = get_logger("api.main")


# ---------------------------------------------------------------------------
# Upload identity + safe persistence helpers (D6-A)
# ---------------------------------------------------------------------------

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_path_component(value: Any, default: str = "unknown") -> str:
    """Sanitize a single path segment to prevent traversal / odd characters."""
    cleaned = _UNSAFE_PATH_CHARS.sub("_", str(value).strip()).strip("._")
    return cleaned or default


def _resolve_upload_identity(raw_bytes: bytes, source_id: str | None) -> str | None:
    """Resolve the logical-filing ``source_id`` for an upload.

    An explicitly supplied ``source_id`` wins; otherwise a single validated SEC
    accession number is extracted from the filing header. Returns ``None`` when
    identity cannot be determined unambiguously (never guessed).
    """
    if source_id:
        return source_id
    from metadata_extractor import extract_accession

    preview = raw_bytes[:200_000].decode("utf-8", errors="replace")
    return extract_accession(None, preview)


def _resolve_upload_save_path(
    ticker: str,
    fiscal_year: str,
    filename: str,
    source_id: str | None,
    doc_type: str | None,
) -> Path:
    """Return a non-colliding destination path for an uploaded document.

    Legacy layout is preserved when no ``source_id`` is supplied:
    ``DATA_DIR/{ticker}/10-K/{fiscal_year}/{filename}``. When identity is known
    the file is isolated under ``.../{doc_type|10-K}/{fiscal_year}/{source_id}/``
    so re-uploads of the same filing overwrite intentionally while distinct
    filings never collide. A filename collision appends a short uuid prefix so
    an existing file is never silently overwritten.
    """
    doc_segment = _safe_path_component(doc_type or "10-K", default="10-K")
    save_dir = (
        DATA_DIR
        / _safe_path_component(ticker)
        / doc_segment
        / _safe_path_component(fiscal_year)
    )
    if source_id:
        save_dir = save_dir / _safe_path_component(source_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    target = save_dir / safe_name
    if target.exists():
        target = save_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    return target


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: init pipeline via MLflow Production alias → fallback. Shutdown: close connections."""
    logger.info("API starting up — loading FinancialRAGPipeline")

    # Dynamic loading: tries MLflow Production alias, falls back to direct init
    pipeline = get_or_create_pipeline()
    # Fail fast if enforcement is enabled but the guardrail is unavailable:
    # otherwise financial/rewrite answers would be released unverified.
    _validate_guardrail_config(pipeline)
    app.state.pipeline = pipeline
    app.state.warmup_completed = False

    # Warm-up phase — pre-load models & establish connections
    try:
        await _run_warmup(pipeline)
        app.state.warmup_completed = True
    except Exception as exc:
        logger.warning("Warm-up failed (non-blocking): %s", exc)

    # Bootstrap the Super Admin account if none exists (best-effort, env-gated).
    try:
        await seed_super_admin()
    except Exception as exc:
        logger.warning("Super Admin bootstrap failed (non-blocking): %s", exc)

    yield

    logger.info("API shutting down — closing pipeline")
    pipeline.close()


app = FastAPI(
    title="Financial RAG API",
    version="1.0.0",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Advanced Rate Limiting (slowapi + Redis)
# ---------------------------------------------------------------------------
# The limiter instance lives in app.api.rate_limiter and uses a dynamic key
# function: authenticated users are bucketed by user_id, guests by client IP.
# We register it on app.state (required by slowapi) and an exception handler
# that returns a structured HTTP 429 payload.
app.state.limiter = limiter
if hasattr(limiter, "init_app"):
    limiter.init_app(app)

# Register the AuthN/AuthZ routers (signup/login/refresh/logout/me/change-password
# + admin role management & audit logs). RBAC is enforced inside those routers.
app.include_router(auth_router)
app.include_router(admin_router)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Return a structured JSON 429 when a rate limit is exceeded."""
    limiter_instance = getattr(request.app.state, "limiter", None)
    response = JSONResponse(
        status_code=429,
        content=RateLimitErrorResponse(
            detail="Rate limit exceeded: too many requests. Please retry shortly.",
            error_code="RATE_LIMIT_EXCEEDED",
            limit="10/minute",
            retry_after_seconds=60,
        ).model_dump(),
    )
    # Surface X-RateLimit-* headers when available.
    if limiter_instance is not None:
        try:
            response = limiter_instance._inject_headers(
                response, getattr(request.state, "view_rate_limit", None)
            )
        except Exception:
            pass
    return response


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


async def _run_warmup(pipeline: "FinancialRAGPipeline") -> None:
    """Pre-load embedding model and pre-warm DB connections.

    Runs a lightweight embedding call to load nomic-embed-text-v1.5 into
    GPU memory (the #1 cold-start bottleneck), then pings Qdrant, MongoDB
    and Redis so the first user query sees sub-second response times.
    All errors are caught and logged — a warm-up failure never crashes
    the server.
    """
    # 1. Embedding model — the main cold-start cost (~2–3 min first load)
    logger.info("Warm-up: loading embedding model...")
    pipeline._embedding_engine.embed_single("warmup")
    logger.info("Warm-up: embedding model loaded")

    # 1b. Reranker (CrossEncoder) — previously lazy-loaded on the first user
    # request (~29s cold). Pre-loading it here moves that cost into startup so
    # the first query does not pay the model-load spike.
    logger.info("Warm-up: loading reranker...")
    _rerank_t0 = time.perf_counter()
    pipeline.warm_reranker()
    logger.info(
        "Warm-up: reranker loaded in %.1fs",
        time.perf_counter() - _rerank_t0,
    )

    # 2. Qdrant persistent client connection
    logger.info("Warm-up: connecting to Qdrant...")
    pipeline._qdrant_indexer.count_points()
    logger.info("Warm-up: Qdrant ready")

    # 3. MongoDB connection
    logger.info("Warm-up: connecting to MongoDB...")
    pipeline._mongo_indexer.count_documents()
    logger.info("Warm-up: MongoDB ready")

    # 4. Redis cache connection (lazy-init)
    logger.info("Warm-up: connecting to Redis...")
    if pipeline._cache is not None:
        pipeline._cache._get_client()
    logger.info("Warm-up: Redis ready")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP %d on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail, error_code=f"HTTP_{exc.status_code}").model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 on invalid payloads — returns a clean structured ErrorResponse."""
    logger.warning(
        "Validation error on %s: %s",
        request.url.path,
        json.dumps(exc.errors(), default=str)[:500],
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            detail="Invalid request payload.",
            error_code="VALIDATION_ERROR",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error. Please try again later.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Middleware: structured request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = await call_next(request)
    elapsed_ms = _elapsed_ms(t0)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "api_request request_id=%s method=%s path=%s status=%d elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# CORS — must be added AFTER log_requests so CORSMiddleware wraps it
# (Starlette reverses add_middleware order: last added = outermost).
# Handles preflight OPTIONS before any other middleware.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://172.23.42.43:3000",   # WSL2 network access
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,       # Chrome sends PNA header for localhost
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(request: Request) -> FinancialRAGPipeline:
    """Get pipeline from app state (raises 503 if unavailable)."""
    pipe: FinancialRAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipe


# ---------------------------------------------------------------------------
# Per-session conversation memory
# ---------------------------------------------------------------------------
# The shared pipeline singleton holds a single ConversationMemory. To support
# multi-turn coreference (e.g. "first company" -> Apple) without purging
# context on every request AND without leaking history across different
# callers, we key the message history by ``session_id`` and swap the live
# memory's underlying list per request. Anonymous callers (no session_id)
# keep the original stateless behaviour (fresh memory each turn).
_SESSION_MEMORY: dict[str, list] = {}


def _restore_session_memory(pipe, session_id: str | None) -> None:
    """Load this session's prior turns into the pipeline, or start fresh."""
    if session_id and session_id in _SESSION_MEMORY:
        pipe._generator._memory._messages = list(_SESSION_MEMORY[session_id])
    else:
        pipe.reset_memory()


def _persist_session_memory(pipe, session_id: str | None) -> None:
    """Save the post-query conversation history for this session."""
    if session_id:
        _SESSION_MEMORY[session_id] = list(pipe._generator._memory.messages)


def _clean_meta_str(value: Any) -> str:
    """Normalise a metadata string; null-ish/N-A/UNKNOWN sentinels become ''."""
    if value is None:
        return ""
    cleaned = str(value).strip()
    if cleaned.lower() in ("", "n/a", "na", "none", "null", "unknown"):
        return ""
    return _strip_internal_markers(cleaned)


_INTERNAL_MARKER_RES = (
    re.compile(r"%%TABLE_[0-9_]*"),
    re.compile(r"\[\s*TABLE\s+\d+\s*\]"),
)


def _strip_internal_markers(value: Any) -> str:
    """Purge internal ``%%TABLE_N%%`` / ``[TABLE N]`` markers from output.

    These artifacts are ingestion/context-building internals. They can leak
    into the user-visible answer via the LLM echoing a section label or via the
    enriched source text, so they are removed at the API output boundary
    (defensive; the pipeline already sanitizes before context formatting).
    """
    if value is None:
        return ""
    text = str(value)
    for pattern in _INTERNAL_MARKER_RES:
        text = pattern.sub("", text)
    text = text.replace("%%", "")
    # Collapse internal whitespace exposed by marker removal, then trim.
    return " ".join(text.split())


# --- Analytics relevance normalization (audit logging only) -----------------
# Raw Qdrant cosine similarities for this embedding model cluster tightly
# (~0.55-0.80 for relevant text), so the canonical 0.5/0.8 bucket cut-offs
# pin every chunk into "Mid". Audit events therefore store relevance RESCALED
# onto [0, 1] over the model's realistic band, keeping the High/Mid/Low
# buckets meaningful. Retrieval/generation/guardrail logic is untouched --
# this only affects what analytics logging persists.
_AUDIT_SCORE_FLOOR = 0.40  # cosine below this ~ irrelevant context
_AUDIT_SCORE_CEIL = 0.85   # cosine at/above this ~ near-duplicate relevance


# --- Latency measurement (monotonic) + clamping (D3) -------------------------
# Durations are measured with ``time.perf_counter()`` (monotonic, immune to wall
# clock steps) and clamped to a sane band so a stuck/clock-anomalous request can
# never persist a negative or hour-scale latency into the audit trail.
_MAX_LATENCY_MS = 600_000.0  # 10 minutes


def _elapsed_ms(start: float) -> float:
    """Return clamped elapsed milliseconds since a ``perf_counter()`` mark."""
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed < 0:
        return 0.0
    return elapsed if elapsed <= _MAX_LATENCY_MS else _MAX_LATENCY_MS


# --- Ingestion task status registry ------------------------------------------
# Tracks background ingestion tasks so the frontend can poll their progress via
# GET /api/v1/documents/tasks/{task_id}. In-memory because ingestion runs
# in-process (FastAPI BackgroundTasks) by default; with USE_ARQ_QUEUE=true the
# worker is a separate process and tasks stay "queued" from the API's view.
_INGESTION_TASKS: dict[str, dict[str, Any]] = {}


def _normalize_audit_score(raw: Any) -> float:
    """Rescale a raw similarity score onto [0, 1] over the realistic band.

    Returns 0.0 for scoreless/injected chunks so they stay out of the
    analytics buckets.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 0.0
    span = _AUDIT_SCORE_CEIL - _AUDIT_SCORE_FLOOR
    normalized = (value - _AUDIT_SCORE_FLOOR) / span if span > 0 else value
    return round(min(max(normalized, 0.0), 1.0), 4)


def _context_chunk_citation_score(ctx: dict, citation: str) -> int:
    """Score one request-context chunk against an LLM citation string.

    Matches on metadata tokens (ticker/fiscal_year/section) plus a light
    content-token overlap, so a chunk is only ranked for an LLM citation when
    there is genuine signal — never a bare ticker coincidence. Neutral chunks
    score 0 (they are not excluded; they just rank below clearly-cited chunks).
    """
    if not citation:
        return 0
    payload_ = ctx.get("payload") or {}
    cit = citation.lower()
    rel = 0
    tk = _clean_meta_str(payload_.get("ticker")).lower()
    fy = _clean_meta_str(payload_.get("fiscal_year")).lower()
    sec = _clean_meta_str(payload_.get("section")).lower()
    if tk and tk in cit:
        rel += 3
    if fy and fy in cit:
        rel += 2
    if sec and len(sec) > 2 and sec in cit:
        rel += 1
    head = _strip_internal_markers(
        str(payload_.get("raw_text") or payload_.get("text") or "")
    )[:200].lower()
    overlap = {w for w in head.replace("$", " ").split() if len(w) > 4} & set(cit.split())
    if len(overlap) >= 2:
        rel += 2
    return rel


def _rank_contexts_by_citations(
    contexts: list[dict],
    citation_strings: list[str],
) -> list[dict]:
    """Order request-scoped context chunks, LLM-cited chunks first.

    ``contexts`` is the exact retrieval/grounding set for THIS request. Only
    chunks present in it can ever be ranked or returned — unrelated chunks from
    a prior query/cache cannot leak in. Ordering is a stable sort by
    (citation-match score desc, retrieval score desc). Contexts without any
    citation signal keep their retrieval-score ordering.
    """
    best: dict[str, int] = {}
    for citation in (citation_strings or []):
        if not citation or not citation.strip():
            continue
        top_cid, top_rel = None, 0
        for ctx in contexts:
            cid = str(ctx.get("chunk_id") or "")
            if not cid:
                continue
            rel = _context_chunk_citation_score(ctx, citation)
            if rel > top_rel:
                top_cid, top_rel = cid, rel
        if top_cid:
            best[top_cid] = max(best.get(top_cid, 0), top_rel)

    def _rank(ctx: dict) -> tuple[int, float]:
        cid = str(ctx.get("chunk_id") or "")
        return (best.get(cid, 0), float(ctx.get("score") or 0.0))

    return sorted(contexts, key=_rank, reverse=True)


def _source_record_from_context(ctx: dict, snippet: int = 800) -> dict:
    """Build a client-facing source record from one request-context chunk.

    Metadata that is missing/incomplete is emitted as an empty string (never
    fabricated); the chunk_id is always the real retrieved chunk id.
    """
    payload_ = ctx.get("payload") or {}
    content = str(payload_.get("raw_text") or payload_.get("text") or "")[:snippet]
    return {
        "chunk_id": str(ctx.get("chunk_id") or ""),
        "text": _strip_internal_markers(content) if content else "",
        "score": float(ctx.get("score") or 0.0),
        "ticker": _clean_meta_str(payload_.get("ticker")),
        "fiscal_year": _clean_meta_str(payload_.get("fiscal_year")),
        "section": _clean_meta_str(payload_.get("section")),
        "source_file": _clean_meta_str(payload_.get("source_file")),
        "page_number": payload_.get("page_number"),
    }


def _build_stream_sources(
    raw_source_strings: list[str],
    request_contexts: list[dict],
    limit: int = 6,
) -> list[dict]:
    """Build the SSE ``sources`` payload strictly from THIS request's context.

    Rejects empty contexts (no sources emitted rather than fabricating one),
    deduplicates by chunk_id, and never consults pipeline-global stores.
    """
    good = [ctx for ctx in (request_contexts or []) if (ctx.get("chunk_id") or "").strip()]
    if not good:
        return []
    ordered = _rank_contexts_by_citations(good, raw_source_strings)
    seen: set[str] = set()
    sources: list[dict] = []
    for ctx in ordered:
        cid = str(ctx.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        sources.append(_source_record_from_context(ctx))
        if len(sources) >= limit:
            break
    return sources


def _build_sources(result: dict[str, Any]) -> list[SourceCitation]:
    """Extract source citations from pipeline result.

    Sources are built from the per-request ``retrieved_chunks`` grounding
    context populated by ``query()``. LLM citation strings (e.g.
    "NVIDIA - 2026 - Revenue") are used ONLY to reorder that same context via
    token matching — they are never themselves emitted as chunk ids, so a
    stale/unrelated citation can never surface. Missing metadata degrades to
    empty values; an empty context yields no citations.
    """
    retrieved = result.get("retrieved_chunks") or []
    good = [c for c in retrieved if (c.get("chunk_id") or "").strip()]
    if not good:
        return []

    parsed = result.get("parsed")
    citation_strings: list[str] = []
    if parsed is not None and hasattr(parsed, "sources"):
        citation_strings = list(parsed.sources or [])
    ordered = _rank_contexts_by_citations(good, citation_strings)

    seen: set[str] = set()
    sources: list[SourceCitation] = []
    for ctx in ordered:
        cid = str(ctx.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        payload_ = ctx.get("payload") or {}
        sources.append(SourceCitation(
            chunk_id=cid,
            score=min(float(ctx.get("score") or 0.0), 1.0),
            ticker=_clean_meta_str(payload_.get("ticker")) or "UNKNOWN",
            fiscal_year=_clean_meta_str(payload_.get("fiscal_year")) or "UNKNOWN",
            section=_clean_meta_str(payload_.get("section")) or "General",
            text_snippet=_strip_internal_markers(
                str(payload_.get("raw_text") or payload_.get("text") or "")[:300]
            ),
        ))
        if len(sources) >= 6:
            break
    return sources


# ---------------------------------------------------------------------------
# Audit logging helpers (Advanced API Feature #2)
# ---------------------------------------------------------------------------


def _build_audit_event(
    request_id: str,
    user_identifier: str,
    payload: ChatQueryRequest,
    result: dict[str, Any],
    answer: str,
    sources: list[SourceCitation],
    elapsed_ms: float,
) -> AuditLogEvent:
    """Construct a dynamic :class:`AuditLogEvent` from a completed RAG call.

    Best-effort extraction: the pipeline exposes different telemetry keys across
    code paths, so every section degrades gracefully when a field is missing.
    """
    # --- retrieved chunks (ids, NORMALIZED scores, metadata, ticker tags) ---
    # D4: persist the same normalized [0,1] score the stream path writes
    # (``_normalize_audit_score``) and drop scoreless/injected chunks, so the
    # analytics context-quality buckets are comparable across both paths.
    retrieved_context = result.get("retrieved_chunks") or []
    retrieved: list[dict[str, Any]] = []
    for ctx in retrieved_context:
        try:
            raw_score = float(ctx.get("score") or 0.0)
        except (TypeError, ValueError):
            raw_score = 0.0
        if raw_score <= 0:
            continue
        payload_ = ctx.get("payload") or {}
        retrieved.append(
            {
                "chunk_id": str(ctx.get("chunk_id") or ""),
                "score": _normalize_audit_score(raw_score),
                "ticker": _clean_meta_str(payload_.get("ticker")),
                "fiscal_year": _clean_meta_str(payload_.get("fiscal_year")),
                "section": _clean_meta_str(payload_.get("section")),
                "ticker_tag": _clean_meta_str(payload_.get("ticker")) or "UNKNOWN",
            }
        )
    if not retrieved and result.get("parsed") is not None:
        parsed = result["parsed"]
        if hasattr(parsed, "sources"):
            retrieved = [
                {
                    "chunk_id": s,
                    "ticker_tag": str(s).split(" - ")[0] if " - " in str(s) else "UNKNOWN",
                }
                for s in parsed.sources
            ]

    # --- execution metadata (latency, tokens, cost, guardrail, errors) ---
    execution_metadata: dict[str, Any] = {
        "total_latency_ms": round(elapsed_ms, 2),
        "ttft_ms": result.get("ttft_ms"),
        "token_usage": result.get("token_usage", {}),
        "model_cost_usd": result.get("cost_usd"),
        "model_used": result.get("model_used", "unknown"),
        "cache_hit": result.get("cache_hit", False),
        # D1: persist the REAL verdict. ``not_run`` covers paths where the
        # guardrail never executed (disabled, GENERAL route, security halt) so
        # analytics can never mistake an unverified answer for a PASS.
        "guardrail_status": result.get("guardrail_status", "not_run"),
        # True only when enforcement changed the delivered answer (FAIL swap or
        # evaluator-error withhold); always False while advisory.
        "guardrail_enforced": bool(result.get("guardrail_enforced", False)),
        "errors": result.get("error"),
    }

    return AuditLogEvent(
        request_id=request_id,
        user_identifier=user_identifier,
        user_query={
            "raw": payload.user_query,
            "standalone": result.get("expanded_query", payload.user_query),
        },
        retrieved_chunks=retrieved,
        llm_prompts={
            "system_instruction": result.get("system_prompt", "n/a"),
            "prompt_payload": result.get("user_prompt", "n/a"),
        },
        generated_response={
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
        },
        execution_metadata=execution_metadata,
    )


def _schedule_audit_log(background_tasks: BackgroundTasks, event: AuditLogEvent) -> None:
    """Persist an audit event off the request path (Arq when enabled)."""
    audit_logger = get_audit_logger()
    if USE_ARQ_QUEUE:
        # Off-load to the Redis-backed Arq worker pool (see app/api/worker.py).
        background_tasks.add_task(_enqueue_audit_via_arq, event.model_dump(mode="json"))
    else:
        # Default: in-process FastAPI background task (zero extra infra).
        background_tasks.add_task(audit_logger.log_event, event)


# Distinct from the guardrail's FAIL fallback (SAFE_FALLBACK): delivered when
# enforcement is ON but the evaluator ERRORED / TIMED OUT, so no verdict exists.
# This is the only way to avoid presenting unverified content as though it had
# been checked -- the SSE contract has no separate status channel.
GUARDRAIL_UNAVAILABLE_MESSAGE = (
    "I couldn't verify this answer against the source data, so I'm not "
    "returning an unverified response. Please try again."
)


def _validate_guardrail_config(pipeline: Any) -> None:
    """Fail fast on contradictory config: enforcement on, guardrail unavailable.

    A financial/rewrite request must be verifiable while ``GUARDRAIL_ENFORCE``
    is enabled; if the loaded pipeline has the guardrail disabled or absent its
    answers would be released unverified. Since the default advisory
    configuration (``GUARDRAIL_ENFORCE=false``) makes this a no-op, only an
    explicit contradictory configuration is rejected.

    Uses strict identity checks (``is not True`` / ``is None``) rather than
    generic truthiness so a ``MagicMock`` or any unexpected object cannot pass
    the guard accidentally.
    """
    if not GUARDRAIL_ENFORCE:
        return
    enabled = getattr(pipeline, "_enable_guardrail", None)
    guardrail = getattr(pipeline, "_guardrail", None)
    if enabled is not True or guardrail is None:
        raise RuntimeError(
            "Invalid configuration: GUARDRAIL_ENFORCE=true but the loaded "
            "pipeline has the guardrail disabled or unavailable "
            f"(enable_guardrail={enabled!r}, guardrail={type(guardrail).__name__}). "
            "Financial answers would be released unverified. Set "
            "GUARDRAIL_ENFORCE=false or construct the pipeline with "
            "enable_guardrail=True."
        )


def _guardrail_should_swap(status: Any, final_output: Any) -> bool:
    """Opt-in enforcement predicate shared by the sync and streaming paths.

    Only a *definite* FAIL (``status == "failed"``) that carries a safe fallback
    triggers a swap. ``error`` (evaluator crash/timeout) and ``not_run``
    (guardrail disabled, GENERAL route, security halt) never swap -- the
    evaluator failed, not the LLM -- and are never treated as a PASS. Advisory
    by default: with ``GUARDRAIL_ENFORCE`` disabled this always returns False.
    """
    return bool(GUARDRAIL_ENFORCE and status == "failed" and final_output)


def _guardrail_should_withhold(status: Any) -> bool:
    """True when enforcement must NOT release the generated answer.

    Only for an evaluator ERROR / timeout: verification ran but produced no
    verdict, so releasing the answer would present unverified content as though
    it had been checked. ``failed`` is handled separately by the safe-fallback
    swap. ``not_run`` is *not* withheld -- it means no verification was attempted
    (guardrail disabled, non-financial route, security halt), which is a
    deliberate, already-safe response rather than an evaluator fault.
    """
    return bool(GUARDRAIL_ENFORCE and status in ("error", "timeout"))


def _apply_guardrail_enforcement(
    result: dict[str, Any], answer: str, sources: list
) -> tuple[str, list, bool]:
    """Apply the guardrail delivery policy to the delivered (answer, sources).

    Advisory by default (``GUARDRAIL_ENFORCE`` off): the generated answer is
    returned unchanged; the verdict is only reported/persisted. When enforcing:
      * definite FAIL            -> safe fallback, sources cleared;
      * evaluator ERROR / timeout -> withhold the unverified answer and deliver a
        distinct "verification unavailable" notice, sources cleared;
      * PASS / not_run           -> deliver the generated answer.
    Returns ``(answer, sources, enforced)``; ``enforced`` is True only when
    enforcement changed what is delivered (FAIL swap or error withhold).
    """
    status = result.get("guardrail_status")
    final_output = result.get("guardrail_final_output")
    if _guardrail_should_swap(status, final_output):
        return final_output, [], True
    if _guardrail_should_withhold(status):
        return GUARDRAIL_UNAVAILABLE_MESSAGE, [], True
    return answer, sources, False


def _sanitize_session_memory(pipe, delivered_answer: str) -> None:
    """Redact the last assistant turn when enforcement replaced the answer.

    Without this, a FAIL-swapped or ERROR-withheld response would still leave the
    *unverified* generated answer in the session's conversation memory, where a
    later turn could treat it as trusted history.
    """
    memory = getattr(getattr(pipe, "_generator", None), "memory", None)
    if memory is None:
        return
    try:
        memory.replace_last_ai(delivered_answer)
    except Exception:
        logger.warning("Session memory sanitize failed (non-blocking)", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/chat", response_model=ChatQueryResponse)
@limiter.limit("10/minute")
async def chat(
    payload: ChatQueryRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    _user: str | None = Depends(get_current_user),
):
    """Main RAG query endpoint (rate-limited: 10 req/min per user/IP)."""
    t0 = time.perf_counter()
    pipe = _get_pipeline(request)
    _restore_session_memory(pipe, payload.session_id)
    request_id = getattr(request.state, "request_id", "unknown")
    user_identifier = (
        getattr(request.state, "user", None)
        or (request.client.host if request.client else "guest")
    )

    logger.info(
        "chat_request request_id=%s query=%.80s ticker=%s year=%s session=%s user=%s",
        request_id, payload.user_query, payload.ticker, payload.fiscal_year,
        payload.session_id, user_identifier,
    )

    result = pipe.query(
        user_query=payload.user_query,
        ticker=payload.ticker,
        fiscal_year=payload.fiscal_year,
        session_id=payload.session_id,
        tickers=payload.tickers,
    )

    elapsed = _elapsed_ms(t0)
    answer = result.get("raw_output", "")
    parsed = result.get("parsed")
    if parsed is not None and hasattr(parsed, "answer"):
        answer = parsed.answer

    # Defensive output hygiene: no internal %%TABLE_N%%/[TABLE N] artifacts.
    answer = _strip_internal_markers(answer)

    sources = _build_sources(result)

    # D1 enforcement (opt-in via GUARDRAIL_ENFORCE): a definite FAIL replaces
    # the delivered answer with the guardrail's safe fallback; an evaluator
    # ERROR/timeout withholds the unverified answer behind a distinct notice.
    # Default (advisory) keeps the answer text while still reporting/persisting
    # the real verdict below.
    answer, sources, guardrail_enforced = _apply_guardrail_enforcement(
        result, answer, sources
    )
    result["guardrail_enforced"] = guardrail_enforced
    # Memory is written AFTER enforcement so that, when enforcement replaced or
    # withheld the answer, the unverified generated answer is never retained as
    # trusted conversational history (correct session isolation is unchanged).
    if guardrail_enforced:
        _sanitize_session_memory(pipe, answer)
    _persist_session_memory(pipe, payload.session_id)

    # --- Asynchronous audit logging (fire-and-forget) ---
    audit_event = _build_audit_event(
        request_id=request_id,
        user_identifier=user_identifier,
        payload=payload,
        result=result,
        answer=answer,
        sources=sources,
        elapsed_ms=elapsed,
    )
    _schedule_audit_log(background_tasks, audit_event)

    return ChatQueryResponse(
        answer=answer,
        sources=sources,
        execution_time_ms=round(elapsed, 2),
        guardrail_status=GuardrailStatus(
            passed=bool(result.get("guardrail_passed", False)),
            status=result.get("guardrail_status", "not_run"),
            verified_claims=result.get("guardrail_verified_claims", []),
            failed_claims=result.get("guardrail_failed_claims", []),
        ),
        model_used=result.get("model_used", "unknown"),
        cache_hit=result.get("cache_hit", False),
        request_id=request_id,
    )


@app.post("/api/v1/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(
    payload: ChatQueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    _user: str | None = Depends(get_current_user),
):
    """SSE streaming endpoint for real-time token generation.

    Rate-limited (10 req/min per user/IP). A dynamic audit event is persisted
    after the stream completes, capturing the full generated answer.
    """
    pipe = _get_pipeline(request)
    _restore_session_memory(pipe, payload.session_id)
    request_id = getattr(request.state, "request_id", "unknown")
    user_identifier = (
        getattr(request.state, "user", None)
        or (request.client.host if request.client else "guest")
    )
    collected: list[str] = []

    from sse_starlette.sse import EventSourceResponse

    # Per-request grounding context filled by query_stream() via context_sink.
    # Sources/citations are derived ONLY from this list so a parallel or prior
    # query's context (the shared pipeline slot) can never leak into this
    # response.
    request_contexts: list[dict] = []
    # Per-request guardrail verdict, filled by query_stream() via guardrail_sink
    # once the answer has finished streaming. Lets the audit persist the REAL
    # status (the verdict necessarily arrives after the tokens, by design).
    guardrail_verdict: dict = {}

    async def event_generator() -> AsyncIterator[dict]:
        """Stream the clean answer live while hiding internal JSON structure.

        Raw LLM tokens form a structured JSON document (internal_thought,
        extracted_raw_data, answer, sources). As tokens arrive we decode ONLY
        the top-level ``answer`` string value and forward it to the client
        immediately, so the user sees answer content streaming before
        generation completes. The full raw document is still accumulated for
        the final parse / structured sources / memory / audit steps.
        """
        import json as _json

        extractor = StreamingAnswerExtractor()
        full_text: list[str] = []

        # D2 (Phase 2): when enforcement is ON nothing generated may reach the
        # client before the verdict is known -- otherwise a FAIL could only be
        # corrected *after* a hallucinated answer had already been streamed.
        # With enforcement OFF (default) tokens stream live exactly as before.
        withhold_until_verdict = bool(GUARDRAIL_ENFORCE)
        buffered_tokens: list[str] = []

        # --- Phase 1: Stream tokens through live as they are generated --------
        try:
            async for token in pipe.query_stream(
                user_query=payload.user_query,
                ticker=payload.ticker,
                fiscal_year=payload.fiscal_year,
                session_id=payload.session_id,
                tickers=payload.tickers,
                context_sink=request_contexts,
                guardrail_sink=guardrail_verdict,
            ):
                if token:
                    full_text.append(token)
                frag = extractor.feed(token)
                if frag:
                    clean = _strip_internal_markers(frag)
                    if clean:
                        if withhold_until_verdict:
                            buffered_tokens.append(clean)
                        else:
                            yield {"event": "token", "data": clean}
        except asyncio.CancelledError:
            # Client disconnected mid-stream; persist what we have and stop.
            if full_text:
                collected.append(extractor.raw_text or "".join(full_text))
                _persist_session_memory(pipe, payload.session_id)
            raise
        except Exception as exc:
            # Generation failed; surface an error and end cleanly so the client
            # does not hang waiting for the final events.
            logger.exception("Streaming generation failed: %s", exc)
            yield {"event": "error", "data": _json.dumps({"detail": str(exc)})}

        # Release any trailing decoded answer text not yet emitted.
        tail = extractor.flush()
        if tail:
            clean_tail = _strip_internal_markers(tail)
            if clean_tail:
                if withhold_until_verdict:
                    buffered_tokens.append(clean_tail)
                else:
                    yield {"event": "token", "data": clean_tail}

        collected_text = extractor.raw_text or "".join(full_text)
        collected.append(collected_text)

        # --- Phase 2: Parse JSON and extract clean answer + sources -----------
        answer_text = collected_text  # fallback: emit raw text
        raw_source_strings: list[str] = []
        try:
            parsed = _json.loads(collected_text)
            if isinstance(parsed, dict) and parsed.get("answer"):
                answer_text = parsed["answer"]
                raw_source_strings = parsed.get("sources", [])
        except (ValueError, TypeError, KeyError):
            pass  # not JSON — emit raw text as answer

        # Defensive output hygiene: internal %%TABLE_N%%/[TABLE N] artifacts
        # echoed by the LLM are purged before anything is streamed to the UI.
        answer_text = _strip_internal_markers(answer_text)

        # --- Phase 2b: Build structured SourceChunk objects --------------------
        # The LLM emits citation strings (e.g. "NVIDIA - Data Center - 2026").
        # We enrich this response from the EXACT chunks retrieved/grounded for
        # THIS request (captured via context_sink) so the frontend source
        # inspector drawer shows the real chunks behind the answer. Chunks are
        # ordered LLM-cited first, then by retrieval score; unrelated chunks
        # from prior queries or other requests can never appear because the
        # source set is bounded by request_contexts.
        structured_sources: list[dict] = _build_stream_sources(
            raw_source_strings, request_contexts
        )

        # --- Phase 2c: Opt-in guardrail enforcement (D2) ----------------------
        # Same policy as the sync endpoint (shared predicates): under enforcement
        # ONLY a definite PASS releases the generated answer. A definite FAIL
        # swaps in the safe fallback; an evaluator ERROR/timeout withholds the
        # unverified answer behind a distinct notice (the SSE contract has no
        # status channel, so releasing it would look like a verified answer).
        # `not_run` (guardrail disabled / non-financial route) is delivered.
        guardrail_status = guardrail_verdict.get("status")
        guardrail_swapped = _guardrail_should_swap(
            guardrail_status, guardrail_verdict.get("final_output")
        )
        guardrail_withheld = _guardrail_should_withhold(guardrail_status)
        guardrail_enforced = guardrail_swapped or guardrail_withheld
        guardrail_verdict["enforced"] = guardrail_enforced
        if guardrail_swapped:
            answer_text = guardrail_verdict["final_output"]
            structured_sources = []
        elif guardrail_withheld:
            answer_text = GUARDRAIL_UNAVAILABLE_MESSAGE
            structured_sources = []
        elif withhold_until_verdict:
            # PASS / not_run: release the buffered generated answer now.
            for chunk in buffered_tokens:
                yield {"event": "token", "data": chunk}

        # Memory is written AFTER enforcement so an enforced/withheld answer does
        # not leave the unverified generated answer in conversation memory.
        if guardrail_enforced:
            _sanitize_session_memory(pipe, answer_text)
        _persist_session_memory(pipe, payload.session_id)

        # Audit must persist the DELIVERED answer (parsed answer, or the
        # enforced fallback) -- matching the sync path -- instead of the raw
        # LLM JSON document.
        collected[:] = [answer_text]

        # --- Phase 3: Fallback re-stream for non-JSON output -------------------
        # If we never decoded an answer field (the LLM produced plain text), emit
        # the sanitized text in small chunks so the user still gets a response.
        if not extractor.answer_found:
            words = answer_text.split()
            chunk_size = 3  # words per SSE frame
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size]).strip()
                if chunk:
                    yield {"event": "token", "data": chunk}
                await asyncio.sleep(0.015)

        # --- Phase 4: Final structured events ----------------------------------
        yield {"event": "answer", "data": answer_text}
        if structured_sources:
            yield {"event": "sources", "data": _json.dumps(structured_sources)}
        yield {"event": "done", "data": ""}

    async def _log_stream_audit() -> None:
        """Build + persist an audit trace once streaming finishes.

        ``retrieved_chunks`` is filled from the per-request grounding context
        captured for this exact query (not the shared pipeline slot, which a
        subsequent request may rewrite) so the analytics context-quality
        buckets reflect streaming traffic. Injected augmentation tables carry
        score 0.0 and are excluded so only real retrieval relevance scores are
        bucketed.
        """
        scored_chunks = [
            {
                "chunk_id": ctx.get("chunk_id", ""),
                "score": _normalize_audit_score(ctx.get("score")),
                "ticker": (ctx.get("payload") or {}).get("ticker"),
                "fiscal_year": (ctx.get("payload") or {}).get("fiscal_year"),
                "section": (ctx.get("payload") or {}).get("section"),
            }
            for ctx in request_contexts
            if float(ctx.get("score") or 0.0) > 0
        ]
        audit_event = AuditLogEvent(
            request_id=request_id,
            user_identifier=user_identifier,
            user_query={"raw": payload.user_query, "standalone": payload.user_query},
            retrieved_chunks=scored_chunks,
            llm_prompts={
                "system_instruction": "n/a (streaming)",
                "prompt_payload": "n/a (streaming)",
            },
            generated_response={"answer": "".join(collected), "sources": []},
            execution_metadata={
                "mode": "stream",
                "model_used": "unknown",
                # D1: real streaming verdict; "not_run" when the guardrail did
                # not execute. Never fabricate a PASS.
                "guardrail_status": guardrail_verdict.get("status", "not_run"),
                "guardrail_passed": bool(guardrail_verdict.get("passed", False)),
                "guardrail_enforced": bool(guardrail_verdict.get("enforced", False)),
            },
        )
        _schedule_audit_log(background_tasks, audit_event)

    # Runs after the SSE response has been fully streamed to the client.
    background_tasks.add_task(_log_stream_audit)

    return EventSourceResponse(event_generator())


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    admin: UserPrincipal = Depends(require_roles(["admin"])),
    source_id: str | None = None,
    doc_type: str | None = None,
):
    """Upload an SEC filing document and trigger the ingestion pipeline.

    Admin-only: server-side RBAC (``require_roles(["admin"])``) rejects
    anonymous callers (401) and authenticated non-admin users (403) before any
    file persistence, task creation, embedding, or indexing. Rate-limited
    (10 req/min per user/IP). An audit event recording the upload is persisted
    in the background.
    """
    ticker = request.query_params.get("ticker", "UNKNOWN")
    fiscal_year = request.query_params.get("fiscal_year", "UNKNOWN")
    user_identifier = (
        getattr(request.state, "user", None)
        or admin.user_id
        or (request.client.host if request.client else "guest")
    )

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in APIFileParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(APIFileParser.SUPPORTED_EXTENSIONS))}",
        )

    raw_bytes = await file.read()

    # Resolve stable logical-filing identity (explicit param, else accession
    # from the filing header) and persist without silent overwrites.
    resolved_source_id = _resolve_upload_identity(raw_bytes, source_id)
    save_path = _resolve_upload_save_path(
        ticker, fiscal_year, file.filename, resolved_source_id, doc_type
    )
    save_path.write_bytes(raw_bytes)

    # Run ingestion in background (FastAPI BackgroundTasks by default; arq
    # queue when USE_ARQ_QUEUE=true — see app/api/worker.py for the worker).
    task_id = str(uuid.uuid4())[:8]
    _INGESTION_TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "filename": file.filename,
    }
    request_id = getattr(request.state, "request_id", "unknown")
    if USE_ARQ_QUEUE:
        from app.api.worker import enqueue_ingestion

        await enqueue_ingestion(
            save_path, ticker, fiscal_year, task_id, resolved_source_id, doc_type
        )
    else:
        background_tasks.add_task(
            _ingest_document,
            save_path,
            ticker,
            fiscal_year,
            task_id,
            resolved_source_id,
            doc_type,
        )

    # --- Audit log the upload (fire-and-forget) ---
    upload_audit = AuditLogEvent(
        request_id=request_id,
        user_identifier=user_identifier,
        user_query={"raw": f"UPLOAD {file.filename}", "standalone": f"UPLOAD {file.filename}"},
        retrieved_chunks=[],
        llm_prompts={"system_instruction": "n/a (upload)", "prompt_payload": "n/a (upload)"},
        generated_response={
            "answer": f"Queued ingestion of {file.filename}",
            "sources": [{"ticker": ticker, "fiscal_year": fiscal_year, "task_id": task_id}],
        },
        execution_metadata={
            "mode": "upload",
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "task_id": task_id,
            "queue": "arq" if USE_ARQ_QUEUE else "fastapi-background",
        },
    )
    _schedule_audit_log(background_tasks, upload_audit)

    logger.info(
        "Document upload queued: request_id=%s file=%s ticker=%s year=%s task=%s queue=%s",
        request_id, file.filename, ticker, fiscal_year, task_id,
        "arq" if USE_ARQ_QUEUE else "fastapi-background",
    )

    return DocumentUploadResponse(
        filename=file.filename,
        ticker=ticker,
        fiscal_year=fiscal_year,
        task_id=task_id,
    )


def _ingest_document(
    file_path: Path,
    ticker: str,
    fiscal_year: str,
    task_id: str,
    source_id: str | None = None,
    doc_type: str | None = None,
):
    """Run the full ingestion pipeline on an uploaded document (background)."""
    from cleaning import clean_financial_text
    from metadata_extractor import extract_metadata
    from hybrid_chunker import chunk_document
    from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer

    logger.info("Ingestion task %s started: %s", task_id, file_path.name)
    if task_id in _INGESTION_TASKS:
        _INGESTION_TASKS[task_id]["status"] = "processing"

    try:
        file_bytes = file_path.read_bytes()
        filename = file_path.name
        parser = APIFileParser()
        parsed = parser.parse_file(file_bytes, filename)

        cleaned = clean_financial_text(parsed["text_content"])
        base_metadata = extract_metadata(
            file_path=file_path,
            content=parsed["raw_html"],
            chunk_text=cleaned[:2000],
            chunk_index=0,
            source_id=source_id,
        )
        if doc_type:
            base_metadata["doc_type"] = doc_type
        year_match = re.search(r"(\d{4})", fiscal_year)
        if year_match:
            base_metadata["fiscal_year"] = year_match.group(1)
        chunks = chunk_document(
            text=cleaned,
            tables=parsed["tables"],
            file_path=file_path,
            metadata_base=base_metadata,
        )

        if not chunks:
            logger.warning("Ingestion task %s produced zero chunks", task_id)
            if task_id in _INGESTION_TASKS:
                _INGESTION_TASKS[task_id].update(status="completed", chunks_created=0)
            return {"task_id": task_id, "chunks_created": 0, "mongo_count": 0, "qdrant_count": 0}

        embedder = EmbeddingEngine()
        mongo = MongoDBIndexer()

        all_embeddings = embedder.embed([c["text"] for c in chunks], batch_size=8)

        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None and pipeline._qdrant_indexer is not None:
            qdrant = pipeline._qdrant_indexer
            owns_qdrant = False
        else:
            qdrant = QdrantIndexer()
            owns_qdrant = True

        resolved_sid = source_id or base_metadata.get("source_id")
        if resolved_sid:
            # D6-B: identity is known — replace this filing's chunks safely
            # (verify writes in both stores, then delete stale within scope).
            from safe_replacement import replace_source

            for chunk in chunks:
                chunk.setdefault("metadata", {})["source_id"] = resolved_sid
            summary = replace_source(resolved_sid, chunks, all_embeddings, mongo, qdrant)
            mongo_count = summary["mongo_present"]
            qdrant_count = summary["qdrant_present"]
        else:
            # Identity unresolved — upsert only, never delete.
            qdrant_count = qdrant.upsert_vectors(chunks, all_embeddings)
            mongo_count = mongo.upsert_chunks(chunks)

        if owns_qdrant:
            qdrant.close()
        mongo.close()

        logger.info(
            "Ingestion task %s done: %d chunks, %d mongo, %d qdrant",
            task_id, len(chunks), mongo_count, qdrant_count,
        )
        if task_id in _INGESTION_TASKS:
            _INGESTION_TASKS[task_id].update(status="completed", chunks_created=len(chunks))
        return {
            "task_id": task_id,
            "chunks_created": len(chunks),
            "mongo_count": mongo_count,
            "qdrant_count": qdrant_count,
        }
    except Exception as exc:
        logger.error("Ingestion task %s failed: %s", task_id, exc, exc_info=True)
        if task_id in _INGESTION_TASKS:
            _INGESTION_TASKS[task_id].update(status="failed", error=str(exc))
        return {"task_id": task_id, "chunks_created": 0, "mongo_count": 0, "qdrant_count": 0, "error": str(exc)}


@app.get("/api/v1/documents/tasks/{task_id}", response_model=IngestionTaskStatusResponse)
async def get_ingestion_task_status(
    task_id: str,
    _user: str | None = Depends(get_current_user),
):
    """Return the current status of a background ingestion task.

    Polled by the frontend Ingestion tab (every 1.5s) to transition the UI
    from queued/processing to completed/failed. Intentionally not rate-limited
    so polling is never throttled.
    """
    task = _INGESTION_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown ingestion task '{task_id}'")
    return IngestionTaskStatusResponse(**task)


@app.delete("/api/v1/cache", response_model=CacheFlushResponse)
async def flush_cache(
    request: Request,
    current_user: UserPrincipal = Depends(require_roles(["admin"])),
):
    """Admin-only: flush all cached RAG responses from Redis.

    Guarded by RBAC (require_roles(["admin"])) — previously anonymous callers
    could wipe the entire semantic cache. Rate limiting is intentionally not
    applied so the admin control plane is never throttled.
    """
    pipe = _get_pipeline(request)
    if pipe._cache is None:
        return CacheFlushResponse(status="cache_disabled", keys_removed=0)
    count = pipe._cache.flush_all()
    if count < 0:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return CacheFlushResponse(status="ok", keys_removed=count)


@app.get("/health", response_model=HealthCheckResponse)
async def health(request: Request):
    """Health check endpoint — probes MongoDB, Qdrant, and Redis.

    Also reports MongoDB↔Qdrant count parity (D6). The parity check is
    *informational only*: a drift is surfaced as a warning and never flips the
    liveness status, so a known corpus drift cannot make the service look down.
    """
    services = HealthServiceStatus()

    mongo_chunks: int | None = None
    qdrant_points: int | None = None

    # MongoDB check (+ raw_chunks count for parity)
    try:
        import pymongo
        from config.settings import MONGODB_URI, MONGODB_DB
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        try:
            mongo_chunks = client[MONGODB_DB]["raw_chunks"].count_documents({})
        except Exception as exc:
            logger.warning("Health: mongo raw_chunks count failed: %s", exc)
        client.close()
        services.mongodb = "ok"
    except Exception as exc:
        services.mongodb = f"unavailable ({exc})"

    # Qdrant check (+ point count for parity)
    try:
        pipe = _get_pipeline(request)
        qdrant_points = pipe._qdrant_indexer.count_points()
        services.qdrant = "ok"
    except Exception as exc:
        services.qdrant = f"unavailable ({exc})"

    # D6: MongoDB↔Qdrant count parity (informational; never affects liveness).
    if isinstance(mongo_chunks, int) and isinstance(qdrant_points, int):
        delta = mongo_chunks - qdrant_points
        if delta == 0:
            services.mongo_qdrant_parity = f"ok ({mongo_chunks})"
        else:
            services.mongo_qdrant_parity = (
                f"drift (mongo={mongo_chunks}, qdrant={qdrant_points}, delta={delta})"
            )
            logger.warning(
                "Health: MongoDB(%d) / Qdrant(%d) count drift (delta=%d)",
                mongo_chunks,
                qdrant_points,
                delta,
            )
    else:
        services.mongo_qdrant_parity = "unknown"

    # Redis check
    try:
        import redis
        from config.settings import REDIS_URL
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        r.close()
        services.redis = "ok"
    except Exception:
        services.redis = "unavailable (not critical)"

    overall = "ok" if services.mongodb == "ok" else "degraded"
    warmup = getattr(request.app.state, "warmup_completed", False)
    return HealthCheckResponse(status=overall, services=services, warmup_completed=warmup)


@app.get("/api/v1/analytics/summary", response_model=AnalyticsSummaryResponse)
@limiter.limit("30/minute")
async def get_analytics_summary(
    request: Request,
    response: Response,
    days: int = 30,
    _user: str | None = Depends(get_current_user),
):
    """Aggregated system analytics — latency, evaluation scores, context quality.

    Reads from MongoDB ``rag_audit_logs`` (last *days* days) and the persisted
    ``artifacts/evaluation_scores.json`` file produced by the Module 6
    evaluation pipeline.
    """
    from app.api.analytics import get_analytics_summary as _aggregate

    try:
        data = await _aggregate(days=days)
        return AnalyticsSummaryResponse(**data)
    except Exception as exc:
        logger.error("Analytics aggregation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Analytics data unavailable")


@app.get("/api/v1/audit/logs")
@limiter.limit("30/minute")
async def get_audit_logs(
    request: Request,
    response: Response,
    limit: int = 50,
    _user: str | None = Depends(get_current_user),
):
    """Return recent RAG audit logs from MongoDB (most-recent first).

    Rate-limited (30 req/min per user/IP). Useful for ops/debugging the
    dynamic ``rag_audit_logs`` collection.
    """
    logger_ = get_audit_logger()
    try:
        logs = await logger_.get_recent_logs(limit=limit)
        return {"count": len(logs), "logs": logs}
    except Exception as exc:
        logger.error("Audit log retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Audit log store unavailable")


# ---------------------------------------------------------------------------
# Admin-only privileged operations (RBAC: require_roles(["admin"]))
# ---------------------------------------------------------------------------
# NOTE: The read endpoints /api/v1/audit/logs and /api/v1/documents/upload remain
# publicly reachable (rate-limited) for backward compatibility with the existing
# audited API suite. Privileged *mutations* (clear, ingest, role management) are
# strictly admin-gated below and tested for 403 (user) / 2xx (admin).


@app.delete("/api/v1/db/clear")
@limiter.limit("5/minute")
async def db_clear(
    request: Request,
    response: Response,
    _admin: UserPrincipal = Depends(require_roles(["admin"])),
):
    """Admin-only: drop MongoDB + Qdrant collections (best-effort, idempotent).

    Used by operators to reset the knowledge base. Always returns 200 — partial
    failures are logged, never surfaced as a 5xx, so the control plane stays up.
    """
    pipe = getattr(request.app.state, "pipeline", None)
    cleared: list[str] = []
    try:
        if pipe is not None:
            try:
                pipe._mongo_indexer.drop_collection()
                cleared.append("mongo")
            except Exception as exc:  # pragma: no cover
                logger.warning("db/clear mongo drop failed: %s", exc)
            try:
                pipe._qdrant_indexer.delete_collection()
                cleared.append("qdrant")
            except Exception as exc:  # pragma: no cover
                logger.warning("db/clear qdrant drop failed: %s", exc)
        else:
            cleared.append("pipeline_unavailable")
    except Exception as exc:  # pragma: no cover
        logger.error("db/clear unexpected error: %s", exc, exc_info=True)
        return {"status": "error", "detail": "Clear attempt failed; see server logs."}
    return {
        "status": "cleared" if cleared else "noop",
        "detail": f"Cleared: {', '.join(cleared)}" if cleared else "No action taken.",
    }


@app.post("/api/v1/ingest", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def admin_ingest(
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    _admin: UserPrincipal = Depends(require_roles(["admin"])),
    source_id: str | None = None,
    doc_type: str | None = None,
):
    """Admin-only document ingestion (RBAC-gated alias of /documents/upload).

    Functionally identical to the public upload endpoint, but restricted to the
    ``admin`` role so only operators can mutate the knowledge base.
    """
    ticker = request.query_params.get("ticker", "UNKNOWN")
    fiscal_year = request.query_params.get("fiscal_year", "UNKNOWN")
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in APIFileParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(APIFileParser.SUPPORTED_EXTENSIONS))}",
        )

    raw_bytes = await file.read()
    resolved_source_id = _resolve_upload_identity(raw_bytes, source_id)
    save_path = _resolve_upload_save_path(
        ticker, fiscal_year, file.filename, resolved_source_id, doc_type
    )
    save_path.write_bytes(raw_bytes)

    task_id = str(uuid.uuid4())[:8]
    request_id = getattr(request.state, "request_id", "unknown")
    if USE_ARQ_QUEUE:
        from app.api.worker import enqueue_ingestion

        await enqueue_ingestion(
            save_path, ticker, fiscal_year, task_id, resolved_source_id, doc_type
        )
    else:
        background_tasks.add_task(
            _ingest_document,
            save_path,
            ticker,
            fiscal_year,
            task_id,
            resolved_source_id,
            doc_type,
        )

    logger.info(
        "Admin ingest: admin=%s file=%s ticker=%s year=%s task=%s",
        _admin.user_id, file.filename, ticker, fiscal_year, task_id,
    )
    return DocumentUploadResponse(
        filename=file.filename,
        ticker=ticker,
        fiscal_year=fiscal_year,
        task_id=task_id,
    )
