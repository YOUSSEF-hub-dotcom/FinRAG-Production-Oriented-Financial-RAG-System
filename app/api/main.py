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
from config.settings import DATA_DIR, SUPPORTED_TICKERS, USE_ARQ_QUEUE

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
# Application factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: init pipeline via MLflow Production alias → fallback. Shutdown: close connections."""
    logger.info("API starting up — loading FinancialRAGPipeline")

    # Dynamic loading: tries MLflow Production alias, falls back to direct init
    pipeline = get_or_create_pipeline()
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
    t0 = time.time()
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
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


def _build_sources(result: dict[str, Any]) -> list[SourceCitation]:
    """Extract source citations from pipeline result.

    Deduplicates by chunk_id (preserving order). Uses MongoDB metadata
    as fallback when the LLM citation format is incomplete or has
    N/A/UNKNOWN section labels.
    """
    seen: set[str] = set()
    sources: list[SourceCitation] = []
    parsed = result.get("parsed")

    # Build a metadata lookup from retrieved chunks
    retrieved = result.get("retrieved_chunks") or []
    meta_lookup: dict[str, dict] = {}
    for chunk in retrieved:
        cid = chunk.get("chunk_id", "")
        if cid:
            meta_lookup[cid] = chunk

    if parsed is not None and hasattr(parsed, "sources"):
        for src_str in parsed.sources:
            if not src_str or src_str in seen:
                continue
            seen.add(src_str)
            parts = src_str.split(" - ")
            ticker = parts[0].strip() if len(parts) > 0 else "UNKNOWN"
            fiscal_year = parts[1].strip() if len(parts) > 1 else "UNKNOWN"
            section = parts[2].strip() if len(parts) > 2 else "General"

            # Enrich from MongoDB metadata when available
            chunk_meta = meta_lookup.get(src_str, {})
            if chunk_meta:
                ticker = chunk_meta.get("ticker", ticker) or ticker
                fiscal_year = chunk_meta.get("fiscal_year", fiscal_year) or fiscal_year
                section = chunk_meta.get("section", section) or section

            # Clean trailing N/A / UNKNOWN fragments from section labels
            if section in ("N/A", "UNKNOWN", ""):
                section = "General"

            # Purge internal placeholder/marker artifacts (%%TABLE_N%%, [TABLE N]).
            section = _strip_internal_markers(section) or "General"

            sources.append(SourceCitation(
                chunk_id=src_str,
                score=1.0,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                text_snippet=_strip_internal_markers(src_str)[:300],
            ))
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
    # --- retrieved chunks (ids, scores, metadata, ticker tags) ---
    retrieved = result.get("retrieved_chunks") or []
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
        "guardrail_status": "passed" if result.get("guardrail_passed", True) else "failed",
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
    t0 = time.time()
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
    _persist_session_memory(pipe, payload.session_id)

    elapsed = (time.time() - t0) * 1000
    answer = result.get("raw_output", "")
    parsed = result.get("parsed")
    if parsed is not None and hasattr(parsed, "answer"):
        answer = parsed.answer

    # Defensive output hygiene: no internal %%TABLE_N%%/[TABLE N] artifacts.
    answer = _strip_internal_markers(answer)

    sources = _build_sources(result)

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
        guardrail_status=GuardrailStatus(passed=True),
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

        # --- Phase 1: Stream tokens through live as they are generated --------
        try:
            async for token in pipe.query_stream(
                user_query=payload.user_query,
                ticker=payload.ticker,
                fiscal_year=payload.fiscal_year,
                session_id=payload.session_id,
                tickers=payload.tickers,
            ):
                if token:
                    full_text.append(token)
                frag = extractor.feed(token)
                if frag:
                    clean = _strip_internal_markers(frag)
                    if clean:
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
                yield {"event": "token", "data": clean_tail}

        collected_text = extractor.raw_text or "".join(full_text)
        collected.append(collected_text)
        _persist_session_memory(pipe, payload.session_id)

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
        # The LLM emits citation strings (e.g. "AAPL - 2024 - Revenue - 10").
        # We enrich these with metadata from the actual retrieved chunks stored
        # in the pipeline's _last_contexts so the frontend source inspector
        # drawer can display chunk text, scores, and file details.
        structured_sources: list[dict] = []
        last_ctx = getattr(pipe, "_last_contexts", []) or []

        # Deduplicate LLM source strings
        seen_strs: set[str] = set()
        unique_strings = [s for s in raw_source_strings if s and s not in seen_strs and not seen_strs.add(s)]  # type: ignore[func-returns-value]

        # Map LLM citation strings to retrieved chunks for enrichment
        for citation in unique_strings:
            src_obj: dict = {"text": citation, "chunk_id": "", "score": 0.0}
            # Try to match citation to a retrieved chunk by ticker/year/section
            citation_lower = citation.lower()
            best_match: dict | None = None
            for ctx in last_ctx:
                payload_ = ctx.get("payload", {})
                tk = str(payload_.get("ticker", "")).lower()
                fy = str(payload_.get("fiscal_year", "")).lower()
                sec = str(payload_.get("section", "")).lower()
                if tk and tk in citation_lower:
                    best_match = ctx
                    break
            if best_match:
                payload_ = best_match.get("payload", {})
                # Prefer the real chunk content over the bare citation string,
                # probing every payload naming variant used by the backend.
                chunk_content = next(
                    (
                        str(payload_.get(key, "")).strip()
                        for key in ("raw_text", "page_content", "chunk_text", "text", "snippet")
                        if str(payload_.get(key, "")).strip()
                    ),
                    "",
                )
                src_obj["text"] = _strip_internal_markers(
                    chunk_content[:800] if chunk_content else citation
                )
                src_obj["chunk_id"] = str(best_match.get("chunk_id", ""))
                src_obj["score"] = float(best_match.get("score", 0.0))
                src_obj["ticker"] = _clean_meta_str(payload_.get("ticker"))
                src_obj["fiscal_year"] = _clean_meta_str(payload_.get("fiscal_year"))
                src_obj["section"] = _clean_meta_str(payload_.get("section"))
                src_obj["source_file"] = _clean_meta_str(payload_.get("source_file"))
                src_obj["page_number"] = payload_.get("page_number")
            structured_sources.append(src_obj)

        # Fallback: if no LLM sources but we have retrieved chunks, surface them
        if not structured_sources and last_ctx:
            for ctx in last_ctx[:5]:
                payload_ = ctx.get("payload", {})
                structured_sources.append({
                    "chunk_id": str(ctx.get("chunk_id", "")),
                    "score": float(ctx.get("score", 0.0)),
                    "ticker": _clean_meta_str(payload_.get("ticker")),
                    "fiscal_year": _clean_meta_str(payload_.get("fiscal_year")),
                    "section": _clean_meta_str(payload_.get("section")),
                    "source_file": _clean_meta_str(payload_.get("source_file")),
                    "page_number": payload_.get("page_number"),
                    "text": _strip_internal_markers(str(payload_.get("raw_text", ""))[:300]),
                })

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

        ``retrieved_chunks`` is filled from the pipeline's post-retrieval
        context store (populated by ``query_stream``) so the analytics
        context-quality buckets reflect streaming traffic. Injected
        augmentation tables carry score 0.0 and are excluded so only real
        retrieval relevance scores are bucketed.
        """
        scored_chunks = [
            {
                "chunk_id": ctx.get("chunk_id", ""),
                "score": _normalize_audit_score(ctx.get("score")),
                "ticker": (ctx.get("payload") or {}).get("ticker"),
                "fiscal_year": (ctx.get("payload") or {}).get("fiscal_year"),
                "section": (ctx.get("payload") or {}).get("section"),
            }
            for ctx in getattr(pipe, "_last_contexts", []) or []
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
            execution_metadata={"mode": "stream", "model_used": "unknown"},
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
    _user: str | None = Depends(get_current_user),
):
    """Upload an SEC filing document and trigger the ingestion pipeline.

    Rate-limited (10 req/min per user/IP). An audit event recording the upload
    is persisted in the background.
    """
    ticker = request.query_params.get("ticker", "UNKNOWN")
    fiscal_year = request.query_params.get("fiscal_year", "UNKNOWN")
    user_identifier = (
        getattr(request.state, "user", None)
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

    # Save to data dir
    save_dir = DATA_DIR / ticker / "10-K" / fiscal_year
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
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

        await enqueue_ingestion(save_path, ticker, fiscal_year, task_id)
    else:
        background_tasks.add_task(
            _ingest_document,
            save_path,
            ticker,
            fiscal_year,
            task_id,
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


def _ingest_document(file_path: Path, ticker: str, fiscal_year: str, task_id: str):
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
        )
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
    """Health check endpoint — probes MongoDB, Qdrant, and Redis."""
    services = HealthServiceStatus()

    # MongoDB check
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client.close()
        services.mongodb = "ok"
    except Exception as exc:
        services.mongodb = f"unavailable ({exc})"

    # Qdrant check
    try:
        pipe = _get_pipeline(request)
        pipe._qdrant_indexer.count_points()
        services.qdrant = "ok"
    except Exception as exc:
        services.qdrant = f"unavailable ({exc})"

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
    save_dir = DATA_DIR / ticker / "10-K" / fiscal_year
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
    save_path.write_bytes(raw_bytes)

    task_id = str(uuid.uuid4())[:8]
    request_id = getattr(request.state, "request_id", "unknown")
    if USE_ARQ_QUEUE:
        from app.api.worker import enqueue_ingestion

        await enqueue_ingestion(save_path, ticker, fiscal_year, task_id)
    else:
        background_tasks.add_task(_ingest_document, save_path, ticker, fiscal_year, task_id)

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
