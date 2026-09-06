"""
Pydantic schemas for FastAPI request/response payloads.

Includes schemas for the Advanced API features:
  * ``RateLimitErrorResponse`` -- structured HTTP 429 payload.
  * ``AuditLogEvent`` (+ helpers) -- dynamic MongoDB RAG execution trace.
"""

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field
from fastapi import UploadFile


class ChatQueryRequest(BaseModel):
    user_query: str = Field(
        ...,
        min_length=3,
        description="Natural language financial question (min 3 chars).",
        examples=["What are Apple's reportable business segments?"],
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Optional ticker filter (e.g. AAPL, MSFT, NVDA).",
    )
    tickers: Optional[list[str]] = Field(
        default=None,
        description="Optional multi-ticker filter for cross-company comparison (e.g. ['AAPL','MSFT']).",
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description="Optional fiscal year filter (e.g. 2025).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversation memory.",
    )


class SourceCitation(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0–1).")
    ticker: str = Field(default="UNKNOWN")
    fiscal_year: str = Field(default="UNKNOWN")
    section: str = Field(default="General")
    text_snippet: str = Field(default="", max_length=300)


class GuardrailStatus(BaseModel):
    passed: bool = Field(default=True)
    verified_claims: list[str] = Field(default_factory=list)
    failed_claims: list[str] = Field(default_factory=list)
    detail: str = Field(default="")


class ChatQueryResponse(BaseModel):
    answer: str = Field(..., description="Generated financial answer.")
    sources: list[SourceCitation] = Field(
        default_factory=list, description="Source chunks used."
    )
    execution_time_ms: float = Field(
        ..., description="End-to-end execution time in milliseconds."
    )
    guardrail_status: GuardrailStatus = Field(
        default_factory=GuardrailStatus, description="Guardrail verification result."
    )
    model_used: str = Field(
        default="unknown", description="LLM model used for generation."
    )
    cache_hit: bool = Field(default=False)
    request_id: Optional[str] = Field(
        default=None,
        description="Trace identifier (matches X-Request-ID response header) for correlating logs.",
    )
    task_id: Optional[str] = Field(
        default=None, description="Background task identifier if applicable."
    )


class HealthServiceStatus(BaseModel):
    mongodb: str = Field(default="unknown")
    qdrant: str = Field(default="unknown")
    redis: str = Field(default="unknown")


class HealthCheckResponse(BaseModel):
    status: str = Field(default="ok")
    services: HealthServiceStatus = Field(default_factory=HealthServiceStatus)
    warmup_completed: bool = Field(default=False, description="Whether the startup warm-up routine finished successfully.")


class DocumentUploadResponse(BaseModel):
    filename: str
    ticker: str
    fiscal_year: str
    chunks_created: int = Field(default=0)
    mongo_count: int = Field(default=0)
    qdrant_count: int = Field(default=0)
    elapsed_seconds: float = Field(default=0.0)
    task_id: Optional[str] = Field(default=None)


class IngestionTaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    ticker: str = Field(default="UNKNOWN")
    fiscal_year: str = Field(default="UNKNOWN")
    filename: str = Field(default="")
    chunks_created: Optional[int] = Field(default=None)
    error: Optional[str] = Field(default=None)


class CacheFlushResponse(BaseModel):
    status: str = Field(default="ok")
    keys_removed: int = Field(default=0, description="Number of cache keys deleted.")


class PerformanceMetrics(BaseModel):
    total_queries: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    avg_ttft_ms: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0, description="Percentage 0-100.")
    guardrail_pass_rate: float = Field(default=0.0, description="Percentage 0-100.")
    p50_latency_ms: float = Field(default=0.0)
    p95_latency_ms: float = Field(default=0.0)
    p99_latency_ms: float = Field(default=0.0)


class EvalScores(BaseModel):
    faithfulness: float = Field(default=0.0)
    answer_relevance: float = Field(default=0.0)
    context_precision: float = Field(default=0.0)
    context_recall: float = Field(default=0.0)
    sample_count: int = Field(default=0)
    judge_model: str = Field(default="unknown")


class ContextQuality(BaseModel):
    high: float = Field(default=0.0, description="Share of chunks with score > 0.8 (%).")
    mid: float = Field(default=0.0, description="Share of chunks with score 0.5-0.8 (%).")
    low: float = Field(default=0.0, description="Share of chunks with score < 0.5 (%).")
    total: int = Field(default=0)


class VolumeStats(BaseModel):
    total_chunks: int = Field(default=0)
    by_ticker: dict[str, int] = Field(default_factory=dict)


class AnalyticsSummaryResponse(BaseModel):
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    eval_scores: EvalScores = Field(default_factory=EvalScores)
    context_quality: ContextQuality = Field(default_factory=ContextQuality)
    volume: VolumeStats = Field(default_factory=VolumeStats)


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description.")
    error_code: str = Field(default="INTERNAL_ERROR")


class RateLimitErrorResponse(BaseModel):
    """Structured payload returned when the rate limit is exceeded (HTTP 429)."""

    detail: str = Field(
        default="Rate limit exceeded. Please retry after the indicated period.",
        description="Human-readable explanation of the throttling.",
    )
    error_code: str = Field(
        default="RATE_LIMIT_EXCEEDED",
        description="Stable machine-readable error code for clients.",
    )
    limit: str = Field(
        default="10/minute",
        description="The rate limit rule that was violated.",
    )
    retry_after_seconds: int = Field(
        default=60,
        description="Seconds the client should wait before retrying (mirrors Retry-After).",
    )


# ---------------------------------------------------------------------------
# Dynamic MongoDB audit-log schema (rag_audit_logs)
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk recorded in the audit trace."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(default="", description="Chunk identifier / point id.")
    text: str = Field(default="", description="Chunk text snippet (may be truncated).")
    score: float = Field(default=0.0, description="Similarity / rerank score (0-1).")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Source metadata (filing, section, page...)."
    )
    ticker_tag: Optional[str] = Field(
        default=None, description="Ticker tag associated with the chunk."
    )


class AuditLogEvent(BaseModel):
    """Unstructured / dynamic execution trace persisted per RAG request.

    Every field beyond the identity columns is intentionally flexible so the
    logger can evolve without schema migrations. ``extra='allow'`` permits
    arbitrary additional keys.
    """

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(
        ..., description="UUID trace correlating logs, headers and responses."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC event time.",
    )
    user_identifier: str = Field(
        ..., description="User ID if authenticated, otherwise client IP."
    )
    user_query: dict[str, Any] = Field(
        default_factory=dict,
        description="Original raw query and standalone expanded query.",
    )
    retrieved_chunks: list[Any] = Field(
        default_factory=list,
        description="Retrieved doc chunks: ids, scores, metadata, ticker tags.",
    )
    llm_prompts: dict[str, Any] = Field(
        default_factory=dict,
        description="System instruction and prompt payload sent to the LLM.",
    )
    generated_response: dict[str, Any] = Field(
        default_factory=dict, description="Final answer text and sources array."
    )
    execution_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Latencies, token usage, model cost, guardrail, errors.",
    )


# ---------------------------------------------------------------------------
# Authentication & Authorization (Module 7: AuthN/AuthZ + RBAC)
# ---------------------------------------------------------------------------


class SignupRequest(BaseModel):
    """Registration payload -- a new account (default role ``user``)."""

    email: EmailStr = Field(..., description="Unique login email (stored lower-cased).")
    password: str = Field(
        ..., min_length=8, description="Plaintext password (min 8 chars); bcrypt-hashed at rest."
    )
    full_name: str = Field(..., min_length=1, description="User's display name.")


class LoginRequest(BaseModel):
    """Email + password credentials for login."""

    email: EmailStr = Field(..., description="Account email.")
    password: str = Field(..., description="Plaintext password.")


class RefreshRequest(BaseModel):
    """Optional body form of the refresh call (token may also come via cookie)."""

    refresh_token: Optional[str] = Field(
        default=None, description="Refresh token (omitted when sent as HttpOnly cookie)."
    )


class ChangePasswordRequest(BaseModel):
    """Authenticated password change -- requires the current password."""

    old_password: str = Field(..., description="Current password (verified before change).")
    new_password: str = Field(..., min_length=8, description="New password (min 8 chars).")


class RoleUpdateRequest(BaseModel):
    """Admin role/state mutation for another account."""

    role: Literal["user", "admin"] = Field(
        ..., description="Target role. Only admins may set 'admin'."
    )
    is_active: Optional[bool] = Field(
        default=None, description="When provided, enable/disable the account."
    )


class TokenResponse(BaseModel):
    """Access + refresh token pair returned by login / signup / refresh."""

    access_token: str = Field(..., description="Short-lived JWT (HS256, ~30 min).")
    refresh_token: str = Field(..., description="Long-lived JWT (HS256, ~7 days).")
    token_type: str = Field(default="bearer", description="Token scheme (always 'bearer').")
    expires_in: int = Field(
        default=1800, description="Access token lifetime in seconds (for client scheduling)."
    )


class UserProfile(BaseModel):
    """Public, safe-to-return user profile (no secrets)."""

    user_id: str = Field(..., description="Stable account id (JWT ``sub``).")
    email: str = Field(..., description="Account email.")
    full_name: str = Field(default="", description="Display name.")
    role: str = Field(default="user", description="RBAC role: 'user' | 'admin'.")
    is_active: bool = Field(default=True, description="Whether the account is enabled.")
    created_at: Optional[datetime] = Field(
        default=None, description="Account creation time (UTC)."
    )


class UserRoleUpdateResponse(BaseModel):
    """Result of an admin role/state mutation."""

    user_id: str = Field(..., description="Affected account id.")
    email: EmailStr = Field(..., description="Affected account email.")
    role: str = Field(..., description="New role.")
    is_active: bool = Field(..., description="New active state.")
