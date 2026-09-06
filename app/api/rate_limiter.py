"""
Dynamic, context-aware rate-limiting key resolver for the Financial RAG API.

slowapi identifies the *subject* of a rate limit through a ``key_func(request)
-> str`` callable. We implement a **context-aware** strategy so that a single
heavy authenticated user can never starve the shared budget of anonymous
guests:

  * **Authenticated users** -- a valid API token / bearer credential was
    supplied -- are rate-limited per ``user_id`` (``user:<user_id>``).
  * **Unauthenticated / guest users** -- fall back to their **client IP
    address** (``request.client.host``) so every visitor gets an independent
    10 requests / minute ceiling (``ip:<client_host>``).

The token -> user_id resolution is intentionally side-effect free and is
re-used by both the limiter key function and the FastAPI ``get_current_user``
dependency, so a credential resolved once is consistently honoured everywhere.
"""

import os
from typing import Optional

from fastapi import Depends, Request

from config.settings import REDIS_URL

# ---------------------------------------------------------------------------
# Token -> user_id resolution
# ---------------------------------------------------------------------------
# Tokens are configured via the ``RAG_API_TOKENS`` environment variable as a
# comma-separated list of ``token:user_id`` pairs, e.g.::
#
#     RAG_API_TOKENS="sk_live_abc:alice,sk_live_def:bob"
#
# A built-in demo token is provided so the system is functional out-of-the-box
# and the test-suite can exercise *user-scoped* limiting without external
# configuration. Production deployments should override it via the env var.
_DEFAULT_TOKEN_MAP: dict[str, str] = {
    "rag_demo_token_123": "demo_user",
}

# slowapi rate-limit string applied to the RAG endpoints (10 requests / minute).
RATE_LIMIT_RULE: str = "10/minute"


def _load_token_map() -> dict[str, str]:
    """Build the token -> user_id map from defaults + ``RAG_API_TOKENS`` env.

    Returns:
        Mapping of bearer/api-key token string to a stable user identifier.
    """
    token_map: dict[str, str] = dict(_DEFAULT_TOKEN_MAP)
    raw = os.getenv("RAG_API_TOKENS", "")
    if not raw:
        return token_map
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        token, user_id = pair.split(":", 1)
        token = token.strip()
        user_id = user_id.strip()
        if token and user_id:
            token_map[token] = user_id
    return token_map


def resolve_user_id(request: Request) -> Optional[str]:
    """Resolve a stable user identifier from the request credentials.

    Resolution order:
      1. ``request.state.user`` -- already resolved by the auth dependency.
      2. ``Authorization: Bearer <token>`` header.
      3. ``X-API-Key: <token>`` header.

    Returns:
        The mapped ``user_id`` for authenticated callers, or ``None`` for
        anonymous guests (rate-limited by IP instead).
    """
    existing = getattr(request.state, "user", None)
    if existing:
        return existing

    headers = request.headers
    auth = headers.get("authorization")
    token: Optional[str] = None
    if auth:
        # Accept both "Bearer <token>" and a bare "<token>".
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        else:
            token = auth
    else:
        token = headers.get("x-api-key")

    if not token:
        return None

    # 1) Real JWT: if a valid access/refresh token is presented, use its
    #    ``sub`` claim as the stable identity for per-user rate buckets.
    try:
        from app.api.auth.jwt import decode_token

        payload = decode_token(token.strip())  # lenient: any valid JWT
        sub = payload.get("sub")
        if sub:
            return sub
    except Exception:
        # Not a JWT (or expired/invalid) -> fall through to the legacy map.
        pass

    # 2) Legacy static token map (demo / test tokens such as
    #    ``rag_demo_token_123``) so pre-existing integrations keep working.
    return _load_token_map().get(token.strip())


def dynamic_rate_limit_key(request: Request) -> str:
    """slowapi ``key_func``: returns the rate-limit bucket key for a request.

    * Authenticated -> ``user:<user_id>``
    * Guest         -> ``ip:<client_host>``

    This is what makes the limiter *context-aware*: the ceiling is per identity
    rather than a single global bucket shared by everyone.
    """
    user_id = resolve_user_id(request)
    if user_id:
        return f"user:{user_id}"

    client_host = "unknown"
    if request.client is not None:
        client_host = request.client.host
    return f"ip:{client_host}"


# ---------------------------------------------------------------------------
# slowapi Limiter instance (Redis-backed storage)
# ---------------------------------------------------------------------------
# ``storage_uri`` points the limiter at Redis so the counter survives across all
# API worker processes (horizontal scaling safe). ``strategy="fixed-window"``
# gives a simple, predictable 60-second window. ``headers_enabled`` exposes
# ``X-RateLimit-*`` response headers for client-side throttling UIs.
from slowapi import Limiter  # noqa: E402

# Rate limiting is ON by default in production. It can be disabled via the
# ``RATE_LIMIT_ENABLED`` env var (set to "false"/"0"/"no") — this is used by the
# test-suite so baseline endpoint tests that issue many unauthenticated
# requests are not throttled.
_RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() not in (
    "false",
    "0",
    "no",
)

limiter = Limiter(
    key_func=dynamic_rate_limit_key,
    storage_uri=REDIS_URL,
    strategy="fixed-window",
    headers_enabled=True,
    swallow_errors=False,
    enabled=_RATE_LIMIT_ENABLED,
)


def get_current_user(request: Request) -> Optional[str]:
    """FastAPI dependency that resolves (and caches) the caller identity.

    Sets ``request.state.user`` so downstream code (audit logging, the rate
    limit key function) consistently reads the same resolved identity without
    re-parsing headers. Returns ``None`` for anonymous guests.
    """
    user_id = resolve_user_id(request)
    try:
        request.state.user = user_id
    except Exception:
        # Some test request objects are immutable; ignore gracefully.
        pass
    return user_id
