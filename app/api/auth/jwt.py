"""
JWT token issuing / validation, password hashing, and the Redis-backed
revocation (logout) blacklist for the Financial RAG AuthN/AuthZ layer.

This module is intentionally dependency-light so it can be imported from the
rate limiter (to resolve the *real* authenticated "user_id" for per-identity
rate buckets) without pulling in the rest of the auth stack.

Design notes
------------
* **JWT (HS256)** carries "sub" (user_id), "role", "email", a unique
  "jti" (used for revocation), "type" ("access" | "refresh"), "iat" and
  "exp". The secret comes from "JWT_SECRET_KEY" (env) -- a dev fallback is
  provided but MUST be overridden in production.
* **Passwords** are hashed with "pwdlib" + "bcrypt" ("$2b$"). bcrypt is
  constant-time and salted; we never store plaintext.
* **Revocation** uses a Redis blacklist keyed by "jti" with a TTL equal to the
  token's remaining lifetime, so a logged-out token is rejected even before it
  naturally expires. The async Redis client is injectable (tests use a fake).
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt
import redis.asyncio as aioredis
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

from config.logging_config import get_logger
from config.settings import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    AUTH_BLACKLIST_PREFIX,
    JWT_ALGORITHM,
    JWT_ISSUER,
    JWT_SECRET_KEY,
    REDIS_URL,
    REFRESH_TOKEN_EXPIRE_DAYS,
)

logger = get_logger("api.auth.jwt")

# bcrypt hasher instance (single global) -- cheap and thread/async safe.
_password_hasher = PasswordHash([BcryptHasher()])

# Module-level injectable Redis client. "None" => lazily built from REDIS_URL.
# Tests inject a fake async Redis so no real network is touched.
_redis_client: Optional["aioredis.Redis"] = None


def set_redis_client(client: Any) -> None:
    """Inject a Redis client (used by the test-suite to avoid real I/O)."""
    global _redis_client
    _redis_client = client


def get_redis_client() -> "aioredis.Redis":
    """Return the active async Redis client, building it on first use."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """Return a bcrypt hash for "plain" (never store the raw password)."""
    return _password_hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verify of "plain" against "hashed"; never raises."""
    try:
        return _password_hasher.verify(plain, hashed)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------


def _create_token(
    token_type: str,
    user_id: str,
    email: Optional[str] = None,
    role: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Build a signed JWT of the given "token_type" with a unique "jti"."""
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: dict[str, Any] = {
        "sub": user_id,
        "type": token_type,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": JWT_ISSUER,
    }
    if email is not None:
        payload["email"] = email
    if role is not None:
        payload["role"] = role
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_access_token(user_id: str, email: Optional[str] = None, role: Optional[str] = None) -> str:
    """Issue a short-lived access token (default 30 minutes)."""
    return _create_token(
        "access",
        user_id,
        email=email,
        role=role,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    """Issue a long-lived refresh token (default 7 days)."""
    return _create_token(
        "refresh",
        user_id,
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: Optional[str] = None) -> dict:
    """Decode + validate a JWT.

    Raises "jwt.PyJWTError" (or subclass) on any failure: bad signature,
    expired, wrong issuer, or missing required claims. When "expected_type"
    is supplied, the "type" claim must match (e.g. reject an access token
    presented where a refresh token is expected).
    """
    payload = jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        issuer=JWT_ISSUER,
        options={"require": ["exp", "sub", "jti", "type"]},
    )
    if expected_type is not None and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError(
            f"Expected '{expected_type}' token, got '{payload.get('type')}'"
        )
    return payload


# ---------------------------------------------------------------------------
# Revocation blacklist (Redis)
# ---------------------------------------------------------------------------


async def blacklist_token(jti: str, exp_timestamp: float) -> None:
    """Add "jti" to the Redis blacklist until the token's natural expiry.

    The TTL equals the remaining lifetime so blacklist keys self-expire and
    never accumulate. A Redis failure is logged but never breaks logout.
    """
    try:
        client = get_redis_client()
        ttl = max(int(exp_timestamp - datetime.now(timezone.utc).timestamp()), 1)
        await client.set(f"{AUTH_BLACKLIST_PREFIX}{jti}", "1", ex=ttl)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Blacklist write failed (token remains usable): %s", exc)


async def is_token_blacklisted(jti: str) -> bool:
    """Return True if "jti" has been revoked (logged out)."""
    if not jti:
        return False
    try:
        client = get_redis_client()
        return await client.exists(f"{AUTH_BLACKLIST_PREFIX}{jti}") > 0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Blacklist read failed (treated as NOT revoked): %s", exc)
        return False


async def revoke_tokens(
    access_jti: str,
    access_exp: float,
    refresh_jti: Optional[str] = None,
    refresh_exp: Optional[float] = None,
) -> None:
    """Revoke the access token (and optionally the refresh token) at logout."""
    await blacklist_token(access_jti, access_exp)
    if refresh_jti is not None:
        await blacklist_token(refresh_jti, refresh_exp or access_exp)
