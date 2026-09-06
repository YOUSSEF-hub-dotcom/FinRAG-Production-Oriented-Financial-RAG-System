"""
FastAPI security dependencies for the Financial RAG AuthN/AuthZ layer.

Two building blocks:

* "get_current_user" -- validates the "Authorization: Bearer <access>"
  header, checks the Redis revocation blacklist, and returns a
  :class:`UserPrincipal`. Raises 401 on missing/invalid/revoked tokens.
* "require_roles(roles)" -- a dependency *factory* that enforces RBAC: only
  principals whose "role" is in the allow-list pass; everyone else gets 403.

Both set "request.state.user" / "request.state.user_obj" so downstream code
(audit logging, the rate-limiter key function) reads a consistent identity.
"""

from typing import Iterable, Optional

from fastapi import Depends, HTTPException, Request, status

from app.api.auth.jwt import decode_token, is_token_blacklisted


class UserPrincipal:
    """Authenticated caller context propagated through FastAPI dependencies."""

    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: str = "user",
        full_name: Optional[str] = None,
        jti: Optional[str] = None,
        exp: Optional[int] = None,
    ) -> None:
        self.user_id = user_id
        self.email = email
        self.role = role
        self.full_name = full_name
        self.jti = jti
        self.exp = exp

    def public(self) -> dict:
        """Dict safe to expose in responses (no secrets)."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "full_name": self.full_name,
        }


async def get_current_user(request: Request) -> UserPrincipal:
    """Resolve + authenticate the caller from the bearer access token.

    Raises:
        HTTPException 401: missing header, malformed/expired token, or a
        token that has been revoked via the logout blacklist.
    """
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Split "Bearer <token>" (tolerate missing space gracefully).
    token = auth.split(None, 1)[1].strip() if " " in auth else ""

    try:
        payload = decode_token(token, expected_type="access")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jti = payload.get("jti")
    if jti and await is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked (logged out)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    principal = UserPrincipal(
        user_id=payload["sub"],
        email=payload.get("email"),
        role=payload.get("role", "user"),
        jti=jti,
        exp=payload.get("exp"),
    )
    # Expose identity to audit logging + the rate-limit key function.
    try:
        request.state.user = principal.user_id
        request.state.user_obj = principal
    except Exception:
        pass
    return principal


def require_roles(roles: Iterable[str]):
    """Dependency factory enforcing RBAC: allow only the given roles.

    Args:
        roles: an iterable of permitted roles, e.g. ``require_roles(["admin"])``.

    Usage::

        @app.get("/admin/secret", dependencies=[Depends(require_roles(["admin"]))])

    or as a parameter dependency to also receive the principal::

        async def view(principal: UserPrincipal = Depends(require_roles(["admin"]))):
            ...
    """
    allowed = set(roles)

    async def _checker(
        principal: UserPrincipal = Depends(get_current_user),
    ) -> UserPrincipal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges for this operation",
            )
        return principal

    return _checker


# Convenience alias used in some endpoint signatures / docs.
get_current_active_user = get_current_user
