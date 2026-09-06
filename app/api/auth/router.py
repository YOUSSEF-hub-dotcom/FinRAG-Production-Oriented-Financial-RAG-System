"""
HTTP routers for the Financial RAG AuthN/AuthZ layer.

"auth_router"  (/api/v1/auth)  -- full authentication lifecycle:
    POST /signup            register a new user (default role "user")
    POST /login             verify credentials -> access + refresh tokens
    POST /refresh           exchange a refresh token for a new access token
    POST /logout            revoke tokens (Redis blacklist) + clear cookie
    GET  /me                current user profile
    POST /change-password   authenticated password change

"admin_router" (/api/v1)    -- admin-only RBAC operations:
    PATCH /users/{user_id}/role   promote/demote role, enable/disable account
    GET   /admin/users            list all users
    GET   /admin/audit/logs       read the structured JSON audit log

All admin endpoints are guarded by "require_roles("admin")". The refresh token
is delivered both as an HttpOnly cookie and in the response body so non-browser
clients (and the test-suite) can use it directly.
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from config.logging_config import get_logger
from config.settings import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    SEED_SUPERADMIN,
    SUPERADMIN_EMAIL,
    SUPERADMIN_FULL_NAME,
    SUPERADMIN_PASSWORD,
)

from app.api.auth.dependencies import UserPrincipal, get_current_user, require_roles
from app.api.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_blacklisted,
    revoke_tokens,
    verify_password,
)
from app.api.auth.service import VALID_ROLES, UserService, get_user_service
from app.api.db_logger import get_audit_logger
from app.api.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    RoleUpdateRequest,
    SignupRequest,
    TokenResponse,
    UserProfile,
    UserRoleUpdateResponse,
)

logger = get_logger("api.auth.router")

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/v1", tags=["admin"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Attach the refresh token as a Secure / HttpOnly cookie."""
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=os.getenv("ENV", "development").lower() == "production",
        samesite="lax",
        max_age=int(60 * 60 * 24 * 7),  # 7 days
        path="/api/v1/auth/refresh",
    )


# ---------------------------------------------------------------------------
# Authentication lifecycle
# ---------------------------------------------------------------------------


@auth_router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(payload: SignupRequest, response: Response):
    """Register a new user (default role "user") and return a token pair.

    Auto-logs the user in by issuing access + refresh tokens.
    """
    svc = get_user_service()
    try:
        user = await svc.create_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            role="user",
        )
    except Exception:
        # DuplicateKeyError (or any integrity failure) -> 409 conflict.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )
    refresh = create_refresh_token(user["user_id"])
    _set_refresh_cookie(response, refresh)
    logger.info("Signup: new user created email=%s role=%s", user["email"], user["role"])
    return TokenResponse(
        access_token=create_access_token(
            user["user_id"], email=user["email"], role=user["role"]
        ),
        refresh_token=refresh,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, response: Response):
    """Verify email + password, then issue an access + refresh token pair."""
    svc = get_user_service()
    user = await svc.authenticate(payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )
    refresh = create_refresh_token(user["user_id"])
    _set_refresh_cookie(response, refresh)
    logger.info("Login: email=%s", user["email"])
    return TokenResponse(
        access_token=create_access_token(
            user["user_id"], email=user["email"], role=user["role"]
        ),
        refresh_token=refresh,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    payload: Optional[RefreshRequest] = None,
):
    """Exchange a valid refresh token for a fresh access token.

    The refresh token is read from the request body or the HttpOnly cookie.
    """
    refresh_token = None
    if payload is not None and payload.refresh_token:
        refresh_token = payload.refresh_token
    else:
        refresh_token = request.cookies.get("refresh_token")

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = decode_token(refresh_token, expected_type="refresh")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired refresh token: {exc}",
        )

    jti = claims.get("jti")
    if jti and await is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
        )

    user_id = claims["sub"]
    user = await get_user_service().get_by_id(user_id)
    role = user.get("role", "user") if user else "user"
    email = user.get("email") if user else None
    logger.info("Refresh: issued new access token for user=%s", user_id)
    return TokenResponse(
        access_token=create_access_token(user_id, email=email, role=role),
        refresh_token=refresh_token,  # unchanged; echoed for convenience
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@auth_router.post("/logout", status_code=200)
async def logout(
    request: Request,
    response: Response,
    principal: UserPrincipal = Depends(get_current_user),
):
    """Revoke the current access token (and refresh token if supplied).

    Blacklists the JWT "jti" values in Redis so the tokens are rejected
    everywhere, then clears the refresh cookie.
    """
    access_jti = principal.jti
    access_exp = float(principal.exp or 0)

    refresh_jti = None
    refresh_exp = None
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        try:
            rc = decode_token(refresh_token, expected_type="refresh")
            refresh_jti = rc.get("jti")
            refresh_exp = float(rc.get("exp", 0))
        except Exception:
            refresh_jti = None

    await revoke_tokens(access_jti, access_exp, refresh_jti, refresh_exp)
    response.delete_cookie("refresh_token", path="/api/v1/auth/refresh")
    logger.info("Logout: user=%s", principal.user_id)
    return {"detail": "Logged out successfully."}


@auth_router.get("/me", response_model=UserProfile)
async def me(principal: UserPrincipal = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    user = await get_user_service().get_by_id(principal.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserProfile(
        user_id=user["user_id"],
        email=user["email"],
        full_name=user.get("full_name", ""),
        role=user.get("role", "user"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at"),
    )


@auth_router.post("/change-password", status_code=200)
async def change_password(
    payload: ChangePasswordRequest,
    principal: UserPrincipal = Depends(get_current_user),
):
    """Change the authenticated user's password (requires the old password)."""
    svc = get_user_service()
    user = await svc.get_by_id(principal.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if not verify_password(payload.old_password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )
    changed = await svc.change_password(principal.user_id, payload.new_password)
    if not changed:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password update failed.",
        )
    logger.info("Change-password: user=%s", principal.user_id)
    return {"detail": "Password changed successfully."}


# ---------------------------------------------------------------------------
# Admin-only RBAC operations
# ---------------------------------------------------------------------------


@admin_router.patch("/users/{user_id}/role", response_model=UserRoleUpdateResponse)
async def update_user_role(
    user_id: str,
    payload: RoleUpdateRequest,
    _admin: UserPrincipal = Depends(require_roles(["admin"])),
):
    """Promote/demote a user's role or enable/disable their account (admin only)."""
    if payload.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"role must be one of {VALID_ROLES}.",
        )
    svc = get_user_service()
    updated = await svc.update_role(user_id, role=payload.role, is_active=payload.is_active)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target user not found.",
        )
    logger.info(
        "Admin role update: admin=%s target=%s -> role=%s active=%s",
        _admin.user_id, user_id, updated["role"], updated["is_active"],
    )
    return UserRoleUpdateResponse(
        user_id=updated["user_id"],
        email=updated["email"],
        role=updated["role"],
        is_active=updated["is_active"],
    )


@admin_router.get("/admin/users", response_model=list[UserProfile])
async def list_users(_admin: UserPrincipal = Depends(require_roles(["admin"]))):
    """List all users (admin only)."""
    users = await get_user_service().list_users()
    return [
        UserProfile(
            user_id=u["user_id"],
            email=u["email"],
            full_name=u.get("full_name", ""),
            role=u.get("role", "user"),
            is_active=u.get("is_active", True),
            created_at=u.get("created_at"),
        )
        for u in users
    ]


@admin_router.get("/admin/audit/logs")
async def admin_audit_logs(
    limit: int = 50,
    _admin: UserPrincipal = Depends(require_roles(["admin"])),
):
    """Read the structured JSON RAG audit log (admin only)."""
    logger_ = get_audit_logger()
    try:
        logs = await logger_.get_recent_logs(limit=limit)
        return {"count": len(logs), "logs": logs}
    except Exception as exc:
        logger.error("Admin audit log retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit log store unavailable",
        )


# ---------------------------------------------------------------------------
# Super Admin bootstrap (called from app lifespan)
# ---------------------------------------------------------------------------


async def seed_super_admin() -> Optional[dict]:
    """Create the Super Admin account on startup if no admin exists yet.

    Best-effort: any MongoDB/Redis failure is logged and swallowed so a missing
    datastore never crashes the API. Controlled by "SEED_SUPERADMIN" (off in
    the test-suite to avoid side effects on startup).
    """
    if not SEED_SUPERADMIN:
        return None
    svc = get_user_service()
    try:
        if await svc.count_admins() > 0:
            return None
        if await svc.get_by_email(SUPERADMIN_EMAIL) is not None:
            return None
        user = await svc.create_user(
            email=SUPERADMIN_EMAIL,
            password=SUPERADMIN_PASSWORD,
            full_name=SUPERADMIN_FULL_NAME,
            role="admin",
        )
        logger.info("Seeded Super Admin account: %s", SUPERADMIN_EMAIL)
        return user
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Super Admin bootstrap skipped: %s", exc)
        return None
    finally:
        try:
            await svc.close()
        except Exception:
            pass
