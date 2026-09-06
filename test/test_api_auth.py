"""
AuthN/AuthZ + RBAC tests for the Financial RAG API.

Covers the full authentication lifecycle (signup, login, refresh, logout +
Redis blacklist, /me, change-password) and RBAC enforcement (a plain "user"
receives 403 on admin-only endpoints while an "admin" is allowed).

The MongoDB user store and the Redis revocation blacklist are fully faked so
the suite runs offline (no real datastore). The pipeline + audit logger are
mocked (mirroring the advanced-feature tests) so the admin mutation endpoints
(/db/clear, /ingest, /admin/audit/logs) execute without side effects.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (
    str(_PROJECT_ROOT),
    str(_src_root),
    str(_src_root / "1_ingestion"),
    str(_src_root / "5_generation"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pymongo  # noqa: E402 (patched in fixtures)
import qdrant_client  # noqa: E402 (patched in fixtures)
import redis  # noqa: E402 (patched in fixtures)

from app.api.auth.jwt import set_redis_client  # noqa: E402
from app.api.auth import router as auth_router  # noqa: E402
from app.api.auth.service import UserService, get_user_service, VALID_ROLES  # noqa: E402
from app.api.main import app  # noqa: E402


def _seed_user(svc, email, password, full_name, role="user"):
    """Synchronously insert a user into the (faked) user store.

    Used by sync fixtures because ``UserService.create_user`` is a coroutine;
    we mirror its behaviour (bcrypt hash, normalised email, role) directly so
    the login handler finds the account when it runs.
    """
    import uuid
    from datetime import datetime, timezone

    from app.api.auth.jwt import hash_password

    coll = svc._collection
    user_id = uuid.uuid4().hex
    coll.users[user_id] = {
        "user_id": user_id,
        "email": email.lower().strip(),
        "full_name": full_name,
        "hashed_password": hash_password(password),
        "role": role if role in VALID_ROLES else "user",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    return coll.users[user_id]


# ===========================================================================
# Fakes (offline MongoDB collection + Redis blacklist)
# ===========================================================================


class FakeRedis:
    """Minimal async Redis stand-in for the JWT revocation blacklist."""

    def __init__(self):
        self._store = {}

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def exists(self, key):
        return 1 if key in self._store else 0

    async def get(self, key):
        return self._store.get(key)


class _FakeCursor:
    """Async-iterable cursor supporting .sort().limit() chaining."""

    def __init__(self, items):
        self._items = list(items)

    def sort(self, *a, **k):
        return self

    def limit(self, n):
        self._items = self._items[:n]
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class FakeUserCollection:
    """In-memory Mongo collection implementing the UserService contract."""

    def __init__(self):
        self.users = {}

    def _match(self, u, filt):
        return all(u.get(k) == v for k, v in filt.items())

    async def insert_one(self, doc):
        if any(u["email"] == doc["email"] for u in self.users.values()):
            from pymongo.errors import DuplicateKeyError

            raise DuplicateKeyError("duplicate email")
        self.users[doc["user_id"]] = dict(doc)
        return MagicMock(inserted_id=doc["user_id"])

    async def find_one(self, filt):
        for u in self.users.values():
            if self._match(u, filt):
                return u
        return None

    async def find_one_and_update(self, filt, update, return_document=None):
        for u in self.users.values():
            if self._match(u, filt):
                u.update(update.get("$set", {}))
                return u
        return None

    async def update_one(self, filt, update):
        for u in self.users.values():
            if self._match(u, filt):
                u.update(update.get("$set", {}))
                return MagicMock(modified_count=1)
        return MagicMock(modified_count=0)

    async def count_documents(self, filt):
        return sum(1 for u in self.users.values() if self._match(u, filt))

    def find(self, filt):
        matched = [u for u in self.users.values() if self._match(u, filt)]
        return _FakeCursor(matched)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def fake_redis():
    r = FakeRedis()
    set_redis_client(r)  # inject for the JWT blacklist
    return r


@pytest.fixture
def fake_users():
    return FakeUserCollection()


@pytest.fixture(autouse=True)
def patch_auth(fake_redis, fake_users):
    """Mock pipeline + audit logger and inject the fake user store / redis."""
    mock_pipe = MagicMock()
    mock_pipe.query.return_value = {"raw_output": "{}", "parsed": None}
    mock_pipe.query_stream = AsyncMock()
    mock_pipe.close = MagicMock()
    audit = MagicMock()
    audit.log_event = MagicMock()
    audit.get_recent_logs = AsyncMock(return_value=[])

    with (
        patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe),
        patch.object(pymongo, "MongoClient") as mock_mongo,
        patch.object(qdrant_client, "QdrantClient") as mock_qdrant,
        patch.object(redis, "from_url") as mock_redis,
        patch("app.api.main.get_audit_logger", return_value=audit),
        patch("app.api.auth.router.get_audit_logger", return_value=audit),
        patch("app.api.auth.router.get_user_service",
              return_value=UserService(collection=fake_users)),
    ):
        mock_mongo.return_value.admin.command.return_value = {"ok": 1}
        mock_qdrant.return_value.get_collections.return_value = MagicMock()
        mock_redis.return_value.ping.return_value = True
        yield {"pipeline": mock_pipe, "audit": audit}


@pytest.fixture
def client(patch_auth):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_tokens(client):
    """Create an admin account directly and return its login tokens."""
    svc = auth_router.get_user_service()
    _seed_user(svc, "admin@example.com", "AdminPass!1", "Admin", "admin")
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass!1"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
def user_tokens(client):
    """Register a normal user and return its login tokens."""
    svc = auth_router.get_user_service()
    _seed_user(svc, "user@example.com", "UserPass!1", "User", "user")
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "UserPass!1"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===========================================================================
# Authentication lifecycle
# ===========================================================================


class TestAuthLifecycle:

    def test_signup_returns_tokens(self, client):
        resp = client.post(
            "/api/v1/auth/signup",
            json={"email": "new@example.com", "password": "Secret!23", "full_name": "New"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"] and body["refresh_token"]

    def test_signup_duplicate_email_409(self, client):
        payload = {"email": "dup@example.com", "password": "Secret!23", "full_name": "D"}
        assert client.post("/api/v1/auth/signup", json=payload).status_code == 201
        assert client.post("/api/v1/auth/signup", json=payload).status_code == 409

    def test_login_invalid_credentials_401(self, client):
        client.post(
            "/api/v1/auth/signup",
            json={"email": "lc@example.com", "password": "Secret!23", "full_name": "L"},
        )
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "lc@example.com", "password": "wrongpass"},
        )
        assert resp.status_code == 401

    def test_login_valid_returns_tokens(self, client):
        client.post(
            "/api/v1/auth/signup",
            json={"email": "lv@example.com", "password": "Secret!23", "full_name": "L"},
        )
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "lv@example.com", "password": "Secret!23"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_me_requires_auth_401(self, client):
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_me_returns_profile(self, client, user_tokens):
        token = user_tokens["access_token"]
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "user@example.com"
        assert body["role"] == "user"

    def test_refresh_issues_new_access(self, client, user_tokens):
        refresh = user_tokens["refresh_token"]
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 200
        assert resp.json()["access_token"]

    def test_logout_blacklists_access_token(self, client, user_tokens):
        token = user_tokens["access_token"]
        assert client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 200
        logout = client.post(
            "/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"}
        )
        assert logout.status_code == 200
        after = client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert after.status_code == 401

    def test_change_password_wrong_old_401(self, client, user_tokens):
        token = user_tokens["access_token"]
        resp = client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"old_password": "notcorrect", "new_password": "BrandNew!9"},
        )
        assert resp.status_code == 401

    def test_change_password_success(self, client, user_tokens):
        token = user_tokens["access_token"]
        resp = client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"old_password": "UserPass!1", "new_password": "BrandNew!9"},
        )
        assert resp.status_code == 200
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com", "password": "BrandNew!9"},
        )
        assert login.status_code == 200


# ===========================================================================
# RBAC enforcement
# ===========================================================================


class TestRBAC:

    def _admin_id(self, admin_tokens, client):
        return client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
        ).json()["user_id"]

    def test_user_forbidden_on_admin_endpoints(self, client, user_tokens, admin_tokens):
        token = user_tokens["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        admin_id = self._admin_id(admin_tokens, client)

        assert client.patch(
            f"/api/v1/users/{admin_id}/role",
            headers=headers, json={"role": "admin"},
        ).status_code == 403
        assert client.get("/api/v1/admin/users", headers=headers).status_code == 403
        assert client.get("/api/v1/admin/audit/logs", headers=headers).status_code == 403
        assert client.delete("/api/v1/db/clear", headers=headers).status_code == 403
        files = {"file": ("f.txt", b"dummy", "text/plain")}
        assert client.post(
            "/api/v1/ingest", files=files,
            params={"ticker": "AAPL", "fiscal_year": "2025"}, headers=headers,
        ).status_code == 403

    def test_admin_allowed_on_admin_endpoints(self, client, admin_tokens, monkeypatch):
        token = admin_tokens["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        assert client.get("/api/v1/admin/users", headers=headers).status_code == 200
        assert client.get("/api/v1/admin/audit/logs", headers=headers).status_code == 200
        assert client.delete("/api/v1/db/clear", headers=headers).status_code == 200

        svc = auth_router.get_user_service()
        target = _seed_user(svc, "target@example.com", "Target!23", "T", "user")
        promo = client.patch(
            f"/api/v1/users/{target['user_id']}/role",
            headers=headers, json={"role": "admin"},
        )
        assert promo.status_code == 200
        assert promo.json()["role"] == "admin"

        monkeypatch.setattr(
            "app.api.main._ingest_document",
            lambda *a, **k: {"task_id": "x", "chunks_created": 0},
        )
        files = {"file": ("f.txt", b"dummy", "text/plain")}
        ingest = client.post(
            "/api/v1/ingest", files=files,
            params={"ticker": "AAPL", "fiscal_year": "2025"}, headers=headers,
        )
        assert ingest.status_code == 200


# ===========================================================================
# Super Admin bootstrap
# ===========================================================================


@pytest.mark.asyncio
async def test_superadmin_bootstrap_creates_admin(fake_users):
    """seed_super_admin creates the admin account when none exists."""
    from app.api.auth.router import seed_super_admin

    import app.api.auth.router as router_mod

    router_mod.SEED_SUPERADMIN = True
    from app.api.auth.service import get_user_service as _gsvc

    with patch(
        "app.api.auth.router.get_user_service",
        return_value=UserService(collection=fake_users),
    ):
        created = await seed_super_admin()
        assert created is not None
        assert created["role"] == "admin"
        assert await seed_super_admin() is None
    router_mod.SEED_SUPERADMIN = False

    svc = _gsvc(collection=fake_users)
    found = await svc.get_by_email("admin@financial-rag.local")
    assert found is not None
    assert found["role"] == "admin"
