"""Verify Fix 1 (env-var mapping) + Fix 2 (admin RBAC on DELETE /api/v1/cache).

Run from repo root:
    python scripts/_env_cachefix_verify.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

failures: list[str] = []


def _check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    if not ok:
        failures.append(name)


def _safe(uri: str) -> str:
    if "@" in uri:
        return uri.split("@")[-1]
    return uri.split("://")[-1]


# ---------------------------------------------------------------------------
# Fix 1: config.settings must honour legacy .env keys before local defaults
# ---------------------------------------------------------------------------
import config.settings as S  # noqa: E402

env_mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
env_mongo_db = os.getenv("MONGO_DB_NAME") or os.getenv("MONGODB_DB")
env_redis_url = os.getenv("REDIS_URL")
if not env_redis_url:
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    env_redis_url = f"redis://{host}:{port}/0"

if env_mongo_uri:
    _check(
        "settings.MONGODB_URI honours env",
        S.MONGODB_URI == env_mongo_uri,
        f"expected '{_safe(env_mongo_uri)}' got '{_safe(S.MONGODB_URI)}'",
    )
else:
    _check(
        "settings.MONGODB_URI default",
        S.MONGODB_URI == "mongodb://localhost:27017/financial_rag",
        f"got '{_safe(S.MONGODB_URI)}'",
    )

if env_mongo_db:
    _check(
        "settings.MONGODB_DB honours env",
        S.MONGODB_DB == env_mongo_db,
        f"expected '{env_mongo_db}' got '{S.MONGODB_DB}'",
    )
else:
    _check("settings.MONGODB_DB default", S.MONGODB_DB == "financial_rag", S.MONGODB_DB)

if env_redis_url:
    _check(
        "settings.REDIS_URL honours env",
        S.REDIS_URL == env_redis_url,
        f"expected '{_safe(env_redis_url)}' got '{_safe(S.REDIS_URL)}'",
    )
else:
    _check("settings.REDIS_URL default", S.REDIS_URL == "redis://localhost:6379/0", S.REDIS_URL)

# ---------------------------------------------------------------------------
# Fix 2: DELETE /api/v1/cache is admin-only (401 anonymous / 403 user / 200 admin)
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

from app.api.auth.jwt import create_access_token  # noqa: E402
from app.api.main import app  # noqa: E402


class _StubCache:
    def flush_all(self) -> int:
        return 7


class _StubPipe:
    def __init__(self, cache):
        self._cache = cache


# Note: TestClient used WITHOUT a context manager so the heavy lifespan
# (pipeline init + warm-up) never runs; we stub app.state.pipeline instead.
app.state.pipeline = _StubPipe(_StubCache())
client = TestClient(app)


def _unauth():
    return client.delete("/api/v1/cache")


def _as(role: str | None):
    headers = {}
    if role is not None:
        headers["Authorization"] = "Bearer " + create_access_token("u1", email="u@x.com", role=role)
    return client.delete("/api/v1/cache", headers=headers)


r = _unauth()
_check("cache flush anonymous -> 401", r.status_code == 401, f"got {r.status_code}")

r = _as("user")
_check("cache flush user -> 403", r.status_code == 403, f"got {r.status_code}")

r = _as("admin")
ok_body = r.status_code == 200 and r.json().get("status") == "ok" and r.json().get("keys_removed") == 7
_check("cache flush admin -> 200 ok", ok_body, f"got {r.status_code} {r.text[:120]!r}")

app.state.pipeline = _StubPipe(None)
r = _as("admin")
ok_disabled = r.status_code == 200 and r.json().get("status") == "cache_disabled"
_check("cache flush admin w/o cache -> cache_disabled", ok_disabled, f"got {r.status_code} {r.text[:120]!r}")

print()
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    sys.exit(1)
print("RESULT: PASS (env-mapping + cache RBAC both verified)")