"""
Tests for base-API hardening: request-id tracing, structured validation
errors, and the Arq (Redis-backed) background worker infrastructure.

Covers:
  - X-Request-ID response header + request_id echo in /chat payload
  - 422 validation errors returned as structured ErrorResponse
  - Arq WorkerSettings structure (functions, redis dsn, max_jobs, timeout)
  - enqueue_ingestion graceful degradation when Redis is unreachable
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymongo
import pytest
import qdrant_client
import redis
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

from app.api.main import app
from app.api.worker import WorkerSettings, enqueue_ingestion, ingest_document_job


@pytest.fixture(autouse=True)
def patch_pipeline():
    """Mock the pipeline so TestClient lifespans are cheap and offline."""
    mock_pipe = MagicMock()
    mock_pipe.query.return_value = {
        "raw_output": "{}",
        "parsed": MagicMock(answer="Mock answer.", sources=[]),
        "model_used": "mock",
        "cache_hit": False,
    }
    mock_pipe.query_stream = MagicMock()
    mock_pipe.close = MagicMock()

    with (
        patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe) as mock_cls,
        patch.object(pymongo, "MongoClient"),
        patch.object(qdrant_client, "QdrantClient"),
        patch.object(redis, "from_url"),
    ):
        yield mock_pipe


@pytest.fixture
def client(patch_pipeline):
    with TestClient(app) as c:
        yield c


class TestRequestIdTracing:

    def test_response_includes_x_request_id(self, client):
        resp = client.post("/api/v1/chat", json={"user_query": "Revenue?"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID")

    def test_chat_payload_echoes_request_id(self, client):
        resp = client.post("/api/v1/chat", json={"user_query": "Revenue?"})
        body = resp.json()
        assert body["request_id"] == resp.headers.get("X-Request-ID")

    def test_client_supplied_request_id_is_preserved(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"user_query": "Revenue?"},
            headers={"X-Request-ID": "trace-abc-123"},
        )
        assert resp.headers.get("X-Request-ID") == "trace-abc-123"
        assert resp.json()["request_id"] == "trace-abc-123"

    def test_health_returns_x_request_id(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID")


class TestStructuredValidationErrors:

    def test_empty_body_returns_structured_422(self, client):
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        assert "detail" in body

    def test_short_query_returns_structured_422(self, client):
        resp = client.post("/api/v1/chat", json={"user_query": "ab"})
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "VALIDATION_ERROR"

    def test_invalid_json_returns_structured_422(self, client):
        resp = client.post("/api/v1/chat", data="not-json")
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "VALIDATION_ERROR"


class TestArqWorkerSettings:

    def test_worker_settings_functions_resolve(self):
        # Both the ingestion job and the new audit-log persistence job are
        # registered on the Arq worker pool (Advanced API Feature #2).
        names = [f.__name__ for f in WorkerSettings.functions]
        assert "ingest_document_job" in names
        assert "log_audit_event_job" in names

    def test_worker_settings_redis_dsn_uses_localhost(self):
        assert WorkerSettings.redis_settings.host == "localhost"
        assert WorkerSettings.redis_settings.port == 6379

    def test_worker_settings_job_bounds(self):
        assert WorkerSettings.max_jobs >= 1
        assert WorkerSettings.job_timeout >= 60

    def test_ingest_job_is_coroutine(self):
        import inspect
        assert inspect.iscoroutinefunction(ingest_document_job)

    def test_enqueue_is_coroutine(self):
        import inspect
        assert inspect.iscoroutinefunction(enqueue_ingestion)

    @pytest.mark.asyncio
    async def test_enqueue_degrades_gracefully_on_redis_failure(self):
        import app.api.worker as worker_module
        with patch.object(worker_module, "create_pool", side_effect=RuntimeError("no redis")):
            ok = await enqueue_ingestion(
                Path("/tmp/fake.txt"), "AAPL", "2025", "t1"
            )
        assert ok is False
