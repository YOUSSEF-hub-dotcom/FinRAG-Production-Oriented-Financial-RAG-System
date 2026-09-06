"""
Advanced API Feature tests: Dynamic Rate Limiting (slowapi + Redis) and
Dynamic MongoDB Audit Logging (rag_audit_logs).

These tests verify:
  * Authenticated users are rate-limited per user_id (HTTP 429 on excess).
  * Guest users are rate-limited per client IP (HTTP 429 on excess).
  * The 429 body is a structured RateLimitErrorResponse.
  * Audit events are built + persisted to MongoDB (unit with injected fake
    collection, integration with the real local Mongo, and via the live
    /api/v1/chat endpoint + /api/v1/audit/logs).

The rate limiter is disabled by default in the test session (see conftest.py);
these tests flip ``limiter.enabled`` on for their own scope and flush the
Redis counters so runs are isolated.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient

import pymongo  # noqa: F401  (patched in fixtures)
import qdrant_client  # noqa: F401  (patched in fixtures)

# --- path setup so app modules import cleanly ---
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

from app.api.db_logger import MongoAuditLogger, get_audit_logger  # noqa: E402
from app.api.main import app  # noqa: E402
from app.api.rate_limiter import limiter  # noqa: E402
from app.api.schemas import AuditLogEvent, RateLimitErrorResponse  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def patch_pipeline():
    """Mock the pipeline + health deps, and stub the audit logger (no real Mongo)."""
    mock_pipe = MagicMock()
    mock_pipe.query.return_value = {
        "raw_output": json.dumps({
            "internal_thought": "Test reasoning $100B revenue.",
            "extracted_raw_data": "Revenue $100B.",
            "answer": "Apple reported $100B in revenue.",
            "sources": ["AAPL - 2025 - Business Overview - 10"],
        }),
        "parsed": MagicMock(
            answer="Apple reported $100B in revenue.",
            extracted_raw_data="Revenue $100B.",
            sources=["AAPL - 2025 - Business Overview - 10"],
            model_dump_json=lambda: '{"answer":"Apple reported $100B in revenue."}',
        ),
        "model_used": "openai/gpt-oss-20b",
        "fallback_triggered": False,
        "ttft_ms": 234.5,
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "cost_usd": 0.0001,
        "cache_hit": False,
        "pipeline_run_id": "test1234",
    }

    async def mock_stream(**kwargs):
        yield "Apple"
        yield " reported"
        yield " $100B"
        yield " in revenue."

    mock_pipe.query_stream = mock_stream
    mock_pipe.close = MagicMock()

    with (
        patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe) as mock_cls,
        patch.object(pymongo, "MongoClient") as mock_mongo,
        patch.object(qdrant_client, "QdrantClient") as mock_qdrant,
        patch.object(redis, "from_url") as mock_redis,
        patch("app.api.main.get_audit_logger") as mock_audit,
    ):
        mock_mongo.return_value.admin.command.return_value = {"ok": 1}
        mock_qdrant.return_value.get_collections.return_value = MagicMock()
        mock_redis.return_value.ping.return_value = True
        # get_audit_logger() returns a MagicMock logger whose log_event is a Mock.
        mock_audit.return_value = MagicMock()
        mock_audit.return_value.log_event = MagicMock()
        # get_recent_logs is awaited by the audit-logs endpoint.
        mock_audit.return_value.get_recent_logs = AsyncMock(return_value=[])

        yield {
            "pipeline": mock_pipe,
            "pipeline_cls": mock_cls,
            "audit_logger": mock_audit.return_value,
        }


@pytest.fixture
def client(patch_pipeline):
    """FastAPI TestClient with mocked pipeline + stubbed audit logger."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def enable_rate_limit():
    """Temporarily enable the rate limiter and flush Redis counters.

    Uses the real ``redis.Redis`` constructor (NOT ``from_url``) because the
    autouse ``patch_pipeline`` fixture mocks ``redis.from_url`` -- otherwise the
    flush would be a no-op and counts would leak across rate-limit tests.

    Restores the disabled state afterwards (baseline safety).
    """
    r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
    r.flushdb()
    previous = limiter.enabled
    limiter.enabled = True
    try:
        yield r
    finally:
        limiter.enabled = previous
        r.flushdb()
        r.close()


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:

    def test_guest_rate_limit_allows_then_429(self, client, enable_rate_limit):
        """Guests (IP-keyed) get 10 OK then a 429."""
        for i in range(10):
            resp = client.post(
                "/api/v1/chat", json={"user_query": "What is Apple's revenue?"}
            )
            assert resp.status_code == 200, f"request {i} expected 200"
        resp_429 = client.post(
            "/api/v1/chat", json={"user_query": "What is Apple's revenue?"}
        )
        assert resp_429.status_code == 429

    def test_guest_429_returns_structured_payload(self, client, enable_rate_limit):
        for _ in range(10):
            client.post("/api/v1/chat", json={"user_query": "Revenue please?"})
        resp = client.post("/api/v1/chat", json={"user_query": "Revenue please?"})
        assert resp.status_code == 429
        body = resp.json()
        assert body["error_code"] == "RATE_LIMIT_EXCEEDED"
        assert "10/minute" in body["limit"]
        assert body["retry_after_seconds"] >= 1
        # Validate it parses as the schema model.
        RateLimitErrorResponse(**body)

    def test_user_rate_limit_per_user_id(self, client, enable_rate_limit):
        """Authenticated callers are bucketed by user_id, not IP."""
        auth = "Bearer rag_demo_token_123"
        for i in range(10):
            resp = client.post(
                "/api/v1/chat",
                json={"user_query": "What is Apple's revenue?"},
                headers={"Authorization": auth},
            )
            assert resp.status_code == 200, f"user request {i} expected 200"
        resp_429 = client.post(
            "/api/v1/chat",
            json={"user_query": "What is Apple's revenue?"},
            headers={"Authorization": auth},
        )
        assert resp_429.status_code == 429
        assert resp_429.json()["error_code"] == "RATE_LIMIT_EXCEEDED"

    def test_user_and_guest_buckets_are_independent(self, client, enable_rate_limit):
        """Exhausting the guest bucket must not affect an authenticated user."""
        for _ in range(10):
            client.post("/api/v1/chat", json={"user_query": "Guest call"})
        # Guest now blocked...
        assert (
            client.post("/api/v1/chat", json={"user_query": "Guest call"}).status_code
            == 429
        )
        # ...but an authenticated user still gets through.
        auth_resp = client.post(
            "/api/v1/chat",
            json={"user_query": "User call"},
            headers={"Authorization": "Bearer rag_demo_token_123"},
        )
        assert auth_resp.status_code == 200

    def test_rate_limit_headers_present(self, client, enable_rate_limit):
        resp = client.post("/api/v1/chat", json={"user_query": "Headers test"})
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers


# ===========================================================================
# Audit Logging - unit (injected fake collection)
# ===========================================================================


class TestAuditLoggingUnit:

    def test_log_event_inserts_schema_fields(self):
        """With an injected fake collection the event is persisted with all keys."""
        fake_collection = MagicMock()
        fake_collection.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="abc123")
        )
        logger = MongoAuditLogger(collection=fake_collection)

        event = AuditLogEvent(
            request_id="req-1",
            user_identifier="user:demo_user",
            user_query={"raw": "What is revenue?", "standalone": "What is Apple revenue?"},
            retrieved_chunks=[
                {"chunk_id": "c1", "score": 0.91, "ticker_tag": "AAPL"}
            ],
            llm_prompts={
                "system_instruction": "You are a financial assistant.",
                "prompt_payload": "Context: ... Question: ...",
            },
            generated_response={
                "answer": "Apple reported $100B.",
                "sources": [{"chunk_id": "c1", "ticker": "AAPL"}],
            },
            execution_metadata={
                "total_latency_ms": 1234.5,
                "token_usage": {"total_tokens": 30},
                "model_cost_usd": 0.0001,
                "guardrail_status": "passed",
            },
        )

        result = asyncio.run(logger.log_event(event))
        assert result == "abc123"
        fake_collection.insert_one.assert_awaited_once()
        doc = fake_collection.insert_one.call_args.args[0]
        assert doc["request_id"] == "req-1"
        assert doc["user_identifier"] == "user:demo_user"
        assert doc["user_query"]["raw"] == "What is revenue?"
        assert doc["retrieved_chunks"][0]["ticker_tag"] == "AAPL"
        assert doc["generated_response"]["answer"] == "Apple reported $100B."
        assert doc["execution_metadata"]["total_latency_ms"] == 1234.5

    def test_log_event_swallows_errors(self):
        """A Mongo failure must never raise - it returns None instead."""
        fake_collection = MagicMock()
        fake_collection.insert_one = AsyncMock(side_effect=Exception("boom"))
        logger = MongoAuditLogger(collection=fake_collection)
        result = asyncio.run(logger.log_event({"request_id": "x"}))
        assert result is None


# ===========================================================================
# Audit Logging - integration (real local Mongo)
# ===========================================================================


@pytest.mark.asyncio
async def test_audit_log_roundtrip_real_mongo():
    """Insert + read-back against the real local MongoDB (skipped if down)."""
    logger = MongoAuditLogger(collection_name="rag_audit_logs_test")
    try:
        await logger.connect()
        # Verify reachability before asserting (skip if the store is down).
        await logger._client.admin.command("ping")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"MongoDB unavailable: {exc}")

    try:
        event = AuditLogEvent(
            request_id="integration-req",
            user_identifier="user:integration",
            user_query={"raw": "raw q", "standalone": "standalone q"},
            retrieved_chunks=[{"chunk_id": "c1", "score": 0.8}],
            llm_prompts={"system_instruction": "sys", "prompt_payload": "prompt"},
            generated_response={"answer": "ans", "sources": []},
            execution_metadata={"total_latency_ms": 10.0},
        )
        inserted = await logger.log_event(event)
        assert inserted is not None

        logs = await logger.get_recent_logs(limit=10, user_identifier="user:integration")
        assert any(l["request_id"] == "integration-req" for l in logs)
    finally:
        # Clean up the test collection.
        try:
            coll = await logger._get_collection()
            await coll.drop()
        except Exception:
            pass
        await logger.close()


# ===========================================================================
# Audit Logging - via live endpoints
# ===========================================================================


class TestAuditLoggingEndpoints:

    def test_chat_triggers_audit_log(self, client, patch_pipeline):
        """A successful chat schedules an audit event via the (stubbed) logger."""
        resp = client.post(
            "/api/v1/chat", json={"user_query": "What is Apple's revenue?"}
        )
        assert resp.status_code == 200
        audit_logger = patch_pipeline["audit_logger"]
        audit_logger.log_event.assert_called_once()
        event_arg = audit_logger.log_event.call_args.args[0]
        assert isinstance(event_arg, AuditLogEvent)
        assert event_arg.request_id
        assert event_arg.user_identifier  # IP or user

    def test_audit_logs_endpoint_returns_list(self, client, patch_pipeline):
        """GET /api/v1/audit/logs returns a structured list payload."""
        resp = client.get("/api/v1/audit/logs?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert "count" in body
        assert "logs" in body
        assert isinstance(body["logs"], list)
