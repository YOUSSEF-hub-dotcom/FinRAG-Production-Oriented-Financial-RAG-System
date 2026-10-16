"""Focused tests for Phase 2 (D2): streaming guardrail enforcement.

Verifies that ``POST /api/v1/chat/stream`` applies the SAME opt-in
``GUARDRAIL_ENFORCE`` policy as the sync ``POST /api/v1/chat``:

* disabled (default) -> advisory: the real verdict is reported/persisted but
  the streamed answer is left intact.
* enabled + ``failed`` -> the delivered ``answer`` becomes the guardrail safe
  fallback, ``sources`` are dropped, and generated content is withheld from the
  client until the verdict is known.
* enabled + ``error`` / ``timeout`` -> the unverified answer is withheld behind
  the distinct "verification unavailable" notice (never swapped to the FAIL
  fallback, never treated as a PASS).
* enabled + ``not_run`` -> delivered unchanged (no verification was attempted:
  GENERAL / security-halt), never treated as a PASS.

All external services are mocked; no network/DB access and no paid API.
"""

import json
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

from app.api import main  # noqa: E402
from app.api.main import app  # noqa: E402

# Sentinel "safe fallback" the fake guardrail returns on a definite FAIL.
SAFE = "SAFE FALLBACK: numerical verification failed."
VERIFIED = "Apple reported $394,328M in revenue."
HALLUCINATED = "Apple reported $999,999M in revenue."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_token(answer: str, sources: list[str]) -> str:
    return json.dumps(
        {
            "internal_thought": "reasoning with $394,328M",
            "extracted_raw_data": "Net sales: $394,328M",
            "answer": answer,
            "sources": sources,
        }
    )


def _install_stream(
    mock_pipe: MagicMock,
    *,
    tokens: list[str],
    verdict_status: str,
    final_output: str | None,
    with_context: bool = True,
) -> None:
    """Install a fake async ``query_stream`` that mimics the real contract.

    Tokens are yielded first (as the real generator does), then the per-request
    ``guardrail_sink`` is populated -- exactly the ordering the endpoint relies
    on to buffer generated content until the verdict is available.
    """

    async def _stream(**kwargs):
        sink = kwargs.get("guardrail_sink")
        context_sink = kwargs.get("context_sink")
        if with_context and context_sink is not None:
            context_sink.append(
                {
                    "chunk_id": "AAPL_txt_0000",
                    "score": 0.9,
                    "payload": {
                        "ticker": "AAPL",
                        "fiscal_year": "2025",
                        "section": "Revenue",
                        "text": "Net sales were $394,328M.",
                        "raw_text": "Net sales were $394,328M.",
                    },
                }
            )
        for t in tokens:
            yield t
        if sink is not None:
            if verdict_status == "not_run":
                sink.setdefault("passed", False)
                sink.setdefault("status", "not_run")
            else:
                sink.update(
                    {
                        "passed": verdict_status == "passed",
                        "status": verdict_status,
                        "final_output": final_output,
                        "verified_claims": [],
                        "failed_claims": [],
                        "error": None,
                    }
                )

    mock_pipe.query_stream = _stream


def _sse_events(text: str) -> list[tuple[str, str]]:
    """Parse an SSE body into an ordered ``(event, data)`` list."""
    events: list[tuple[str, str]] = []
    event = "message"
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            events.append((event, line[len("data:"):].strip()))
    return events


def _event_payloads(text: str) -> dict[str, list[str]]:
    """Group SSE data payloads by event name."""
    out: dict[str, list[str]] = {}
    for event, data in _sse_events(text):
        out.setdefault(event, []).append(data)
    return out


@pytest.fixture(scope="module")
def api_client():
    """TestClient over the real app with the pipeline mocked (no external I/O)."""
    mock_pipe = MagicMock()
    mock_pipe.close = MagicMock()
    with (
        patch("app.api.main.get_or_create_pipeline", return_value=mock_pipe),
        patch("app.api.main._run_warmup", new=AsyncMock()),
    ):
        with TestClient(app) as c:
            yield c, mock_pipe


def _post_stream(client, captured: list, *, enforce: bool):
    with (
        patch.object(main, "GUARDRAIL_ENFORCE", enforce),
        patch.object(
            main,
            "_schedule_audit_log",
            side_effect=lambda _bt, event: captured.append(event),
        ),
    ):
        return client.post(
            "/api/v1/chat/stream",
            json={"user_query": "What is Apple's revenue?", "ticker": "AAPL"},
        )


# ---------------------------------------------------------------------------
# Policy predicate (fast, no HTTP)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "enforce,status,final_output,expected",
    [
        (True, "failed", SAFE, True),
        (False, "failed", SAFE, False),
        (True, "error", SAFE, False),
        (True, "not_run", None, False),
        (True, "failed", "", False),
        (True, "passed", VERIFIED, False),
    ],
)
def test_should_swap_policy(enforce, status, final_output, expected):
    with patch.object(main, "GUARDRAIL_ENFORCE", enforce):
        assert main._guardrail_should_swap(status, final_output) is expected


@pytest.mark.parametrize(
    "enforce,status,expected",
    [
        (True, "error", True),
        (True, "timeout", True),
        (True, "failed", False),
        (True, "not_run", False),
        (True, "passed", False),
        (False, "error", False),
    ],
)
def test_should_withhold_policy(enforce, status, expected):
    with patch.object(main, "GUARDRAIL_ENFORCE", enforce):
        assert main._guardrail_should_withhold(status) is expected


# ---------------------------------------------------------------------------
# Streaming enforcement over HTTP
# ---------------------------------------------------------------------------

def test_enforce_on_fail_withholds_hallucination_and_returns_fallback(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(HALLUCINATED, ["AAPL - 2025 - Revenue - 10"])],
        verdict_status="failed",
        final_output=SAFE,
    )
    captured: list = []

    resp = _post_stream(client, captured, enforce=True)

    assert resp.status_code == 200
    assert HALLUCINATED not in resp.text  # withheld until verdict => never sent
    payloads = _event_payloads(resp.text)
    assert payloads.get("answer") == [SAFE]
    assert "sources" not in payloads
    assert payloads.get("done") == [""]
    assert captured, "audit event was scheduled"
    event = captured[-1]
    assert event.generated_response["answer"] == SAFE
    assert event.execution_metadata["guardrail_status"] == "failed"
    assert event.execution_metadata["guardrail_enforced"] is True


def test_enforce_on_pass_streams_verified_answer_with_sources(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(VERIFIED, ["AAPL - 2025 - Revenue - 10"])],
        verdict_status="passed",
        final_output=VERIFIED,
    )
    captured: list = []

    resp = _post_stream(client, captured, enforce=True)

    assert resp.status_code == 200
    payloads = _event_payloads(resp.text)
    assert payloads.get("answer") == [VERIFIED]
    assert "sources" in payloads
    assert captured[-1].generated_response["answer"] == VERIFIED
    assert captured[-1].execution_metadata["guardrail_status"] == "passed"
    assert captured[-1].execution_metadata["guardrail_enforced"] is False


def test_enforce_off_fail_is_advisory(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(HALLUCINATED, ["AAPL - 2025 - Revenue - 10"])],
        verdict_status="failed",
        final_output=SAFE,
    )
    captured: list = []

    resp = _post_stream(client, captured, enforce=False)

    assert resp.status_code == 200
    payloads = _event_payloads(resp.text)
    assert payloads.get("answer") == [HALLUCINATED]
    assert HALLUCINATED in resp.text
    event = captured[-1]
    assert event.generated_response["answer"] == HALLUCINATED
    assert event.execution_metadata["guardrail_status"] == "failed"
    assert event.execution_metadata["guardrail_enforced"] is False


def test_enforce_on_error_withholds_unverified_answer(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(VERIFIED, ["AAPL - 2025 - Revenue - 10"])],
        verdict_status="error",
        final_output=SAFE,
    )
    captured: list = []

    resp = _post_stream(client, captured, enforce=True)

    # Evaluator errored -> the unverified generated content must not be released.
    assert VERIFIED not in resp.text
    payloads = _event_payloads(resp.text)
    assert payloads.get("answer") == [main.GUARDRAIL_UNAVAILABLE_MESSAGE]
    assert payloads.get("answer") != [SAFE]  # distinct from the FAIL fallback
    assert "sources" not in payloads
    assert payloads.get("done") == [""]
    assert captured[-1].generated_response["answer"] == main.GUARDRAIL_UNAVAILABLE_MESSAGE
    assert captured[-1].execution_metadata["guardrail_status"] == "error"
    assert captured[-1].execution_metadata["guardrail_enforced"] is True


def test_enforce_on_not_run_never_swaps(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(VERIFIED, [])],
        verdict_status="not_run",
        final_output=None,
    )
    captured: list = []

    resp = _post_stream(client, captured, enforce=True)

    # not_run = no verification was attempted -> deliver (not withheld).
    assert VERIFIED in resp.text
    payloads = _event_payloads(resp.text)
    assert payloads.get("answer") == [VERIFIED]
    assert captured[-1].execution_metadata["guardrail_status"] == "not_run"
    assert captured[-1].execution_metadata["guardrail_enforced"] is False


# ---------------------------------------------------------------------------
# Transport identity: the stream endpoint permits anonymous callers (guests are
# rate-limited by IP via app.api.rate_limiter.get_current_user). A missing or
# unusable token must NOT break the endpoint -- it degrades to guest identity.
# ---------------------------------------------------------------------------

def test_stream_allows_anonymous(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(VERIFIED, [])],
        verdict_status="passed",
        final_output=VERIFIED,
    )
    resp = client.post("/api/v1/chat/stream", json={"user_query": "Revenue?"})
    assert resp.status_code == 200


def test_stream_unusable_bearer_degrades_to_guest(api_client):
    client, mock_pipe = api_client
    _install_stream(
        mock_pipe,
        tokens=[_json_token(VERIFIED, [])],
        verdict_status="passed",
        final_output=VERIFIED,
    )
    resp = client.post(
        "/api/v1/chat/stream",
        json={"user_query": "Revenue?"},
        headers={"Authorization": "Bearer expired.or.invalid.token"},
    )
    assert resp.status_code == 200
