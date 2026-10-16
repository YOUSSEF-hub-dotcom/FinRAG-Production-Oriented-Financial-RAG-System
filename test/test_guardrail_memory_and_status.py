"""Focused tests for the guardrail enforcement closure:

1. Session-memory correctness -- when enforcement replaces/withholds the
   delivered answer (FAIL swap or evaluator ERROR), the unverified generated
   answer must NOT be retained as trusted conversational memory, in BOTH the
   synchronous and streaming paths. When enforcement is off (or the verdict is
   PASS / not_run) the existing memory behaviour is preserved.

2. ERROR / timeout / not_run semantics -- statuses stay distinct from PASS/FAIL;
   under enforcement an evaluator ERROR/timeout withholds the unverified answer
   behind a distinct notice, while ``not_run`` (no verification attempted) is
   delivered. Covered at the route level for both /chat and /chat/stream.

All external services are mocked; no network/DB access and no paid API.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
from generator import ConversationMemory  # noqa: E402

SAFE = "SAFE FALLBACK: numerical verification failed."
UNAVAILABLE = main.GUARDRAIL_UNAVAILABLE_MESSAGE
GENERATED = "Apple reported $999,999M in revenue."
SECURITY_RESPONSE = "I can't help with that request."


def _raw(answer: str) -> str:
    return json.dumps(
        {
            "internal_thought": "reasoning",
            "extracted_raw_data": "Net sales: $394,328M",
            "answer": answer,
            "sources": ["AAPL - 2025 - Revenue - 10"],
        }
    )


class _FakeGenerator:
    def __init__(self):
        self._memory = ConversationMemory(k=3)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory


class _FakePipe:
    """Minimal pipeline double exposing the memory + query contracts the API uses."""

    def __init__(self, result: dict):
        self._result = result
        self._generator = _FakeGenerator()
        self.closed = False

    def reset_memory(self) -> None:
        self._generator._memory.clear()

    def _append_turn(self, query: str) -> None:
        self._generator._memory.add_human(query)
        self._generator._memory.add_ai(self._result.get("raw_output", ""))

    def query(self, **kwargs):
        self._append_turn(kwargs.get("user_query", ""))
        return dict(self._result)

    async def query_stream(self, **kwargs):
        self._append_turn(kwargs.get("user_query", ""))
        context_sink = kwargs.get("context_sink")
        if context_sink is not None:
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
        yield self._result.get("raw_output", "")
        sink = kwargs.get("guardrail_sink")
        if sink is None:
            return
        status = self._result.get("guardrail_status")
        if status is None or status == "silent":
            # Deliberate security-halt (or omitted verdict): the real pipeline
            # returns before it ever touches the guardrail sink, so leave the
            # sink empty -- enforcement must still deliver the canned response.
            return
        if status == "not_run":
            sink.setdefault("passed", False)
            sink.setdefault("status", "not_run")
        else:
            sink.update(
                {
                    "passed": status == "passed",
                    "status": status,
                    "final_output": self._result.get("guardrail_final_output"),
                    "verified_claims": [],
                    "failed_claims": [],
                    "error": None,
                }
            )

    def close(self) -> None:
        self.closed = True


def _result(status: str, *, passed: bool, final_output=None, answer: str = GENERATED) -> dict:
    return {
        "raw_output": _raw(answer),
        "parsed": SimpleNamespace(answer=answer),
        "model_used": "test-stub",
        "cache_hit": False,
        "guardrail_status": status,
        "guardrail_passed": passed,
        "guardrail_final_output": final_output,
        "retrieved_chunks": [
            {
                "chunk_id": "AAPL_txt_0000",
                "score": 0.9,
                "payload": {
                    "ticker": "AAPL",
                    "fiscal_year": "2025",
                    "section": "Revenue",
                    "raw_text": "Net sales were $394,328M.",
                    "text": "Net sales were $394,328M.",
                },
            }
        ],
    }


def _security_result() -> dict:
    """A security-halt result: canned response, no guardrail verdict at all."""
    r = _result("not_run", passed=False)
    r.pop("guardrail_status", None)
    r["parsed"] = None
    r["raw_output"] = SECURITY_RESPONSE
    return r


@pytest.fixture(scope="module")
def api_client():
    fake_pipe = _FakePipe(_result("not_run", passed=False))
    with (
        patch("app.api.main.get_or_create_pipeline", return_value=fake_pipe),
        patch("app.api.main._run_warmup", new=AsyncMock()),
    ):
        with TestClient(app) as c:
            yield c, fake_pipe


@pytest.fixture(autouse=True)
def _clear_sessions():
    main._SESSION_MEMORY.clear()
    yield
    main._SESSION_MEMORY.clear()


def _post_chat(client, fake_pipe, result, *, enforce, session_id=None):
    fake_pipe._result = result
    payload = {"user_query": "What is Apple's revenue?", "ticker": "AAPL"}
    if session_id:
        payload["session_id"] = session_id
    captured: list = []
    with (
        patch.object(main, "GUARDRAIL_ENFORCE", enforce),
        patch.object(main, "_schedule_audit_log", side_effect=lambda _bt, e: captured.append(e)),
    ):
        resp = client.post("/api/v1/chat", json=payload)
    return resp, captured


def _post_stream(client, fake_pipe, result, *, enforce, session_id=None):
    fake_pipe._result = result
    payload = {"user_query": "What is Apple's revenue?", "ticker": "AAPL"}
    if session_id:
        payload["session_id"] = session_id
    captured: list = []
    with (
        patch.object(main, "GUARDRAIL_ENFORCE", enforce),
        patch.object(main, "_schedule_audit_log", side_effect=lambda _bt, e: captured.append(e)),
    ):
        resp = client.post("/api/v1/chat/stream", json=payload)
    return resp, captured


def _last_ai(session_id: str) -> str:
    msgs = main._SESSION_MEMORY[session_id]
    for m in reversed(msgs):
        if type(m).__name__ == "AIMessage":
            return m.content
    return ""


# ---------------------------------------------------------------------------
# Synchronous path
# ---------------------------------------------------------------------------

def test_sync_enforce_off_fail_is_advisory_and_memory_unchanged(api_client):
    client, fake_pipe = api_client
    res, _ = _post_chat(
        client,
        fake_pipe,
        _result("failed", passed=False, final_output=SAFE),
        enforce=False,
        session_id="s-advisory",
    )
    body = res.json()
    assert res.status_code == 200
    assert body["answer"] == GENERATED
    assert body["sources"], "advisory path keeps sources"
    assert body["guardrail_status"]["status"] == "failed"
    # Existing behaviour preserved: raw generated JSON retained in memory.
    assert _last_ai("s-advisory") == _raw(GENERATED)


def test_sync_enforce_on_fail_swaps_and_sanitizes_memory(api_client):
    client, fake_pipe = api_client
    res, captured = _post_chat(
        client,
        fake_pipe,
        _result("failed", passed=False, final_output=SAFE),
        enforce=True,
        session_id="s-fail",
    )
    body = res.json()
    assert body["answer"] == SAFE
    assert body["sources"] == []
    assert body["guardrail_status"]["status"] == "failed"
    assert body["guardrail_status"]["passed"] is False
    # The unverified generated answer must NOT be retained as trusted memory.
    assert _last_ai("s-fail") == SAFE
    assert captured[-1].generated_response["answer"] == SAFE
    assert captured[-1].execution_metadata["guardrail_enforced"] is True


def test_sync_enforce_on_error_withholds_and_sanitizes_memory(api_client):
    client, fake_pipe = api_client
    res, _ = _post_chat(
        client,
        fake_pipe,
        _result("error", passed=False, final_output=SAFE),
        enforce=True,
        session_id="s-error",
    )
    body = res.json()
    assert body["answer"] == UNAVAILABLE
    assert body["answer"] != SAFE
    assert body["sources"] == []
    assert body["guardrail_status"]["status"] == "error"
    assert _last_ai("s-error") == UNAVAILABLE


def test_sync_enforce_on_not_run_delivers_and_keeps_raw_memory(api_client):
    client, fake_pipe = api_client
    res, _ = _post_chat(
        client,
        fake_pipe,
        _result("not_run", passed=False),
        enforce=True,
        session_id="s-notrun",
    )
    body = res.json()
    assert body["answer"] == GENERATED
    assert body["guardrail_status"]["status"] == "not_run"
    # No verification was attempted -> memory keeps existing (raw) behaviour.
    assert _last_ai("s-notrun") == _raw(GENERATED)


def test_sync_memory_is_session_isolated(api_client):
    client, fake_pipe = api_client
    _post_chat(
        client, fake_pipe, _result("failed", passed=False, final_output=SAFE),
        enforce=True, session_id="iso-a",
    )
    _post_chat(
        client, fake_pipe, _result("not_run", passed=False),
        enforce=True, session_id="iso-b",
    )
    assert _last_ai("iso-a") == SAFE
    assert GENERATED in _last_ai("iso-b")
    assert set(main._SESSION_MEMORY) == {"iso-a", "iso-b"}


def test_sync_no_session_is_stateless(api_client):
    client, fake_pipe = api_client
    _post_chat(
        client, fake_pipe, _result("failed", passed=False, final_output=SAFE),
        enforce=True, session_id=None,
    )
    assert main._SESSION_MEMORY == {}


def test_sync_enforce_on_security_halt_delivered_as_not_run(api_client):
    """A deliberate security-halt has NO verdict: it must be delivered (never
    withheld as if verification failed) and reported as not_run, not PASS."""
    client, fake_pipe = api_client
    res, captured = _post_chat(
        client, fake_pipe, _security_result(), enforce=True, session_id="s-sec",
    )
    body = res.json()
    assert res.status_code == 200
    assert body["answer"] == SECURITY_RESPONSE
    assert body["answer"] != UNAVAILABLE
    assert body["guardrail_status"]["status"] == "not_run"
    assert body["guardrail_status"]["passed"] is False
    assert captured[-1].execution_metadata["guardrail_enforced"] is False
    # Delivered as-is -> memory keeps the security response unchanged.
    assert _last_ai("s-sec") == SECURITY_RESPONSE


def test_sync_enforce_off_missing_parse_error_is_advisory(api_client):
    """GUARDRAIL_ENFORCE=false: a financial-route verification failure (new
    ``error`` status) is reported but the answer is delivered unchanged."""
    client, fake_pipe = api_client
    res, _ = _post_chat(
        client, fake_pipe, _result("error", passed=False), enforce=False,
        session_id="s-advisory-err",
    )
    body = res.json()
    assert body["answer"] == GENERATED
    assert body["sources"], "advisory path keeps sources"
    assert body["guardrail_status"]["status"] == "error"
    assert _last_ai("s-advisory-err") == _raw(GENERATED)


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------

def _sse_answer(text: str) -> list[str]:
    out, event = [], "message"
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:") and event == "answer":
            out.append(line[len("data:"):].strip())
    return out


def test_stream_enforce_on_error_withholds_and_sanitizes_memory(api_client):
    client, fake_pipe = api_client
    res, captured = _post_stream(
        client,
        fake_pipe,
        _result("error", passed=False, final_output=SAFE),
        enforce=True,
        session_id="st-error",
    )
    assert res.status_code == 200
    assert GENERATED not in res.text
    assert _sse_answer(res.text) == [UNAVAILABLE]
    assert _last_ai("st-error") == UNAVAILABLE
    assert captured[-1].execution_metadata["guardrail_status"] == "error"
    assert captured[-1].execution_metadata["guardrail_enforced"] is True


def test_stream_enforce_on_fail_swaps_and_sanitizes_memory(api_client):
    client, fake_pipe = api_client
    res, _ = _post_stream(
        client,
        fake_pipe,
        _result("failed", passed=False, final_output=SAFE),
        enforce=True,
        session_id="st-fail",
    )
    assert GENERATED not in res.text
    assert _sse_answer(res.text) == [SAFE]
    assert _last_ai("st-fail") == SAFE


def test_stream_enforce_off_fail_is_advisory_and_memory_unchanged(api_client):
    client, fake_pipe = api_client
    res, _ = _post_stream(
        client,
        fake_pipe,
        _result("failed", passed=False, final_output=SAFE),
        enforce=False,
        session_id="st-advisory",
    )
    assert GENERATED in res.text
    assert _sse_answer(res.text) == [GENERATED]
    assert _last_ai("st-advisory") == _raw(GENERATED)


def test_stream_enforce_on_not_run_delivers_and_keeps_raw_memory(api_client):
    client, fake_pipe = api_client
    res, _ = _post_stream(
        client,
        fake_pipe,
        _result("not_run", passed=False),
        enforce=True,
        session_id="st-notrun",
    )
    assert _sse_answer(res.text) == [GENERATED]
    assert _last_ai("st-notrun") == _raw(GENERATED)


def test_stream_enforce_on_security_halt_delivered_as_not_run(api_client):
    """Streaming security-halt: the pipeline never populates the verdict sink,
    so enforcement must not withhold the canned response; audit reports not_run."""
    client, fake_pipe = api_client
    res, captured = _post_stream(
        client,
        fake_pipe,
        _security_result(),
        enforce=True,
        session_id="st-sec",
    )
    assert res.status_code == 200
    assert SECURITY_RESPONSE in res.text
    assert _sse_answer(res.text) == [SECURITY_RESPONSE]
    assert captured[-1].execution_metadata["guardrail_status"] == "not_run"
    assert captured[-1].execution_metadata["guardrail_enforced"] is False
    assert _last_ai("st-sec") == SECURITY_RESPONSE


# ---------------------------------------------------------------------------
# Startup guardrail-configuration validation (contradictory-config fail-fast)
# ---------------------------------------------------------------------------

def test_validate_guardrail_config_raises_when_enforce_on_and_disabled():
    fake = SimpleNamespace(_enable_guardrail=False, _guardrail=object())
    with patch.object(main, "GUARDRAIL_ENFORCE", True):
        with pytest.raises(RuntimeError, match="GUARDRAIL_ENFORCE=true"):
            main._validate_guardrail_config(fake)


def test_validate_guardrail_config_raises_when_enforce_on_and_no_guardrail():
    fake = SimpleNamespace(_enable_guardrail=True, _guardrail=None)
    with patch.object(main, "GUARDRAIL_ENFORCE", True):
        with pytest.raises(RuntimeError, match="GUARDRAIL_ENFORCE=true"):
            main._validate_guardrail_config(fake)


def test_validate_guardrail_config_ok_when_enforce_off():
    fake = SimpleNamespace(_enable_guardrail=False, _guardrail=None)
    with patch.object(main, "GUARDRAIL_ENFORCE", False):
        main._validate_guardrail_config(fake)  # no-op, must not raise


def test_validate_guardrail_config_ok_when_enforce_on_and_enabled():
    fake = SimpleNamespace(_enable_guardrail=True, _guardrail=object())
    with patch.object(main, "GUARDRAIL_ENFORCE", True):
        main._validate_guardrail_config(fake)  # valid, must not raise
