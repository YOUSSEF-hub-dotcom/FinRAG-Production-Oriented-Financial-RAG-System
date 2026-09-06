#!/usr/bin/env python3
"""Forensic SSE streaming-latency diagnostic (EXTERNAL client, read-only).

Signs up a fresh user, then POSTs to /api/v1/chat/stream and records, per
SSE event, the arrival timing relative to request start. This measures the
raw network-visible stream exactly as the browser/SSE client would see it.

Output: one line per event: <elapsed_ms> <event> <data_len> <prefix...>
plus a summary line. Used by _run_stream_diag.sh.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"


def api_post(path, body, token=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def signup(idx):
    email = "diag_stream_%d_%d@example.com" % (idx, int(time.time()))
    data = api_post("/api/v1/auth/signup",
                    {"email": email, "password": "Diagnostic!123", "full_name": "Diag Stream"})
    return data["access_token"]


def stream_and_trace(label, query, ticker, session_id, token):
    body = {"user_query": query, "ticker": ticker}
    if session_id:
        body["session_id"] = session_id
    req = urllib.request.Request(BASE + "/api/v1/chat/stream", data=json.dumps(body).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "text/event-stream")

    t0 = time.time()
    events = []
    first_event_ms = None
    token_events = 0
    token_bytes = 0
    last_token_ms = None
    done_ms = None
    answer_ms = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            buf = b""
            event_type = "message"
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", "replace").strip("\r")
                    if not line:
                        continue
                    now_ms = int((time.time() - t0) * 1000)
                    if first_event_ms is None:
                        first_event_ms = now_ms
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        prefix = data[:80].replace("\n", " ")
                        events.append((now_ms, event_type, len(data), prefix))
                        if event_type == "token":
                            token_events += 1
                            token_bytes += len(data)
                            last_token_ms = now_ms
                        elif event_type == "answer":
                            answer_ms = now_ms
                        elif event_type == "done":
                            done_ms = now_ms
    except Exception as exc:
        now_ms = int((time.time() - t0) * 1000)
        events.append((now_ms, "EXC", 0, str(exc)))
        if first_event_ms is None:
            first_event_ms = now_ms

    total_ms = int((time.time() - t0) * 1000)
    print("===== %s =====" % label)
    print("query=%r ticker=%r" % (query, ticker))
    if not events:
        print("NO EVENTS");
        return
    # inter-event gaps for token events
    token_times = [e[0] for e in events if e[1] == "token"]
    gaps = [b - a for a, b in zip(token_times, token_times[1:])] if len(token_times) > 1 else []
    print("total_ms=%d first_event_ms=%s first_token_ms=%s" % (
        total_ms, first_event_ms, token_times[0] if token_times else None))
    print("token_events=%d token_bytes=%d last_token_ms=%s answer_ms=%s done_ms=%s" % (
        token_events, token_bytes, last_token_ms, answer_ms, done_ms))
    if gaps:
        print("token_gap_ms min=%d med=%d max=%d p95=%d" % (
            min(gaps), sorted(gaps)[len(gaps)//2], max(gaps),
            sorted(gaps)[int(len(gaps)*0.95)-1] if gaps else 0))
    for ev in events:
        print("  %5dms %-8s %5d  %s" % ev)
    print()


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    token = signup(idx)
    base_session = "diag_stream_%d" % idx

    # TEST 1: simple single-ticker
    stream_and_trace(
        "TEST1 simple single-ticker",
        "How much revenue did Apple report in fiscal year 2025?",
        "ALL", None, token)

    # TEST 2: multi-ticker (matches prior repro scope)
    stream_and_trace(
        "TEST2 multi-ticker",
        "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025",
        "ALL", None, token)

    # TEST 3: memory follow-up same session (Turn 2)
    session2 = base_session + "_mem"
    stream_and_trace(
        "TEST3a memory turn 1",
        "How much did Apple spend on Research and Development in fiscal year 2025?",
        "ALL", session2, token)
    stream_and_trace(
        "TEST3b memory turn 2 follow-up",
        "How much did the second company spend on Research and Development in that same fiscal year?",
        "ALL", session2, token)


if __name__ == "__main__":
    main()
