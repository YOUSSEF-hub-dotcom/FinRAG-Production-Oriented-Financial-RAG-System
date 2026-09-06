"""Live Demo QA — drives the RUNNING FastAPI backend (same path as the web
dashboard). Authenticates with the provided credentials, then posts each test
to /api/v1/chat. Captures answer + source citations + model, and cross-checks
ground truth from the primary 10-K text in MongoDB.

NOTE: /chat now restores per-session memory (keyed by session_id) instead of
resetting on every request, so cross-turn coreference ("first company") IS
preserved within a session. Use a stable session_id across turns to exercise it.
"""
import sys, json, re, urllib.request, urllib.error
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
from pymongo import MongoClient

BASE = "http://127.0.0.1:8000/api/v1"
EMAIL = "youssefaboali122@gmail.com"
PASSWORD = "12345abcde"

def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

st, resp = post("/auth/login", {"email": EMAIL, "password": PASSWORD})
print("LOGIN http=%s keys=%s" % (st, list(resp.keys())))
token = resp.get("access_token")
assert token, "LOGIN FAILED: " + str(resp)

def chat(q, ticker=None, fy=None, sid=None):
    body = {"user_query": q}
    if ticker: body["ticker"] = ticker
    if fy: body["fiscal_year"] = fy
    if sid: body["session_id"] = sid
    return post("/chat", body, token=token)

mongo = MongoClient("mongodb://localhost:27017")
chunks = mongo["financial_rag"]["raw_chunks"]
def gt(ticker, fy):
    docs = list(chunks.find({"ticker": ticker, "fiscal_year": str(fy)}))
    blob = "\n".join((d.get("raw_text") or d.get("text") or "") for d in docs)
    out = {}
    for label in ["Total net sales", "Total revenue", "Operating income",
                  "Research and Development", "Net income", "Intelligent Cloud"]:
        m = re.search(r"[\s\S]{0,40}" + re.escape(label) + r"[\s\S]{0,90}", blob, re.IGNORECASE)
        if m:
            out[label] = re.sub(r"\s+", " ", m.group(0)).strip()
    return out

def show(title, q, st, resp, ground=None):
    print("\n" + "=" * 72)
    print(title)
    print("Q: " + q)
    print("HTTP " + str(st) + ("  (MODEL %s | cache_hit %s)" %
          (resp.get("model_used"), resp.get("cache_hit")) if st == 200 else ""))
    print("-" * 72)
    if st != 200:
        print("ERROR BODY: " + str(resp)[:400]); return
    print("ANSWER:\n" + str(resp.get("answer", ""))[:1000])
    srcs = resp.get("sources", [])
    print("-" * 72)
    print("CITATIONS (%d):" % len(srcs))
    for s in srcs:
        print("  [%s|%s|%s] %s score=%.3f" % (s.get("ticker"), s.get("fiscal_year"),
              s.get("section"), s.get("chunk_id"), float(s.get("score") or 0)))
        print("     " + str(s.get("text_snippet", ""))[:130])
    if ground:
        print("-" * 72)
        print("GROUND TRUTH (primary 10-K, MongoDB):")
        for k, v in ground.items():
            if isinstance(v, dict):
                print("  " + k + ":")
                for kk, vv in v.items():
                    print("    " + kk + ": " + str(vv))
            else:
                print("  " + k + ": " + str(v))
    print("=" * 72)

sid = "qa-session-A"
q1 = "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025."
st1, r1 = chat(q1, "ALL", "2025", sid)
show("TEST 1 - Multi-Ticker Quantitative & Balanced Retrieval", q1, st1, r1,
     {t: gt(t, 2025) for t in ("AAPL", "MSFT", "NVDA")})

q2 = "How much did the first company spend on Research and Development in that same fiscal year?"
st2, r2 = chat(q2, "ALL", "2025", sid)
show("TEST 2 - Memory & Coreference Resolution", q2, st2, r2,
      {"AAPL FY2025 R&D (ground truth)": gt("AAPL", 2025).get("Research and Development", "n/a")})

q3 = "What is Amazon's net income for FY2025, and what is Tesla's autonomous driving roadmap?"
st3, r3 = chat(q3, "ALL", "2025", sid)
show("TEST 3 - Guardrail & Out-of-Scope (OOC) Detection", q3, st3, r3)

q4 = "What was Microsoft's Total Revenue and Intelligent Cloud segment revenue in FY2025?"
st4, r4 = chat(q4, "MSFT", "2025", "qa-session-B")
msft_blob = "\n".join((d.get("raw_text") or "") for d in
                     chunks.find({"ticker": "MSFT", "fiscal_year": "2025"}))
msft_fig = bool(re.search(r"106[,\d]*265", msft_blob) and
                re.search(r"281[,\d]*724", msft_blob))
show("TEST 4 - Single-Ticker High Precision Segment Extraction", q4, st4, r4,
     {"MSFT FY2025": gt("MSFT", 2025),
      "Corpus contains 281,724 / 106,265?": str(msft_fig)})

print("\nQA COMPLETE")
