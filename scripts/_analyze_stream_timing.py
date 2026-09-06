import json

log = "/home/youssef/Financial_RAG/uvicorn_streamdiag.log"
evs = []
for ln in open(log):
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    try:
        j = json.loads(ln)
    except Exception:
        continue
    evs.append((j.get("timestamp", ""), j.get("message", ""), j.get("function", "")))

print("events:", len(evs))
for t, m, f in evs:
    if any(k in m for k in
           ["Cross-entity", "Context prepared", "Stream complete",
            "Memory coreference", "api_request", "Hybrid search start",
            "Rerank complete: 39", "Rerank complete: 40",
            "Balanced multi-ticker retrieval"]):
        print(t[11:26], "|", (f or "")[:12].ljust(12), "|", m[:120])
