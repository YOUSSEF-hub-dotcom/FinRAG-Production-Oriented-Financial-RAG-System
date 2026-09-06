import json, os
P = "artifacts/ab2_benchmark/candidates.json"
d = json.load(open(P))
for q in d["bench_queries"]:
    rbt = q.get("raw_by_ticker") or {}
    tickers = q.get("tickers") or []
    # rebuild chunks with ticker stamped from raw_by_ticker order
    if not rbt or not tickers:
        continue
    stamped = []
    for tk in tickers:
        for c in rbt.get(tk) or []:
            c = dict(c)
            c["ticker"] = tk
            stamped.append(c)
    if stamped:
        q["chunks"] = stamped
        q["count"] = len(stamped)
json.dump(d, open(P, "w"), indent=2)
# verify
from collections import Counter
d2 = json.load(open(P))
for q in d2["bench_queries"]:
    print(q["query_id"], dict(Counter(c.get("ticker") for c in q["chunks"])))