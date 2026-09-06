"""Experiment D (corrected): TOP_N 8 vs 6 simulation.

Model matches production selection:
  - single ticker: global top-N of the ticker's scored list
  - multi ticker : per-ticker top-N (production runs aprocess_scored per ticker
                   then merges + dedups), so each ticker is represented

Uses saved fp32-large scores (expA_fp16.json) — no model re-run.
No production changes.
"""
import json, os, sys

sys.path.insert(0, "/home/youssef/Financial_RAG")

CAND = "artifacts/ab2_benchmark/candidates.json"
SCORES = "artifacts/ab2_benchmark/expA_fp16.json"
OUT = "artifacts/ab2_benchmark/expD_topn.json"

N_TOP = [8, 6]


def evidence(chunks):
    joined = "\n".join((c.get("text") or "") for c in chunks)
    tbls = [c for c in chunks if c.get("contains_table")]
    return {
        "apple_rev_416161": "$416,161" in joined,
        "apple_opinc_133050": "$133,050" in joined,
        "msft_rev_281724": "$281,724" in joined,
        "nvda_rev_130497": "$130,497" in joined,
        "msft_netinc_101832": "$101,832" in joined,
        "iphone_209586": "$209,586" in joined,
        "services_109158": "$109,158" in joined,
        "table_chunks": len(tbls),
        "tickers_present": sorted(set(c.get("ticker") for c in chunks)),
    }


def select_topn(chunks, scores, q, top_n):
    """Production-equivalent selection."""
    tickers_sorted = sorted(set(c.get("ticker") for c in chunks))
    if len(tickers_sorted) > 1:
        # per-ticker top-N (multi-ticker path selects per ticker then merges)
        selected = []
        seen = set()
        for tk in TICKER_ORDER:
            idxs = [i for i, c in enumerate(chunks) if c.get("ticker") == tk]
            idxs.sort(key=lambda i: scores[i], reverse=True)
            for i in idxs[:top_n]:
                cid = chunks[i].get("chunk_id")
                if cid in seen:
                    continue
                seen.add(cid)
                selected.append(chunks[i])
        return selected
    else:
        idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in idx[:top_n]]


# preserve the query's intended ticker order (AAPL, MSFT, NVDA)
TICKER_ORDER = ["AAPL", "MSFT", "NVDA"]


def main():
    with open(CAND) as f:
        data = json.load(f)
    queries = {q["query_id"]: q for q in data["bench_queries"]}
    with open(SCORES) as f:
        sc = json.load(f)

    out = {"per_query": {}}
    print(f"{'Query':<6} {'Type':<20} {'n8→n6 kept':>11} {'n6 ev_eq':>9} "
          f"{'n6 tables':>10} {'n6 tickers':>14} {'n6 dropped_ev':>14}")
    print("  " + "-" * 84)
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q6", "Q7", "Q8"]:
        q = queries[qid]
        scores = sc["fp32"]["scores"][qid]
        chunks = q["chunks"]
        evs = {}
        sel = {}
        for n in N_TOP:
            s = select_topn(chunks, scores, q, n)
            sel[str(n)] = [c.get("chunk_id") for c in s]
            evs[str(n)] = evidence(s)
        kept = evs["8"]["tickers_present"] == evs["6"]["tickers_present"]
        lost = [k for k in evs["8"]
                if k.startswith(("apple", "msft", "nvda", "iphone", "services"))
                and evs["8"][k] and not evs["6"][k]]
        out["per_query"][qid] = {
            "category": q["category"], "count": len(chunks),
            f"top8_evidence": evs["8"], "top6_evidence": evs["6"],
            "top8_ids": sel["8"], "top6_ids": sel["6"],
            "ticker_coverage_kept": kept,
            "table_chunks_n8": evs["8"]["table_chunks"],
            "table_chunks_n6": evs["6"]["table_chunks"],
            "tickers_n6": evs["6"]["tickers_present"],
            "evidence_lost_n6": lost,
            "context_chunks_n8": len(sel["8"]), "context_chunks_n6": len(sel["6"]),
        }
        print(f"{qid:<6} {q['category']:<20} {str(kept):>11} {str(evs['8']==evs['6']):>9} "
              f"{evs['6']['table_chunks']:>10} {str(evs['6']['tickers_present']):>14} {str(lost):>14}",
              flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)


main()