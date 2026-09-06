"""Experiment C: intelligent pre-filter (RRF-rank-based conservative trimming).

Simulates capping the candidate pool using the CHEAP RRF order (BM25+dense
fusion rank, already computed by retrieval) BEFORE the CrossEncoder, instead of
reducing HYBRID_TOP_K. The reranker still runs at production HYBRID_TOP_K=40
pool size semantics but sees fewer pairs.

Per-ticker caps N = 33/30/27  =>  ~99/90/81 combined for 3-ticker queries;
single-ticker queries use the same N on the 40 chunk list.

Compares vs the FP32-large baseline (full pool) saved from Exp A.
No production changes.
"""
import json, os, sys, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
os.environ.setdefault("RERANKER_DEVICE", "cuda")

import torch  # noqa
from sentence_transformers import CrossEncoder  # noqa

CAND = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/candidates.json"
BASELINE = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/expA_fp16.json"
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/expC_prefilter.json"

MODEL = "BAAI/bge-reranker-large"
DEVICE = "cuda"
BATCH = 32
CAPS = [33, 30, 27]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def top8(chunks, scores, n=8):
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunks[i] for i in idx]


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


def main():
    with open(CAND) as f:
        data = json.load(f)
    queries = {q["query_id"]: q for q in data["bench_queries"]}
    detail_ids = [qid for qid, q in queries.items() if q["count"] > 0]

    with open(BASELINE) as f:
        bl = json.load(f)
    baseline_top8 = bl["fp32"]["top8"]  # full-pool rerank (production)

    print(f"loading {MODEL} fp32 ...", flush=True)
    ce = CrossEncoder(MODEL, device=DEVICE)
    ce.predict([("warm", "x")], batch_size=1, show_progress_bar=False)
    sync()

    out = {"model": MODEL, "batch_size": BATCH, "caps_per_ticker": CAPS,
           "per_query": {}}

    for qid in detail_ids:
        q = queries[qid]
        rbt = q.get("raw_by_ticker") or {}

        # full baseline latency (one measured predict)
        base_pairs = [(q["query"], c.get("text", "")) for c in q["chunks"]]
        sync(); t0 = time.perf_counter()
        base_scores = list(ce.predict(base_pairs, batch_size=BATCH,
                                      show_progress_bar=False, convert_to_numpy=True))
        sync(); base_ms = (time.perf_counter() - t0) * 1000
        base_top = top8(q["chunks"], base_scores, 8)
        base_ids = [c.get("chunk_id") for c in base_top]
        base_ev = evidence(base_top)

        per_cap = {}
        for cap in CAPS:
            # per-ticker cap on RRF order
            capped = []
            for tk, chunks in rbt.items():
                capped.extend(chunks[:cap])
            if not capped:
                capped = q["chunks"][:cap]  # fallback (no raw_by_ticker)
            pairs = [(q["query"], c.get("text", "")) for c in capped]
            sync(); t0 = time.perf_counter()
            scores = list(ce.predict(pairs, batch_size=BATCH,
                                     show_progress_bar=False, convert_to_numpy=True))
            sync(); ms = (time.perf_counter() - t0) * 1000
            top = top8(capped, scores, 8)
            ids = [c.get("chunk_id") for c in top]
            ov = len(set(ids) & set(base_ids))
            ev = evidence(top)
            per_cap[str(cap)] = {
                "candidates_in": len(capped),
                "predict_ms": round(ms, 1),
                "latency_reduction_pct": round((1 - ms / base_ms) * 100, 1),
                "top8_overlap": ov,
                "top8_overlap_pct": round(ov / 8 * 100, 1),
                "dropped_from_baseline": [i for i in base_ids if i not in set(ids)],
                "evidence_identical": ev == base_ev,
                "tickers_present": ev["tickers_present"],
                "table_chunks": ev["table_chunks"],
            }
            print(f"  {qid} cap={cap} in={len(capped)} "
                  f"{ms:.0f}ms ({100*(1-ms/base_ms):.0f}% faster) "
                  f"overlap={ov}/8 ev_eq={ev == base_ev}", flush=True)

        out["per_query"][qid] = {
            "category": q["category"], "count": len(q["chunks"]),
            "baseline_predict_ms": round(base_ms, 1),
            "baseline_top8": base_ids,
            "baseline_evidence": base_ev,
            "caps": per_cap,
        }
        print(f"  {qid} baseline (full n={len(q['chunks'])}) = {base_ms:.0f}ms",
              flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)


main()