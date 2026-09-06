"""Experiment B (lean): bge-reranker-v2-m3 fp16 vs bge-reranker-large fp32 baseline.

Early measurement showed v2-m3 is ~90x slower on this 6GB GPU (120 pairs = 345s).
This lean run characterizes latency + top-8 overlap vs baseline on a subset of
queries with 1 warm + 1 measured predict (enough to reach a verdict).
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
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/expB_v2m3.json"

MODEL_V2 = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda"
BATCH = 16
QUERIES = ["Q1", "Q2", "Q6", "Q8"]  # single-factual, multi, financial-metric, table


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
    with open(BASELINE) as f:
        bl = json.load(f)
    baseline_top8 = bl["fp32"]["top8"]

    print(f"loading {MODEL_V2}...", flush=True)
    t0 = time.perf_counter()
    ce = CrossEncoder(MODEL_V2, device=DEVICE)
    load_s = time.perf_counter() - t0
    ce.model.half()
    sync()
    print(f"loaded in {load_s:.1f}s, fp16 dtype={next(ce.model.parameters()).dtype}",
          flush=True)
    ce.predict([("warm", "x")], batch_size=1, show_progress_bar=False)
    sync()

    out = {"model": MODEL_V2, "dtype": "fp16", "batch_size": BATCH,
           "load_time_s": round(load_s, 1), "per_query": {},
           "vram_reserved_mb": None}

    for qid in QUERIES:
        q = queries[qid]
        pairs = [(q["query"], c.get("text", "")) for c in q["chunks"]]
        # warm
        ce.predict(pairs, batch_size=BATCH, show_progress_bar=False, convert_to_numpy=True)
        sync()
        # measured (single)
        sync(); t0 = time.perf_counter()
        scores = list(ce.predict(pairs, batch_size=BATCH, show_progress_bar=False,
                                 convert_to_numpy=True))
        sync(); dt_ms = (time.perf_counter() - t0) * 1000

        top = top8(q["chunks"], scores, 8)
        ids = [c.get("chunk_id") for c in top]
        base_ids = baseline_top8[qid]
        ov = len(set(ids) & set(base_ids))
        ev = evidence(top)
        base_ev = bl["fp32"]["evidence"][qid]
        out["per_query"][qid] = {
            "category": q["category"], "count": len(q["chunks"]),
            "predict_ms": round(dt_ms, 1),
            "top8_overlap_vs_large": ov,
            "top8_overlap_pct": round(ov / 8 * 100, 1),
            "top8_v2m3_ids": ids,
            "evidence_v2m3": ev,
            "evidence_identical_to_large": ev == base_ev,
            "tickers": ev["tickers_present"], "n_tables": ev["table_chunks"],
        }
        print(f"  {qid} [{q['category']}] n={len(q['chunks'])} "
              f"predict={dt_ms:.0f}ms overlap={ov}/8 ev_identical={ev == base_ev}",
              flush=True)

    out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT} (vram={out['vram_reserved_mb']}MB)", flush=True)


main()