"""Step 7+8: FP32 (GOLD baseline) vs FP16 regression with real production candidates.

- Same query, same candidates, same model, same device (cuda), same batch_size, same top_n.
- Only dtype changes.
- Size sweep: 24 / 40 / 60 / 120 candidates.
- Metrics: mean/median/p95/min/max, torch.cuda.synchronize() around inference,
  GPU util + VRAM sampling, plus ranking/evidence regressions (exact top-8 overlap,
  Jaccard, ticker coverage, table coverage, critical evidence retention).
"""
import json, os, subprocess, sys, threading, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
sys.path.insert(0, "/home/youssef/Financial_RAG/src/4_retrieval")
os.environ.setdefault("RERANKER_DEVICE", "cuda")

import numpy as np          # noqa: E402
import torch                # noqa: E402
from reranker import CrossEncoderReranker  # noqa: E402

import statistics as st     # noqa: E402

ART = "artifacts/ab2_benchmark/candidates.json"
OUT = "artifacts/ab2_benchmark/fp16_impl_regression.json"
BATCH = 32
TOP_N = 8
N_ITER = 5
SIZE_SWEEP = [24, 40, 60, 120]

# Critical evidence markers expected to survive into top-8 (validated vs docs).
EVIDENCE_MARKERS = {
    "Q1": ["416161"],  # AAPL revenue
    "Q2": ["416161", "133050", "281724", "128528", "130497", "81453"],
    "Q3": ["416161", "112010", "281724", "101832", "130497"],  # + margins in ev text
    "Q4": ["209586", "109158"],  # Apple product/services, table-heavy
    "Q5": [],
    "Q6": ["133050"],  # AAPL op income (margin query)
    "Q7": ["128528"],  # MSFT op income
    "Q8": ["281724"],  # MSFT revenue, table-heavy
}
REQ_TICKERS = {
    "Q1": ["AAPL"], "Q2": ["AAPL", "MSFT", "NVDA"], "Q3": ["AAPL", "MSFT", "NVDA"],
    "Q4": ["AAPL"], "Q5": [], "Q6": ["AAPL"], "Q7": ["MSFT"], "Q8": ["MSFT"],
}


def gpu_sample(events, stop):
    """Sample GPU util + mem every 100ms into events list until stop()."""
    while not stop():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip().split(",")
            if len(out) >= 2:
                events.append((int(out[0].strip()), int(out[1].strip())))
        except Exception:
            pass
        time.sleep(0.1)


def timed_predict(rr, query, chunks, n):
    """Time rr.predict_scores n times with cuda sync; return list of ms."""
    lat = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rr.predict_scores(query, chunks)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def summarize(lat):
    return {
        "mean_ms": round(float(np.mean(lat)), 1),
        "median_ms": round(float(np.median(lat)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "min_ms": round(float(np.min(lat)), 1),
        "max_ms": round(float(np.max(lat)), 1),
        "raw_ms": [round(float(x), 1) for x in lat],
    }


def top8_ids(rr, query, chunks):
    scores = rr.predict_scores(query, chunks)
    top = rr.rerank_with_scores(chunks, scores, top_n=TOP_N)
    return [c["chunk_id"] for c in top], top


def evidence_retained(top, qid):
    text = " ".join(c.get("text", "") for c in top)
    tbl = text.count("%%TABLE_")
    return {
        "markers_missing": [m for m in EVIDENCE_MARKERS.get(qid, []) if m not in text],
        "table_markers": tbl,
    }


def ticker_coverage(top, qid):
    cov = {c.get("ticker") for c in top}
    return cov, set(REQ_TICKERS.get(qid, [])) <= cov


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return round(len(sa & sb) / len(sa | sb), 4) if (sa | sb) else 1.0


def exact_overlap(a, b):
    return round(len(set(a) & set(b)) / max(1, len(set(b))), 4)


def main():
    data = json.load(open(ART))
    queries = {q["query_id"]: q for q in data["bench_queries"]}
    qids = ["Q1", "Q2", "Q3", "Q4", "Q6", "Q7", "Q8"]  # Q5 is negative (0 candidates)

    rr32 = CrossEncoderReranker(top_n=TOP_N, batch_size=BATCH, dtype="float32", device="cuda")
    rr16 = CrossEncoderReranker(top_n=TOP_N, batch_size=BATCH, dtype="float16", device="cuda")
    rr32.warm_up()
    rr16.warm_up()
    print("FP32 dtype:", rr32._effective_dtype, "| FP16 dtype:", rr16._effective_dtype, flush=True)
    print("FP32 param:", next(rr32._model.parameters()).dtype,
          "| FP16 param:", next(rr16._model.parameters()).dtype, flush=True)

    result = {
        "config": {"batch_size": BATCH, "top_n": TOP_N, "n_iter": N_ITER,
                   "model": "BAAI/bge-reranker-large"},
        "per_query": {},
        "size_sweep": {},
        "gpu": {},
    }

    # Per-query: FP32 vs FP16 scores, top-8 ids, evidence.
    for qid in qids:
        q = queries[qid]
        chunks = q["chunks"]
        n = len(chunks)
        # discard any duplicates text? keep as-is (production input)
        t32 = timed_predict(rr32, q["query"], chunks, N_ITER)
        t16 = timed_predict(rr16, q["query"], chunks, N_ITER)
        ids32, top32 = top8_ids(rr32, q["query"], chunks)
        ids16, top16 = top8_ids(rr16, q["query"], chunks)
        ev32 = evidence_retained(top32, qid)
        ev16 = evidence_retained(top16, qid)
        cov32, ok32 = ticker_coverage(top32, qid)
        cov16, ok16 = ticker_coverage(top16, qid)
        stable_order = ids32 == ids16
        result["per_query"][qid] = {
            "n_candidates": n,
            "fp32": summarize(t32),
            "fp16": summarize(t16),
            "speedup_pct_mean": round(100 * (np.mean(t32) - np.mean(t16)) / np.mean(t32), 1),
            "exact_top8_overlap": exact_overlap(ids32, ids16),
            "jaccard_top8": jaccard(ids32, ids16),
            "top8_identical": ids32 == ids16,
            "top8_order_identical": stable_order,
            "fp32_top8_ids": ids32,
            "fp16_top8_ids": ids16,
            "fp32_ticker_coverage": sorted(cov32),
            "fp16_ticker_coverage": sorted(cov16),
            "ticker_coverage_kept": ok32 and ok16,
            "evidence_missing_fp32": ev32["markers_missing"],
            "evidence_missing_fp16": ev16["markers_missing"],
            "table_markers_fp32": ev32["table_markers"],
            "table_markers_fp16": ev16["table_markers"],
        }
        print(f"\n{qid} n={n}: fp32 mean {np.mean(t32):.0f}ms fp16 mean {np.mean(t16):.0f}ms "
              f"overlap={exact_overlap(ids32, ids16)} jac={jaccard(ids32, ids16)} "
              f"order_same={stable_order} ev_missing32={ev32['markers_missing']} "
              f"ev_missing16={ev16['markers_missing']}", flush=True)

    # Size sweep on Q2 candidates (multi-ticker, 120 pool).
    pool = queries["Q2"]["chunks"]
    q2 = queries["Q2"]["query"]
    print("\nSize sweep (Q2 pool):", flush=True)
    for n in SIZE_SWEEP:
        chunks = pool[:n]
        t32 = timed_predict(rr32, q2, chunks, N_ITER)
        t16 = timed_predict(rr16, q2, chunks, N_ITER)
        ids32, _ = top8_ids(rr32, q2, chunks)
        ids16, _ = top8_ids(rr16, q2, chunks)
        result["size_sweep"][str(n)] = {
            "fp32": summarize(t32), "fp16": summarize(t16),
            "speedup_pct_mean": round(100 * (np.mean(t32) - np.mean(t16)) / np.mean(t32), 1),
            "exact_top8_overlap": exact_overlap(ids32, ids16),
            "jaccard_top8": jaccard(ids32, ids16),
        }
        print(f"  n={n}: fp32 {np.mean(t32):.0f}ms fp16 {np.mean(t16):.0f}ms "
              f"speedup={100*(np.mean(t32)-np.mean(t16))/np.mean(t32):.1f}% "
              f"overlap={exact_overlap(ids32, ids16)}", flush=True)

    # GPU util + VRAM sampling during a heavy fp16 workload and fp32 workload.
    gpu_events_32, gpu_events_16 = [], []
    stop = {"flag": False}
    chunks120 = queries["Q2"]["chunks"]
    for label, rr, events in (("fp32", rr32, gpu_events_32), ("fp16", rr16, gpu_events_16)):
        events.clear()
        t = threading.Thread(target=gpu_sample, args=(events, lambda: stop["flag"]))
        stop["flag"] = False
        t.start()
        for _ in range(10):
            torch.cuda.synchronize()
            rr.predict_scores(q2, chunks120)
        torch.cuda.synchronize()
        stop["flag"] = True
        t.join()
        util = [e[0] for e in events]
        mem = [e[1] for e in events]
        gpu = {
            "gpu_util_mean": round(st.mean(util), 1) if util else None,
            "gpu_util_max": max(util) if util else None,
            "vram_mb_mean": round(st.mean(mem), 1) if mem else None,
            "vram_mb_peak": max(mem) if mem else None,
            "torch_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }
        result["gpu"][label] = gpu
        print(f"\nGPU[{label}]: util {gpu['gpu_util_mean']}% peak {gpu['gpu_util_max']}% "
              f"vram {gpu['vram_mb_mean']}MB peak {gpu['vram_mb_peak']}MB "
              f"torch_alloc {gpu['torch_allocated_mb']}MB", flush=True)

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved", OUT, flush=True)


if __name__ == "__main__":
    main()