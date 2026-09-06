"""Experiment A (consolidated): FP32 (production) vs FP16 (experimental)
bge-reranker-large on PRE-COLLECTED identical candidates.

Phases:
  1. FP32: per-query latency + scores + top8 + evidence, then size sweep.
  2. FP16: same (separate experimental model instance).
  3. Comparison + report.

No production changes. VRAM-safe (only one model in memory at a time).
"""
import gc, json, os, statistics, sys, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
os.environ.setdefault("RERANKER_DEVICE", "cuda")

import torch  # noqa
from sentence_transformers import CrossEncoder  # noqa

CAND = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/candidates.json"
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/expA_fp16.json"
MODEL = "BAAI/bge-reranker-large"
DEVICE = "cuda"
BATCH = 32
ITERS = 4  # 1 warm + 3 measured


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, iters=ITERS):
    times = []
    for i in range(iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); times.append((time.perf_counter() - t0) * 1000)
    warm = times[1:]
    return {"mean": statistics.mean(warm), "median": statistics.median(warm),
            "min": min(warm), "max": max(warm)}


def make_pred(ce, query, chunks):
    pairs = [(query, c.get("text", "")) for c in chunks]
    def f():
        return ce.predict(pairs, batch_size=BATCH, show_progress_bar=False,
                          convert_to_numpy=True)
    return f


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


def run_all(ce, queries, detail_ids, flat_q2, q1, q2_query, tag):
    print(f"=== {tag} ===", flush=True)
    out = {"latency": {}, "top8": {}, "evidence": {}, "scores": {}}
    for qid in detail_ids:
        q = queries[qid]
        fn = make_pred(ce, q["query"], q["chunks"])
        lat = bench(fn, ITERS)
        scores = list(fn())
        top = top8(q["chunks"], scores, 8)
        out["latency"][qid] = {k: round(v, 1) for k, v in lat.items()}
        out["top8"][qid] = [c.get("chunk_id") for c in top]
        out["evidence"][qid] = evidence(top)
        out["scores"][qid] = [float(x) for x in scores]
        print(f"  {qid} [{q['category']}] n={len(q['chunks'])} "
              f"mean={lat['mean']:.0f}ms", flush=True)

    # size sweep on Q2 flattened candidates (up to len(flat))
    out["size_sweep"] = {}
    for n in (24, 40, 60, 80, 90, 100, 120):
        if n > len(flat_q2):
            continue
        sub = flat_q2[:n]
        sub_scores = {c.get("chunk_id"): None for c in sub}
        fn = make_pred(ce, q2_query, sub)
        lat = bench(fn, ITERS)
        scores = list(fn())
        top = top8(sub, scores, 8)
        out["size_sweep"][str(n)] = {
            "latency_ms": {k: round(v, 1) for k, v in lat.items()},
            "top8_chunk_ids": [c.get("chunk_id") for c in top],
            "tickers": sorted(set(c.get("ticker") for c in top)),
            "n_tables": len([c for c in top if c.get("contains_table")]),
        }
        print(f"  sweep n={n}: mean={lat['mean']:.0f}ms", flush=True)

    # single-ticker 40 explicit
    fn = make_pred(ce, q1["query"], q1["chunks"])
    lat = bench(fn, ITERS)
    out["single_40_latency"] = {k: round(v, 1) for k, v in lat.items()}
    out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
    print(f"  single-40: mean={lat['mean']:.0f}ms  vram={out['vram_reserved_mb']}MB",
         flush=True)
    return out


def main():
    with open(CAND) as f:
        data = json.load(f)
    queries = {q["query_id"]: q for q in data["bench_queries"]}
    detail_ids = [qid for qid, q in queries.items() if q["count"] > 0]
    q1 = queries["Q1"]
    q2 = queries["Q2"]
    flat_q2 = q2["chunks"]

    # warm + fp32
    ce32 = CrossEncoder(MODEL, device=DEVICE)
    ce32.predict([("warm up", "placeholder")], batch_size=1, show_progress_bar=False)
    sync()
    fp32 = run_all(ce32, queries, detail_ids, flat_q2, q1,
                   q2["query"], "FP32 (production)")
    del ce32; gc.collect(); torch.cuda.empty_cache(); sync()

    # fp16 (separate experimental instance)
    ce16 = CrossEncoder(MODEL, device=DEVICE)
    ce16.model.half()
    sync()
    print("fp16 dtype:", next(ce16.model.parameters()).dtype, flush=True)
    ce16.predict([("warm up", "placeholder")], batch_size=1, show_progress_bar=False)
    sync()
    fp16 = run_all(ce16, queries, detail_ids, flat_q2, q1,
                   q2["query"], "FP16 (experimental)")
    del ce16; gc.collect(); torch.cuda.empty_cache(); sync()

    # ---- comparison ----
    print("\n=== COMPARISON FP32 vs FP16 ===", flush=True)
    per_query = []
    for qid in detail_ids:
        ids32 = fp32["top8"][qid]
        ids16 = fp16["top8"][qid]
        ov = len(set(ids32) & set(ids16))
        l32 = fp32["latency"][qid]["mean"]
        l16 = fp16["latency"][qid]["mean"]
        sp = (1 - l16 / l32) * 100 if l32 else None
        per_query.append({
            "query_id": qid,
            "category": queries[qid]["category"],
            "count": len(queries[qid]["chunks"]),
            "fp32_latency_mean_ms": l32, "fp16_latency_mean_ms": l16,
            "speedup_pct": round(sp, 1) if sp is not None else None,
            "top8_overlap": ov, "top8_overlap_pct": round(ov / 8 * 100, 1),
            "evidence_identical": fp32["evidence"][qid] == fp16["evidence"][qid],
        })
        print(f"  {qid}: fp32={l32:.0f}ms fp16={l16:.0f}ms "
              f"speedup={sp:.1f}% overlap={ov}/8 "
              f"ev_identical={per_query[-1]['evidence_identical']}", flush=True)

    # size sweep comparison
    sweep_comp = {}
    for n, v in fp32["size_sweep"].items():
        if n in fp16["size_sweep"]:
            a = v["latency_ms"]["mean"]; b = fp16["size_sweep"][n]["latency_ms"]["mean"]
            ov_ids = set(v["top8_chunk_ids"]) & set(fp16["size_sweep"][n]["top8_chunk_ids"])
            sweep_comp[n] = {"fp32_ms": a, "fp16_ms": b,
                             "speedup_pct": round((1 - b / a) * 100, 1),
                             "top8_overlap": len(ov_ids)}
            print(f"  sweep n={n}: fp32={a:.0f}ms fp16={b:.0f}ms "
                  f"speedup={(1-b/a)*100:.1f}% overlap={len(ov_ids)}/8", flush=True)

    report = {
        "model": MODEL, "device": DEVICE, "batch_size": BATCH,
        "iters": ITERS, "candidates_source": CAND,
        "fp32": fp32, "fp16": fp16,
        "comparison": {"per_query": per_query, "size_sweep": sweep_comp},
        "vram_reserved_mb": {"fp32": fp32["vram_reserved_mb"],
                             "fp16": fp16["vram_reserved_mb"]},
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)


main()