#!/usr/bin/env python3
"""Lean forensic reranker benchmark: boots production pipeline once, fetches
real AAPL/MSFT/NVDA chunks, runs baseline + batch-size + candidate-count sweeps
+ stage breakdown + evidence retention. No production changes."""
import asyncio, os, statistics, sys, time
sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
sys.path.insert(0, "/home/youssef/Financial_RAG/src/4_retrieval")
import torch  # noqa
from pipeline import FinancialRAGPipeline  # noqa

OUT = "/home/youssef/Financial_RAG/scripts/_bench_results.txt"

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench(fn, iters=5, *a, **k):
    times = []
    for i in range(iters):
        sync(); t0 = time.time()
        r = fn(*a, **k)
        sync(); times.append((time.time() - t0) * 1000)
    body = times[1:]
    return r, body

def fmt(t):
    return "mean=%.1f med=%.1f min=%.1f max=%.1f" % (
        statistics.mean(t), statistics.median(t), min(t), max(t))

def ev_in(chunks):
    j = "\n".join((c.get("text") or "") for c in chunks)
    return "".join([
        "A:%s" % ("$416,161" in j),
        "Ao:%s" % ("$133,050" in j),
        "M:%s" % ("$281,724" in j),
        "N:%s" % ("$130,497" in j),
        "tbl:%s" % any((c.get("metadata") or {}).get("contains_table") for c in chunks),
    ])

def tbalance(chunks):
    from collections import Counter
    return dict(Counter((c.get("metadata") or {}).get("ticker") for c in chunks))

async def main():
    log = open(OUT, "w")
    def w(s): log.write(s + "\n"); print(s); log.flush()

    w("boot pipeline ... %.1f" % time.time())
    pipe = FinancialRAGPipeline(enable_hybrid_retrieval=True, enable_post_retrieval=True,
                                enable_cache=False, enable_guardrail=False)
    rr = pipe._post_retrieval._reranker
    rr._lazy_init()
    w("post_retrieval table_shield clean_enabled=%s" % pipe._post_retrieval._table_shield._clean_enabled)
    w("reranker device=%s batch=%d top_n=%d" % (rr._device, rr._batch_size, rr._top_n))

    q = "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025"
    w("fetch real multi-ticker chunks ... %.1f" % time.time())
    t0 = time.time()
    raw = await pipe._balanced_ticker_subretrievals(q, ["AAPL", "MSFT", "NVDA"], fiscal_year="2025", top_k=3)
    w("fetched %d chunks in %.1fs, tickers=%s" % (len(raw), time.time()-t0, tbalance(raw)))

    # ---- PART 3 baseline ----
    w("\n[PART3] BASELINE real %d pairs batch=%d dummy_warm" % (len(raw), rr._batch_size))
    rr.predict_scores(q, raw)
    _, t3 = bench(rr.predict_scores, 6, q, raw)
    w("TYPE pred_scores 24 " + fmt(t3))
    base_top = rr.rerank(q, raw, top_n=8)
    base_ids = [c.get("chunk_id") for c in base_top]
    w("baseline top8 evidence=" + ev_in(base_top))
    w("baseline top8 ticker balance=" + str(tbalance(base_top)))

    # ---- PART 4 batch size sweep ----
    w("\n[PART4] BATCH SIZE SWEEP same %d pairs" % len(raw))
    pairs = [(q, c.get("text","")) for c in raw]
    for bs in (8, 16, 32, 64):
        rr._model.predict(pairs, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)
        sync()
        _, tb = bench(lambda: rr._model.predict(pairs, batch_size=bs, show_progress_bar=False, convert_to_numpy=True), 5)
        mem = torch.cuda.memory_reserved()/1048576
        w("BATCH %d  %s  mem_resv=%.0fMB" % (bs, fmt(tb), mem))

    # ---- PART 5 candidate count sweep ----
    w("\n[PART5] CANDIDATE COUNT SWEEP (production batch=32, real chunks)")
    for n in (12, 16, 20, 24):
        sub = raw[:n]
        rr._model.predict([(q, c.get("text","")) for c in sub], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        sync()
        _, tc = bench(lambda: rr._model.predict([(q, c.get("text","")) for c in sub], batch_size=32, show_progress_bar=False, convert_to_numpy=True), 5)
        sc = list(rr.predict_scores(q, sub))
        top = rr.rerank_with_scores(sub, sc, top_n=8)
        w("CAND %d  %s  ev_top8=%s" % (n, fmt(tc), ev_in(top)))

    # ---- PART 6/7 ranking stability ----
    w("\n[PART6/7] RANKING STABILITY top8 vs baseline (aligned by real chunk order)")
    fs = list(rr.predict_scores(q, raw))
    base_final = rr.rerank_with_scores(raw, fs, top_n=8)
    w("baseline final top8:")
    for i,c in enumerate(base_final):
        w("  %d. %s [%s] score=%.4f" % (i+1, c.get("chunk_id"), (c.get("metadata") or {}).get("ticker"), c.get("rerank_score",0)))
    for n in (12,16,20):
        sub = raw[:n]
        s2 = list(rr.predict_scores(q, sub))
        top = rr.rerank_with_scores(sub, s2, top_n=8)
        sel = [c.get("chunk_id") for c in top]
        inter = len(set(sel) & set(base_ids))
        w("cand=%d overlap_top8_with_baseline=%d/8 dropped=%s" % (n, inter, [i for i in base_ids if i not in set(sel)]))

    # ---- PART 9 stage breakdown ----
    w("\n[PART9] STAGE BREAKDOWN on real top-8")
    _, tp = bench(rr.predict_scores, 3, q, raw)
    w("CrossEncoder.predict(24)  " + fmt(tp))
    top8 = rr.rerank(q, raw, top_n=8)
    ts = pipe._post_retrieval._table_shield
    from cylinder_reorder import cylinder_reorder
    t0 = time.time(); shielded = await ts.shield(top8); dt_sh = (time.time()-t0)*1000
    t0 = time.time(); cylinder_reorder(shielded, None); dt_cyl = (time.time()-t0)*1000
    t0 = time.time(); await pipe._post_retrieval.aprocess_scored(q, raw, list(fs)); dt_post = (time.time()-t0)*1000
    w("TableShield.shield(8)     %.1f ms" % dt_sh)
    w("Cylinder reorder(8)       %.1f ms" % dt_cyl)
    w("aprocess_scored post(no pred, incl rescue+shield+cylinder)  %.1f ms" % dt_post)
    w("TableShield clean_enabled=%s  (False => passthrough, no LLM)" % ts._clean_enabled)
    w("PREDICT_ONLY(sum)=%.1f  POSTONLY(no predict)=%.1f" % (statistics.mean(tp), dt_post))
    log.close()

asyncio.run(main())
