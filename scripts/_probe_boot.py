#!/usr/bin/env python3
import asyncio, sys, time
sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
sys.path.insert(0, "/home/youssef/Financial_RAG/src/4_retrieval")
from pipeline import FinancialRAGPipeline

async def main():
    log = open("/home/youssef/Financial_RAG/scripts/_bench_progress.txt", "a")
    def w(s):
        log.write(s + "\n"); log.flush()
    w("boot pipeline start %.1f" % time.time())
    pipe = FinancialRAGPipeline(
        enable_hybrid_retrieval=True, enable_post_retrieval=True,
        enable_cache=False, enable_guardrail=False)
    w("boot pipeline done %.1f" % time.time())
    r = pipe._post_retrieval._reranker
    w("reranker _device=%s initialised=%s" % (r._device, r._initialised))
    r._lazy_init()
    w("reranker lazy_init done %.1f" % time.time())
    query = "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025"
    t0 = time.time()
    raw = await pipe._balanced_ticker_subretrievals(query, ["AAPL","MSFT","NVDA"], fiscal_year="2025", top_k=3)
    w("retrieval done in %.1f s, chunks=%d" % (time.time()-t0, len(raw)))
    for c in raw[:30]:
        w("  chunk %s ticker=%s t=%s" % (c.get("chunk_id"), (c.get("metadata") or {}).get("ticker"), (c.get("text") or "")[:60]))
    log.close()

asyncio.run(main())
