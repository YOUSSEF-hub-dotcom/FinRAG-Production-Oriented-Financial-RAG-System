"""E2E correctness benchmark: the 5 required queries on the PRODUCTION pipeline.

Production FP32 reranker, enable_hybrid + enable_post_retrieval.
Same pipeline instance across queries => generator memory persists, which is
required for the Q4 coreference follow-up ("the second company").

MEMORY NOTE: E2E latency is not used for config comparisons (Groq network +
possible fallback make it noisy). Only correctness is validated, then inherited
by FP16 from its identical top-8.
"""
import asyncio, json, os, sys, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
os.environ.setdefault("RERANKER_DEVICE", "cuda")

from pipeline import FinancialRAGPipeline  # noqa

OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/e2e_correctness.json"

E2E = [
    # (id, query, tickers, fiscal_year, validation-note)
    ("E1", "What was Apple total annual revenue in FY2025?", ["AAPL"], "2025",
     "expect $416,161M"),
    ("E2", "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.",
     ["AAPL", "MSFT", "NVDA"], "2025",
     "expect AAPL $416,161M / MSFT $281,724M / NVDA $130,497M"),
    ("E3", "Compare the financial health of AAPL, MSFT, and NVDA",
     ["AAPL", "MSFT", "NVDA"], None,
     "expect all three tickers represented"),
    ("E4", "How much did the second company spend on Research and Development in that same fiscal year?",
     None, None,
     "coreference follow-up; expect MSFT R&D ~$32,488M"),
    ("E5", "What was Tesla total revenue in FY2025?", ["TSLA"], "2025",
     "expect unavailable/refusal"),
]

SESSION = "ab2-e2e-correctness-session"


async def main():
    print("boot production pipeline (FP32 reranker)...", flush=True)
    t0 = time.time()
    pipe = FinancialRAGPipeline(
        enable_hybrid_retrieval=True,
        enable_post_retrieval=True,
        enable_cache=False,
        enable_guardrail=False,
    )
    print(f"pipeline booted in {time.time()-t0:.1f}s", flush=True)
    rr = pipe._post_retrieval._reranker
    print(f"reranker: model={rr._model_name} device={rr._device} "
          f"top_n={rr._top_n} batch={rr._batch_size}", flush=True)

    results = []
    for qid, q, tickers, fy, note in E2E:
        t0 = time.time()
        try:
            result = pipe.query(
                q,
                fiscal_year=fy,
                session_id=SESSION,
                top_k=3,
                tickers=tickers,
            )
            raw = result.get("raw_output") or ""
            model_used = result.get("model_used")
            fallback = result.get("fallback_triggered")
            cache_hit = result.get("cache_hit")
            results.append({
                "query_id": qid, "query": q, "note": note,
                "model_used": model_used, "fallback_triggered": fallback,
                "cache_hit": cache_hit,
                "e2e_latency_ms": round((time.time() - t0) * 1000, 1),
                "raw_output": raw,
            })
            print(f"\n--- {qid}: {q[:50]}... [{note}] ---", flush=True)
            print(f"    model={model_used} fallback={fallback} "
                  f"cache={cache_hit} latency={(time.time()-t0):.1f}s", flush=True)
            print(f"    ANSWER: {raw[:300]}", flush=True)
        except Exception as e:
            results.append({"query_id": qid, "query": q, "error": repr(e),
                            "e2e_latency_ms": round((time.time() - t0) * 1000, 1)})
            print(f"--- {qid} ERROR: {e}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)
    try:
        pipe.close()
    except Exception as e:
        print("close:", e, flush=True)


asyncio.run(main())