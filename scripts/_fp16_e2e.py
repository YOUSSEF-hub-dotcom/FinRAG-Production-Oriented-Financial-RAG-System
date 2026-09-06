"""Steps 9-11: E2E correctness + streaming + grounding with the FP16 pipeline.

Boots the REAL production pipeline with RERANKER_DTYPE=float16 (config-driven),
runs the 5 required queries (E1-E5, incl. same-session coreference + negative),
a streaming query, and validates sources/grounding.
"""
import asyncio, json, os, sys, time

# Config-driven FP16: must be set before config.settings is imported.
os.environ["RERANKER_DTYPE"] = "float16"
os.environ.setdefault("RERANKER_DEVICE", "cuda")

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from pipeline import FinancialRAGPipeline  # noqa: E402

OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/fp16_e2e.json"
SESSION = "ab2-fp16-e2e-session"

QUERIES = [
    ("E1", "What was Apple total annual revenue in FY2025?", ["AAPL"], "2025", None),
    ("E2", "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.",
     ["AAPL", "MSFT", "NVDA"], "2025", None),
    ("E3", "Compare the financial health of AAPL, MSFT, and NVDA",
     ["AAPL", "MSFT", "NVDA"], None, None),
    ("E4", "How much did the second company spend on Research and Development in that same fiscal year?",
     None, None, SESSION),
    ("E5", "What was Tesla total revenue in FY2025?", ["TSLA"], "2025", SESSION),
]

STREAM_Q = "What was Apple total annual revenue in FY2025?"
STREAM_EXPECT = ["416161"]


async def main():
    print("booting production pipeline (RERANKER_DTYPE=float16)...", flush=True)
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
          f"requested_dtype={rr._requested_dtype} effective_dtype={rr._effective_dtype}", flush=True)

    results = []
    for qid, q, tickers, fy, sid in QUERIES:
        t0 = time.time()
        try:
            result = pipe.query(q, fiscal_year=fy, session_id=sid, top_k=3, tickers=tickers)
            parsed = result.get("parsed")
            sources = getattr(parsed, "sources", None) or []
            raw = result.get("raw_output") or ""
            results.append({
                "query_id": qid, "query": q,
                "model_used": result.get("model_used"),
                "fallback_triggered": result.get("fallback_triggered"),
                "cache_hit": result.get("cache_hit"),
                "e2e_latency_ms": round((time.time() - t0) * 1000, 1),
                "rerank_ms_context": None,  # E2E includes Groq network; not used for config
                "answer": getattr(parsed, "answer", None) if parsed else None,
                "sources": sources,
                "raw_output": raw,
            })
            ans = getattr(parsed, "answer", "") if parsed else raw[:120]
            print(f"\n--- {qid}: {q[:45]}... ---", flush=True)
            print(f"    model={result.get('model_used')} fallback={result.get('fallback_triggered')} "
                  f"latency={(time.time()-t0):.1f}s", flush=True)
            print(f"    ANSWER: {ans[:200]}", flush=True)
            print(f"    SOURCES: {sources[:3]}", flush=True)
        except Exception as e:
            results.append({"query_id": qid, "query": q, "error": repr(e),
                            "e2e_latency_ms": round((time.time() - t0) * 1000, 1)})
            print(f"--- {qid} ERROR: {e}", flush=True)

    # Streaming regression (Step 10) -- FP16 must not change token streaming.
    print(f"\n--- STREAM: {STREAM_Q[:45]}... ---", flush=True)
    tokens, stream_err = [], None
    t0 = time.time()
    try:
        async for tok in pipe.query_stream(STREAM_Q, ticker="AAPL", fiscal_year="2025",
                                           session_id="ab2-fp16-stream"):
            tokens.append(tok)
    except Exception as e:
        stream_err = repr(e)
    joined = "".join(tokens)
    has_expected = any(exp in joined for exp in STREAM_EXPECT)
    print(f"    tokens={len(tokens)} chars={len(joined)} "
          f"contains_416161={has_expected} err={stream_err} "
          f"latency={(time.time()-t0):.1f}s", flush=True)
    results.append({
        "query_id": "STREAM", "query": STREAM_Q,
        "num_tokens": len(tokens), "stream_chars": len(joined),
        "contains_expected_figure": has_expected, "stream_error": stream_err,
        "sample": joined[:200],
    })

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)
    try:
        pipe.close()
    except Exception as e:
        print("close:", e, flush=True)


asyncio.run(main())