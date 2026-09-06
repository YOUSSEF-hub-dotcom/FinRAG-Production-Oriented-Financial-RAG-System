"""Pipeline-level context equivalence: FP16 vs FP32 through the REAL pipeline.

Runs the production query() path up to the generation boundary (captures the
exact `retrieved_docs` the LLM would receive) for the 5 required E2E queries,
with a stubbed generator so no Groq call is needed. Compares FP16 context vs the
FP32 GOLD run: chunk IDs, ticker coverage, table markers, metadata integrity.
"""
import asyncio, json, os, sys

os.environ.setdefault("RERANKER_DEVICE", "cuda")
dtype = sys.argv[1]
os.environ["RERANKER_DTYPE"] = dtype

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from pipeline import FinancialRAGPipeline  # noqa: E402

OUT = f"/home/youssef/Financial_RAG/artifacts/ab2_benchmark/pipeline_ctx_{dtype}.json"

QUERIES = [
    ("E1", "What was Apple total annual revenue in FY2025?", ["AAPL"], "2025"),
    ("E2", "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.",
     ["AAPL", "MSFT", "NVDA"], "2025"),
    ("E3", "Compare the financial health of AAPL, MSFT, and NVDA", ["AAPL", "MSFT", "NVDA"], None),
    ("E5", "What was Tesla total revenue in FY2025?", ["TSLA"], "2025"),
]

captured = {}


def make_generator_stub(pipe):
    """Wrap pipe._generator.generate to capture the retrieved docs."""
    real = pipe._generator.generate

    def stub(query, retrieved_docs, **kw):
        captured["docs"] = retrieved_docs
        return {
            "raw_output": "", "parsed": None, "model_used": "stub",
            "fallback_triggered": False, "ttft_ms": 0,
        }
    return stub


async def main():
    pipe = FinancialRAGPipeline(
        enable_hybrid_retrieval=True, enable_post_retrieval=True,
        enable_cache=False, enable_guardrail=False,
    )
    rr = pipe._post_retrieval._reranker
    pipe._generator.generate = make_generator_stub(pipe)
    print(f"dtype={dtype} reranker requested={rr._requested_dtype} "
          f"device={rr._device}", flush=True)

    results = []
    for qid, q, tickers, fy in QUERIES:
        captured.clear()
        try:
            pipe.query(q, fiscal_year=fy, session_id="ab2-ctx", top_k=3, tickers=tickers)
            docs = captured.get("docs", [])
        except Exception as e:
            print(f"{qid} ERROR: {e!r}", flush=True)
            results.append({"query_id": qid, "error": repr(e), "n_docs": 0})
            continue
        texts = [d.get("text", "") for d in docs]
        all_text = " ".join(texts)
        results.append({
            "query_id": qid,
            "n_docs": len(docs),
            "reranked_top8_ids": [d.get("chunk_id") for d in docs][:24],  # multi-ticker may exceed 8
            "tickers": sorted({d.get("ticker") for d in docs}),
            "fiscal_years": sorted({d.get("fiscal_year") for d in docs}),
            "table_markers": all_text.count("%%TABLE_"),
            "table_chunks": sum(1 for d in docs if str(d.get("contains_table")).lower() == "true"),
            "item8_mentions": sum(1 for tt in texts if "Item 8" in tt or "ITEM 8" in tt),
            "rerank_scores_sorted": all(
                docs[i].get("rerank_score", 0) >= docs[i + 1].get("rerank_score", 0)
                for i in range(len(docs) - 1)
            ),
        })
        print(f"{qid}: n_docs={results[-1]['n_docs']} tickers={results[-1]['tickers']} "
              f"tables={results[-1]['table_chunks']} item8={results[-1]['item8_mentions']} "
              f"sorted={results[-1]['rerank_scores_sorted']}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"dtype": dtype, "queries": results}, f, indent=2)
    print("saved", OUT, flush=True)
    try:
        pipe.close()
    except Exception as e:
        print("close:", e, flush=True)


asyncio.run(main())