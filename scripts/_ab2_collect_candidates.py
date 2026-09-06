"""Collect real production candidate chunks per benchmark query and save to JSON.

MEASURE/DIAG ONLY. No production changes.

Boots the production pipeline once, runs hybrid retrieval for each benchmark
query, and captures the RAW candidate chunks that feed predict_scores:

  - single ticker: asearch -> raw RRF chunks (cap HYBRID_TOP_K=40)
  - multi ticker : per-ticker asearch flattened (the ~118-pair input)

Saves identical candidate lists to JSON so all reranker experiments test the
exact same input without re-running Qdrant (avoids lock contention).
"""
import asyncio, json, os, sys, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
os.environ.setdefault("RERANKER_DEVICE", "cuda")

from pipeline import FinancialRAGPipeline  # noqa

OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/candidates.json"

# (query_id, category, query, tickers or None, fiscal_year)
BENCH_QUERIES = [
    ("Q1", "single_factual", "What was Apple total annual revenue in FY2025?", ["AAPL"], "2025"),
    ("Q2", "multi_compare", "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025.", ["AAPL", "MSFT", "NVDA"], "2025"),
    ("Q3", "multi_health", "Compare the financial health of AAPL, MSFT, and NVDA.", ["AAPL", "MSFT", "NVDA"], None),
    ("Q4", "table_heavy", "What were Apple's net sales from iPhone products and Services in FY2025?", ["AAPL"], "2025"),
    ("Q5", "negative", "What was Tesla total revenue in FY2025?", ["TSLA"], "2025"),
    ("Q6", "financial_metric", "What was Apple's operating income in FY2025?", ["AAPL"], "2025"),
    ("Q7", "financial_metric", "What was Microsoft's net income in FY2025?", ["MSFT"], "2025"),
    ("Q8", "table_heavy", "What was Microsoft's largest business segment by revenue in FY2025?", ["MSFT"], "2025"),
]


def _strip_meta(c):
    # Keep only serializable, useful fields
    m = c.get("metadata") or {}
    return {
        "chunk_id": c.get("chunk_id") or c.get("id"),
        "text": c.get("text", ""),
        "ticker": m.get("ticker"),
        "fiscal_year": m.get("fiscal_year"),
        "contains_table": bool(m.get("contains_table")),
        "doc_type": m.get("doc_type"),
        "source": m.get("source"),
    }


def _flatten(raw_by_tk, tickers):
    flat = []
    for tk in tickers:
        for c in raw_by_tk.get(tk) or []:
            flat.append(_strip_meta(c))
    return flat


async def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs("/home/youssef/Financial_RAG/data/qdrant_db", exist_ok=True)

    print(f"boot pipeline ... {time.time():.1f}", flush=True)
    t0 = time.time()
    pipe = FinancialRAGPipeline(
        enable_hybrid_retrieval=True,
        enable_post_retrieval=True,
        enable_cache=False,
        enable_guardrail=False,
    )
    print(f"pipeline booted in {time.time()-t0:.1f}s", flush=True)

    results = {}
    per_query = []
    for qid, cat, q, tickers, fy in BENCH_QUERIES:
        t0 = time.time()
        entry = {"query_id": qid, "category": cat, "query": q, "tickers": tickers,
                 "fiscal_year": fy, "chunks": [], "raw_by_ticker": {}, "count": 0,
                 "error": None}
        try:
            if len(tickers) == 1:
                tk = tickers[0]
                flt = {"ticker": tk}
                if fy:
                    flt["fiscal_year"] = fy
                raw = await pipe._hybrid_search.asearch([q], flt)
                chunks = [_strip_meta(c) for c in raw]
                entry["count"] = len(chunks)
                entry["chunks"] = chunks
                entry["raw_by_ticker"] = {tk: chunks}
            else:
                # replicate _batch_multi_ticker_retrieval_core candidate collection
                tks = list(tickers)[:4]
                raw_by_tk = {}
                for tk in tks:
                    flt = {"ticker": tk}
                    if fy:
                        flt["fiscal_year"] = fy
                    try:
                        r = await pipe._hybrid_search.asearch([q], flt)
                        raw_by_tk[tk] = [_strip_meta(c) for c in r]
                    except Exception as e:
                        raw_by_tk[tk] = []
                entry["raw_by_ticker"] = raw_by_tk
                entry["chunks"] = _flatten(raw_by_tk, tks)
                entry["count"] = len(entry["chunks"])
        except Exception as e:
            entry["error"] = repr(e)
        entry["collect_ms"] = (time.time() - t0) * 1000
        per_query.append(entry)
        results[qid] = entry
        print(f"  {qid} [{cat}] {q[:40]}... -> {entry['count']} cands ({entry['collect_ms']:.0f}ms) err={entry.get('error')}", flush=True)

        # ticker breakdown
        from collections import Counter
        tc = Counter(x.get("ticker") for x in entry["chunks"])
        print(f"       tickers: {dict(tc)}", flush=True)

    payload = {"collected_at": time.time(), "bench_queries": per_query}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)

    # IMPORTANT: explicit cleanup
    try:
        pipe.close()
    except Exception as e:
        print("close err:", e, flush=True)


asyncio.run(main())
