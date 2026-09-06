"""Reproduce the NVDA FY2026 chat query through query_stream() with stage capture."""
import asyncio
import sys
import logging

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)

from model_loader import get_or_create_pipeline

pipe = get_or_create_pipeline()

QUERY = (
    "According to the newly uploaded supplemental filing, what was NVIDIA's "
    "Data Center revenue in FY2026, and what was the year-over-year growth percentage?"
)

def show_doc(d, label=""):
    meta = d.get("metadata") or {}
    txt = d.get("text") or ""
    print(f"  [{label}] chunk_id={d.get('chunk_id')} ticker={meta.get('ticker')} "
          f"fy={meta.get('fiscal_year')} section={meta.get('section')} "
          f"src={str(meta.get('source_file'))[-40:]}")
    print(f"      text_len={len(txt)} text_head={txt[:160]!r}")

async def main():
    print("=== _query_fiscal_year(QUERY) [sync path] ===", repr(pipe._query_fiscal_year(QUERY)))

    # ---- wrap hybrid search ----
    orig_asearch = pipe._hybrid_search.asearch
    async def wrapped_asearch(queries, metadata_filter=None):
        docs = await orig_asearch(queries, metadata_filter)
        print("\n=== STAGE A: HYBRID SEARCH OUTPUT (%d docs) ===" % len(docs))
        new_id = "NVDA_txt_4e2da079297d_0000"
        for d in docs:
            new_flag = "  <<<< NEW CHUNK" if d.get("chunk_id") == new_id else ""
            meta = d.get("metadata") or {}
            print(f"  {d.get('chunk_id')} ticker={meta.get('ticker')} fy={meta.get('fiscal_year')} "
                  f"src={str(meta.get('source_file'))[-30:]} rrf={d.get('rrf_score')} "
                  f"dense={d.get('dense_score')} sparse={d.get('sparse_score')} rescued={d.get('rescued_table')}{new_flag}")
        return docs
    pipe._hybrid_search.asearch = wrapped_asearch

    # ---- wrap post-retrieval ----
    orig_aprocess = pipe._post_retrieval.aprocess
    async def wrapped_aprocess(query, chunks):
        rk = await orig_aprocess(query, chunks)
        print("\n=== STAGE B: POST-RETRIEVAL (rerank->top8->shield->cylinder) ===")
        new_id = "NVDA_txt_4e2da079297d_0000"
        for d in rk:
            new_flag = "  <<<< NEW CHUNK" if d.get("chunk_id") == new_id else ""
            meta = d.get("metadata") or {}
            print(f"  out={d.get('chunk_id')} fy={meta.get('fiscal_year')} "
                  f"score={d.get('score')} src={str(meta.get('source_file'))[-30:]}{new_flag}")
        return rk
    pipe._post_retrieval.aprocess = wrapped_aprocess

    # ---- wrap augment_context ----
    orig_aug = pipe._augment_context
    def wrapped_aug(documents, user_query, all_companies=False):
        out = orig_aug(documents, user_query, all_companies=all_companies)
        print("\n=== STAGE C: AUGMENT CONTEXT (%d docs out) ===" % len(out))
        for d in out:
            show_doc(d, "ctx")
        return out
    pipe._augment_context = wrapped_aug

    # ---- wrap generator stream ----
    orig_stream = pipe._generator.stream_tokens
    async def wrapped_stream(query, retrieved_docs, multi_ticker=False):
        print("\n=== STAGE D: GENERATOR INPUT (final LLM context docs) ===")
        for d in retrieved_docs:
            show_doc(d, "llm")
        n = 0
        async for tok in orig_stream(query, retrieved_docs, multi_ticker=multi_ticker):
            n += 1
            yield tok
        print("\n=== STAGE E: stream yielded %d parts ===" % n)
    pipe._generator.stream_tokens = wrapped_stream

    print("\n>>> RUNNING query_stream(ticker='NVDA', fiscal_year=None) ...\n")
    tokens = []
    async for tok in pipe.query_stream(user_query=QUERY, ticker="NVDA", fiscal_year=None):
        tokens.append(tok)

    raw = "".join(tokens)
    print("\n=== FULL RAW LLM OUTPUT ===")
    print(raw)
    print("\n=== RAW len:", len(raw), "===")

    import json
    try:
        parsed = json.loads(raw)
        print("PARSED answer:", json.dumps(parsed.get("answer"), ensure_ascii=False)[:500])
        print("PARSED extracted_raw_data:", parsed.get("extracted_raw_data"))
        print("PARSED sources:", parsed.get("sources"))
    except Exception as exc:
        print("JSON parse failed:", exc)

asyncio.run(main())