"""TARGETED CORRECTNESS DEBUG +- runtime instrumentation of the EXACT
production call: pipe.query(user_query=QUERY, ticker="ALL") i.e. the Next.js
payload {user_query, ticker: "ALL"} (no fiscal_year, no tickers).

No production files are modified; every hook below is a runtime monkeypatch.
"""
import copy, json, os, sys, time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from config.settings import (
    ENABLE_HYBRID_RETRIEVAL, ENABLE_POST_RETRIEVAL, SUPPORTED_TICKERS,
)
from pipeline import FinancialRAGPipeline

QUERY = ("Compare the operating margins and total net revenue between "
         "Apple, Microsoft, and NVIDIA for FY2025.")
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/debug_e2_refusal.json"

CAP = {"stages": {}, "llm": {}}

def wrap(pipe, name):
    orig = getattr(pipe, name)

    def hooked(*a, **kw):
        meth = getattr(type(pipe), name)
        if name == "_balanced_ticker_subretrievals_sync":
            CAP["stages"]["balanced_call"] = {
                "query": a[1] if len(a) > 1 else kw.get("query"),
                "tickers": kw.get("tickers") or (a[2] if len(a) > 2 else None),
                "fiscal_year": kw.get("fiscal_year") or (a[3] if len(a) > 3 else None),
            }
            out = orig(*a, **kw)
            CAP["stages"]["balanced_output"] = copy.deepcopy(out)
            return out
        return orig(*a, **kw)

    setattr(pipe, name, hooked)


def main():
    pipe = FinancialRAGPipeline(
        enable_hybrid_retrieval=ENABLE_HYBRID_RETRIEVAL,
        enable_post_retrieval=ENABLE_POST_RETRIEVAL,
    )
    wrap(pipe, "_balanced_ticker_subretrievals_sync")

    # resolve_table_placeholders
    _orig_resolve = pipe._resolve_table_placeholders
    def hooked_resolve(documents):
        CAP["stages"]["resolve_input"] = copy.deepcopy(documents)
        out = _orig_resolve(documents)
        CAP["stages"]["resolve_output"] = copy.deepcopy(out)
        return out
    pipe._resolve_table_placeholders = hooked_resolve

    # augment_context
    _orig_augment = pipe._augment_context
    def hooked_augment(documents, query="", all_companies=False):
        CAP["stages"]["augment_call"] = {
            "n_input": len(documents),
            "query": query,
            "all_companies": all_companies,
        }
        # capture pre-augment input docs (per ticker group)
        CAP["stages"]["augment_input"] = copy.deepcopy(documents)
        out = _orig_augment(documents, query, all_companies)
        CAP["stages"]["augment_output"] = copy.deepcopy(out)
        return out
    pipe._augment_context = hooked_augment

    # generator: capture the docs passed to generate + the messages + raw out
    _orig_gen = pipe._generator.generate
    def hooked_generate(query, retrieved_docs=None, **kw):
        CAP["llm"]["docs_sent_to_generator"] = copy.deepcopy(retrieved_docs)
        CAP["llm"]["gen_kwargs"] = {k: v for k, v in kw.items() if k != "retrieved_docs"}
        return _orig_gen(query=query, retrieved_docs=retrieved_docs, **kw)
    pipe._generator.generate = hooked_generate

    _orig_call = pipe._generator._call_with_retry
    def hooked_call(model_name, messages, **kw):
        CAP["llm"]["messages"] = [{
            "role": getattr(m, "type", None),
            "content": getattr(m, "content", ""),
        } for m in messages]
        out = _orig_call(model_name, messages, **kw)
        CAP["llm"]["call_returns"] = CAP["llm"].get("call_returns", []) + [{
            "model": model_name, "raw": out[0], "ttft": out[2],
        }]
        return out
    pipe._generator._call_with_retry = hooked_call

    # guardrail: capture what it sees + the verdict
    guard_results = []
    _orig_check = pipe._guardrail.check
    async def hooked_check(*a, **kw):
        CAP["llm"].setdefault("guardrail_calls", []).append({
            "answer": kw.get("answer"), "extracted_raw_data": kw.get("extracted_raw_data"),
        })
        r = await _orig_check(*a, **kw)
        guard_results.append(copy.deepcopy(r))
        return r
    pipe._guardrail.check = hooked_check

    # Pre-query static probes
    probes = {
        "query": QUERY,
        "query_fiscal_year(_QUERY_YEAR_RE)": FinancialRAGPipeline._query_fiscal_year(QUERY),
        "detect_all_tickers": FinancialRAGPipeline._detect_all_tickers(QUERY),
        "detect_ticker": FinancialRAGPipeline._detect_ticker(QUERY),
        "mentioned_tickers": FinancialRAGPipeline._mentioned_tickers(QUERY),
        "has_multi_ticker_intent": FinancialRAGPipeline._has_multi_ticker_intent(QUERY),
        "SUPPORTED_TICKERS": SUPPORTED_TICKERS,
    }
    CAP["probes"] = probes

    t0 = time.time()
    result = pipe.query(user_query=QUERY, ticker="ALL")
    CAP["elapsed_s"] = round(time.time() - t0, 1)
    CAP["result"] = {
        "raw_output": result.get("raw_output"),
        "parsed_answer": getattr(result.get("parsed"), "answer", None),
        "model_used": result.get("model_used"),
        "fallback_triggered": result.get("fallback_triggered"),
    }
    if guard_results:
        CAP["guardrail_verdicts"] = guard_results

    with open(OUT, "w") as f:
        json.dump(CAP, f, indent=2, default=str)
    print("saved", OUT, flush=True)
    print("RESULT:", json.dumps(CAP["result"], default=str), flush=True)
    print("balanced:", len(CAP["stages"].get("balanced_output", [])), flush=True)
    print("resolve in/out:", CAP["stages"].get("resolve_input") is not None,
          len(CAP["stages"].get("resolve_output", [])), flush=True)
    print("augment in/out:",
          CAP["stages"].get("augment_call"),
          len(CAP["stages"].get("augment_output", [])), flush=True)
    try:
        pipe.close()
    except Exception:
        pass


main()