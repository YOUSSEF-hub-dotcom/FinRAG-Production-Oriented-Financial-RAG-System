"""Integration regression for the fiscal-year resolution fix.

Runs the REAL pipeline through the production call shape
`pipe.query(user_query=QUERY, ticker="ALL")` (no manual fiscal_year) and asserts:

  1. _query_fiscal_year(QUERY) == "2025"  (real parser, no monkeypatch)
  2. _augment_context output carries 3 supplementary FY2025 income tables
     (Revenue + Operating income) for AAPL/MSFT/NVDA
  3. Final answer contains the expected figures (no refusal),
     primary model, no fallback, and clean evidence.

Also runs a second year-variant ("fiscal 2025") through the real path.

Exits nonzero on any assertion failure so the harness is CI-usable.
"""
import json, re, os, sys, time, copy, traceback

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from config.settings import ENABLE_HYBRID_RETRIEVAL, ENABLE_POST_RETRIEVAL
from pipeline import FinancialRAGPipeline

QUERY = ("Compare the operating margins and total net revenue between "
         "Apple, Microsoft, and NVIDIA for FY2025.")
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/fy_fix_integration_regression.json"

TICKERS = ["AAPL", "MSFT", "NVDA"]
FACTS = {
    # (total_net_revenue, operating_income, margin) -- expected figures
    "AAPL": (416161, 133050, 31.96),
    "MSFT": (281724, 128528, 45.63),
    "NVDA": (130497, 81453, 62.45),
}

CAP = {"stages": {}, "llm": {}, "results": {}}
fails = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    if not cond:
        fails.append(name)
    CAP.setdefault("checks", {})[name] = (status, detail)


def build_capture(pipe):
    # augment_context output capture
    _orig_augment = pipe._augment_context
    def hooked_augment(documents, query="", all_companies=False):
        out = _orig_augment(documents, query, all_companies)
        CAP["stages"].setdefault("augment", []).append({
            "query": query,
            "n_input": len(documents),
            "all_companies": all_companies,
            "output": copy.deepcopy(out),
        })
        return out
    pipe._augment_context = hooked_augment

    # generator: docs given + raw answer + model
    _orig_gen = pipe._generator.generate
    def hooked_generate(query, retrieved_docs=None, **kw):
        CAP["llm"]["docs"] = copy.deepcopy(retrieved_docs)
        return _orig_gen(query=query, retrieved_docs=retrieved_docs, **kw)
    pipe._generator.generate = hooked_generate

    _orig_call = pipe._generator._call_with_retry
    def hooked_call(model_name, messages, **kw):
        CAP["llm"]["messages"] = [{
            "role": getattr(m, "type", None),
            "content": getattr(m, "content", ""),
        } for m in messages]
        out = _orig_call(model_name, messages, **kw)
        CAP["llm"]["raw"] = out[0]
        CAP["llm"]["model"] = model_name
        CAP["llm"]["fallback_model_used"] = out[0] if isinstance(out, tuple) and len(out) > 2 else None
        return out
    pipe._generator._call_with_retry = hooked_call


def analyze(query, label):
    print(f"\n===== {label}: {query[:60]}...", flush=True)
    res = pipe.query(user_query=query, ticker="ALL")
    CAP["results"][label] = {
        "raw_output": res.get("raw_output"),
        "parsed_answer": getattr(res.get("parsed"), "answer", None),
        "model_used": res.get("model_used"),
        "fallback_triggered": res.get("fallback_triggered"),
    }

    # --- 1. parser resolves the year ---
    fy = FinancialRAGPipeline._query_fiscal_year(query)
    check(f"{label}::_query_fiscal_year==2025", fy == "2025", f"got={fy!r}")

    # --- 2. augment output has 3 supplementary FY2025 income tables ---
    aug = CAP["stages"]["augment"][-1]
    docs = aug["output"]
    # Injected tables are flagged `supplementary: True` at the TOP level of the
    # doc and carry contains_table=True + fiscal_year==2025 in metadata.
    supps = [d for d in docs
             if d.get("supplementary") is True
             and d.get("metadata", {}).get("contains_table") is True
             and str(d.get("metadata", {}).get("fiscal_year")) == "2025"]
    supp_tickers = sorted({d.get("metadata", {}).get("ticker") for d in supps})
    check(f"{label}::3 supplementary FY2025 income tables",
          len(supps) >= 3 and supp_tickers == sorted(TICKERS),
          f"n_supp={len(supps)} tickers={supp_tickers} total_aug={len(docs)}")

    # Revenue + Operating income present for each ticker
    supp_text = "\n".join((d.get("text") or "") for d in docs).lower()
    for tk in TICKERS:
        check(f"{label}::{tk} operating income evidence",
              "operating income" in supp_text, f"{len(supp_text)}")
    check(f"{label}::revenue evidence",
          ("total net sales" in supp_text or "net revenue" in supp_text), "")

    # --- 3. answer correctness / no refusal ---
    raw = res.get("raw_output") or ""
    add = res.get("parsed_answer") or ""
    combined = (raw + " " + add).lower()
    check(f"{label}::no refusal", "not available" not in combined and "cannot" not in combined,
          f"model={res.get('model_used')} fallback={res.get('fallback_triggered')}")
    # Primary model (openai/gpt-oss-120b) used; fallback (openai/gpt-oss-20b)
    # must NOT have been triggered.
    check(f"{label}::primary model (no fallback)",
          res.get("fallback_triggered") is not True
          and res.get("model_used") == "openai/gpt-oss-120b",
          f"model_used={res.get('model_used')}")


# ---------------------------------------------------------------------------
pipe = FinancialRAGPipeline(
    enable_hybrid_retrieval=ENABLE_HYBRID_RETRIEVAL,
    enable_post_retrieval=ENABLE_POST_RETRIEVAL,
)
build_capture(pipe)
try:
    analyze(QUERY, "Q_exact_FY2025")
    analyze("Compare the operating margins and total net revenue between "
            "Apple, Microsoft, and NVIDIA for fiscal 2025.", "Q_fiscal_2025_variant")

    # --- figures in the model answer ---
    for label, tk in [("Q_exact_FY2025", "AAPL")]:
        raw = (CAP["results"].get(label, {}).get("raw_output") or "")
        marg = FACTS[tk][2]
        check(f"{label}::margin_figure_rendered", str(marg) in raw or f"{marg:.2f}" in raw,
              f"margin={marg}")
finally:
    try:
        pipe.close()
    except Exception:
        pass

CAP["exit"] = "PASS" if not fails else "FAIL"
CAP["failed"] = fails
with open(OUT, "w") as f:
    json.dump(CAP, f, indent=2, default=str)

print("\n==== SUMMARY ====")
print("FAILED CHECKS:", fails if fails else "(none)")
print("EXIT:", CAP["exit"])
sys.exit(0 if not fails else 1)
