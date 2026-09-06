"""Confirm (a) income-statement tables exist in Mongo with Operating income rows,
and (b) that routing fiscal_year='2025' into _augment_context (simulated fix)
restores the evidence, with only runtime monkeypatching (no prod edits)."""
import json, re, os, sys, time, copy

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from config.settings import ENABLE_HYBRID_RETRIEVAL, ENABLE_POST_RETRIEVAL
from pipeline import FinancialRAGPipeline

QUERY = ("Compare the operating margins and total net revenue between "
         "Apple, Microsoft, and NVIDIA for FY2025.")
OUT = "/home/youssef/Financial_RAG/artifacts/ab2_benchmark/debug_e2_fixsim.json"
CAP = {}

# ---- (a) Mongo income tables ------------------------------------------------
pipe = FinancialRAGPipeline(
    enable_hybrid_retrieval=ENABLE_HYBRID_RETRIEVAL,
    enable_post_retrieval=ENABLE_POST_RETRIEVAL,
)
mongo = pipe._mongo_indexer
table_check = {}
for tk in ["AAPL", "MSFT", "NVDA"]:
    for yr in ["2025", "2026"]:
        try:
            chunks = mongo.get_chunks_by_filter({"ticker": tk, "fiscal_year": yr})
        except Exception as e:
            table_check[f"{tk}/{yr}"] = {"error": repr(e)}; continue
        tables = [c for c in chunks if c.get("chunk_type") == "table" and c.get("raw_text")]
        detail = []
        for c in tables:
            rt = (c.get("raw_text") or "").lower()
            detail.append({
                "id": c.get("chunk_id"),
                "len": len(c.get("raw_text") or ""),
                "has_operating_income": "operating income" in rt,
                "has_op_margin": "operating margin" in rt,
                "has_revenue": any(k in rt for k in ("total net sales", "net revenue", "'revenue'", " revenue", "total revenue")),
                "has_total_net_sales": "total net sales" in rt,
                "has_net_income": "net income" in rt,
            })
        table_check[f"{tk}/{yr}"] = {"n_tables": len(tables), "tables": detail[:10]}
CAP["mongo_income_tables"] = table_check
print("MONGO:", json.dumps({k: (v if 'error' in v else {'n_tables': v['n_tables'], 'rows': v['tables'][:5]}) for k, v in table_check.items()}, indent=1, default=str)[:2000], flush=True)

# ---- (b) simulated fix: shadow _query_fiscal_year -> "2025" ----------------
orig_static = FinancialRAGPipeline._query_fiscal_year
pipe._query_fiscal_year = staticmethod(lambda q: "2025")

# instrument _augment_context output
aug_out = {}
_orig_auc = pipe._augment_context
def hooked_auc(documents, query="", all_companies=False):
    out = _orig_auc(documents, query, all_companies)
    aug_out["docs"] = copy.deepcopy(out)
    aug_out["call"] = {"n": len(documents), "all_companies": all_companies}
    return out
pipe._augment_context = hooked_auc

# capture LLM messages + raw
llm_cap = {}
_orig_call = pipe._generator._call_with_retry
def hooked_call(model_name, messages, **kw):
    llm_cap["human_context"] = messages[-1].content
    out = _orig_call(model_name, messages, **kw)
    llm_cap["raw"] = out[0]
    llm_cap["model"] = model_name
    return out
pipe._generator._call_with_retry = hooked_call

t0 = time.time()
result = pipe.query(user_query=QUERY, ticker="ALL")
CAP["elapsed_s"] = round(time.time() - t0, 1)
CAP["augment"] = aug_out
CAP["llm"] = llm_cap
CAP["answer"] = result.get("raw_output", "")
CAP["model_used"] = result.get("model_used")
CAP["fallback"] = result.get("fallback_triggered")

with open(OUT, "w") as f:
    json.dump(CAP, f, indent=2, default=str)
print("saved", OUT, flush=True)
print("ANSWER:", CAP["answer"][:800], flush=True)
try:
    pipe.close()
except Exception:
    pass