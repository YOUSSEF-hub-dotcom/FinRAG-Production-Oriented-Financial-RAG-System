import config.settings as S
S.HYBRID_TOP_K = 100
S.POST_RETRIEVAL_RERANK_TOP_N = 20
from pipeline import FinancialRAGPipeline
import pipeline as P
P.HYBRID_TOP_K = 100  # ensure the value propagates to the engine
print("pipeline.HYBRID_TOP_K =", P.HYBRID_TOP_K)

p = FinancialRAGPipeline(enable_hybrid_retrieval=True, enable_post_retrieval=True)
try:
    p.query("What was NVIDIA's Data Center revenue in FY2025?", ticker="NVDA", fiscal_year="2025")
except Exception:
    pass

ctx = getattr(p, "_last_contexts", [])
print(f"\nreranked chunks fed to LLM (candidate_k=100, top_n=20): {len(ctx)}\n")
found_rank = None
for i, c in enumerate(ctx):
    txt = c.get("text") or c.get("raw_text") or ""
    sc = c.get("rerank_score")
    has_dc = ("data center" in txt.lower()) or ("115193" in txt)
    if has_dc and found_rank is None:
        found_rank = i
    print(f"[{i}] score={sc} DC={has_dc} :: {txt[:140].replace(chr(10),' ')}")
print("\n>>> Data Center segment table first appears at rank:", found_rank)
