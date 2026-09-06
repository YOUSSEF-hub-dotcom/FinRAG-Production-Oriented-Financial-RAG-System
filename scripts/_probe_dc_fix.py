import config.settings as S
S.ENABLE_HYBRID_RETRIEVAL = True
S.ENABLE_POST_RETRIEVAL = True
from pipeline import FinancialRAGPipeline

p = FinancialRAGPipeline(enable_hybrid_retrieval=True, enable_post_retrieval=True)
res = None
try:
    res = p.query("What was NVIDIA's Data Center revenue in FY2025?", ticker="NVDA", fiscal_year="2025")
except Exception as e:
    print("query exception:", repr(e)[:200])

ctx = getattr(p, "_last_contexts", []) or []
print(f"\ncontexts fed to LLM: {len(ctx)}\n")
found_rank = None
for i, c in enumerate(ctx):
    txt = (c.get("text") or c.get("raw_text") or "")
    has_fig = "115193" in txt
    has_dc = "data center" in txt.lower()
    if has_fig and found_rank is None:
        found_rank = i
    print(f"[{i}] fig115193={has_fig} dc={has_dc} :: {txt[:120].replace(chr(10),' ')}")

print("\n>>> NVDA DC segment table (115193) first appears at rank:", found_rank)
if res:
    raw = res.get("raw_output") if isinstance(res, dict) else getattr(res, "raw_output", None)
    parsed = res.get("parsed") if isinstance(res, dict) else getattr(res, "parsed", None)
    ans = (parsed.answer if parsed else None) or (raw or "")
    print(">>> model_used:", res.get("model_used") if isinstance(res, dict) else getattr(res, "model_used", None))
    print(">>> ANSWER:", (ans or "")[:300])
    print(">>> has 115186 in answer:", "115186" in (ans or ""))
