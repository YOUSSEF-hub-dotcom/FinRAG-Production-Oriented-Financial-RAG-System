"""Single live query test: verify retrieval + generation produce a real answer."""
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")
sys.path.insert(0, "src/5_generation")
sys.path.insert(0, "src/3_pre_retrieval")
sys.path.insert(0, "src/4_retrieval")

from pipeline import FinancialRAGPipeline

p = FinancialRAGPipeline(
    enable_cache=False,
    enable_guardrail=False,
    enable_pre_retrieval=False,
    enable_hybrid_retrieval=False,
    enable_post_retrieval=False,
    top_k=3,
)

q = "What were Apple's total net sales in fiscal year 2025 (ended September 27, 2025)?"
t0 = time.time()
result = p.query(user_query=q, top_k=3)
elapsed = time.time() - t0

print("ELAPSED: %.1fs" % elapsed)
print("MODEL_USED:", result.get("model_used"))
print("FALLBACK_TRIGGERED:", result.get("fallback_triggered"))
print("RAW_OUTPUT[:600]:", str(result.get("raw_output"))[:600])
print("PARSED:", result.get("parsed"))
print("CONTEXTS:", len(getattr(p, "_last_contexts", []) or []))
for i, c in enumerate((getattr(p, "_last_contexts", []) or [])[:3]):
    print("  ctx%d: ticker=%s year=%s table=%s text[:150]=%r" % (
        i,
        c.get("metadata", {}).get("ticker"),
        c.get("metadata", {}).get("fiscal_year"),
        c.get("metadata", {}).get("contains_table"),
        (c.get("text") or "")[:150],
    ))
p.close()
