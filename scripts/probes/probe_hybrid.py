"""Test full Module 4 path: hybrid + post-retrieval on 2 queries."""
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
    enable_hybrid_retrieval=True,
    enable_post_retrieval=True,
    top_k=3,
)

qs = [
    "What were Apple's total net sales in fiscal year 2025 (ended September 27, 2025)?",
    "What was Microsoft's total revenue in fiscal year 2025 (ended June 30, 2025)?",
]

for q in qs:
    t0 = time.time()
    result = p.query(user_query=q, top_k=3)
    elapsed = time.time() - t0
    print("\n=== Q:", q[:70])
    print("ELAPSED: %.1fs" % elapsed)
    print("MODEL_USED:", result.get("model_used"))
    parsed = result.get("parsed")
    print("ANSWER:", (parsed.answer if parsed else result.get("raw_output", ""))[:400])
    ctx = getattr(p, "_last_contexts", []) or []
    print("CONTEXTS:", len(ctx))
    for i, c in enumerate(ctx):
        m = c.get("metadata", {})
        print("  ctx%d ticker=%s year=%s table=%s len=%d text[:110]=%r" % (
            i, m.get("ticker"), m.get("fiscal_year"), m.get("contains_table"),
            len(c.get("text") or ""), (c.get("text") or "")[:110]))
p.close()
