import json, traceback
from pipeline import FinancialRAGPipeline

p = FinancialRAGPipeline(enable_hybrid_retrieval=True, enable_post_retrieval=True)
queries = [
    ("AAPL", "2025", "What was Apple's total net revenue in FY2025?"),
    ("AAPL", "2025", "What is Apple's operating margin in FY2025?"),
    ("MSFT", "2025", "What was Microsoft's Intelligent Cloud segment revenue in FY2025?"),
    ("NVDA", "2025", "What is NVIDIA's gross margin in FY2025?"),
    ("NVDA", "2025", "What was NVIDIA's Data Center revenue in FY2025?"),
    ("MSFT", "2025", "What was Microsoft's net income in FY2025?"),
]
for t, fy, q in queries:
    try:
        p.query(q, ticker=t, fiscal_year=fy)
    except Exception as e:
        pass  # generation (Groq) will fail on TPD; we only care about retrieval
    ctx = getattr(p, "_last_contexts", [])
    print(f"\n=== {t} {fy}: {q}")
    print(f"  reranked chunks fed to LLM: {len(ctx)}")
    for i, c in enumerate(ctx[:3]):
        txt = c.get("text") or c.get("raw_text") or ""
        sc = c.get("rerank_score")
        print(f"  [{i}] score={sc} :: {txt[:200].replace(chr(10),' ')}")
