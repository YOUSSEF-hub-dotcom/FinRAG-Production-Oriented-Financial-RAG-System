import json, os, sys
sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
from pipeline import FinancialRAGPipeline
import sys
pipe = FinancialRAGPipeline(enable_hybrid_retrieval=True, enable_post_retrieval=True)
mongo = pipe._mongo_indexer
for tk, yr in [("AAPL","2025"), ("MSFT","2025"), ("MSFT","2024"), ("NVDA","2025"), ("NVDA","2026")]:
    chunks = mongo.get_chunks_by_filter({"ticker": tk, "fiscal_year": yr})
    tables = [c for c in chunks if c.get("chunk_type") == "table" and c.get("raw_text")]
    hit = []
    for c in tables:
        rt = (c.get("raw_text") or "").lower()
        if "operating income" in rt and ("net sales" in rt or "net revenue" in rt or " revenue" in rt):
            hit.append((c.get("chunk_id"), len(c.get("raw_text") or ""),
                        "opinc", "tns" if "total net sales" in rt else "netrev" if "net revenue" in rt else "revenue"))
    print(f"{tk}/{yr}: {len(tables)} tables; income-statement hits={hit}")
    # print the raw_text markers for the best candidate
    if hit:
        cid = hit[0][0]
        c = next(t for t in tables if t.get("chunk_id") == cid)
        rt = c.get("raw_text") or ""
        for line in rt.splitlines()[:40]:
            s = line.strip()
            if any(k in s.lower() for k in ("net sales", "revenue", "operating income", "net income", "gross margin")):
                print("    |", s[:110])
try:
    pipe.close()
except Exception:
    pass