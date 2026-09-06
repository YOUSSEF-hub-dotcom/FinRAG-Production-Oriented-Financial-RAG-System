#!/usr/bin/env python3
from pymongo import MongoClient
c = MongoClient("mongodb://localhost:27017")["financial_rag"]["raw_chunks"]
doc = c.find_one({"chunk_id": "NVDA_tbl_bf56f12bfafa_0059"})
print("keys:", list(doc.keys()))
for k in doc:
    if k in ("raw_text", "content", "table_html", "table_text", "text", "table_markdown", "cleaned_text"):
        v = doc.get(k) or ""
        s = str(v)
        print(f"--- {k} len={len(s)} has115193={'115193' in s} hasDC={'data center' in s.lower()}")
        print(repr(s[:700]))
