#!/usr/bin/env python3
import re
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
coll = client["financial_rag"]["raw_chunks"]

nvda = list(coll.find({"ticker": "NVDA"}, {"chunk_id": 1, "chunk_type": 1,
                                           "contains_table": 1, "section": 1,
                                           "raw_text": 1}))

print("=== 2 table chunks mentioning 'Data Center' ===")
for c in nvda:
    t = c.get("raw_text", "") or ""
    if c.get("chunk_type") == "table" and re.search(r"data center", t, re.I):
        print("---")
        print("chunk_id:", c.get("chunk_id"), "section:", c.get("section"))
        print(repr(t[:900]))

print("\n=== chunks containing '115193' (any) ===")
n = 0
for c in nvda:
    t = c.get("raw_text", "") or ""
    if "115193" in t:
        n += 1
        print("chunk_id:", c.get("chunk_id"), "type:", c.get("chunk_type"),
              "contains_table:", c.get("contains_table"))
        print(repr(t[:300]))
        if n >= 5:
            break
print("total with 115193:", n)

print("\n=== chunks containing 'Data Center' AND a revenue-ish number (115\\d{3}) ===")
for c in nvda:
    t = c.get("raw_text", "") or ""
    if re.search(r"data center", t, re.I) and re.search(r"115\d{3}", t):
        print("chunk_id:", c.get("chunk_id"), "type:", c.get("chunk_type"))
        print(repr(t[:400]))
