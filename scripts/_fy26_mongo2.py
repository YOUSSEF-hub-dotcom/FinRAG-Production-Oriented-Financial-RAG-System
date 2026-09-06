"""Re-inspect the Mongo chunk reading the REAL field names (raw_text vs text)."""
import json
import sys

sys.path.insert(0, "/home/youssef/Financial_RAG")
from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

c = MongoClient(MONGODB_URI)
coll = c[MONGODB_DB][MONGODB_COLLECTION]
docs = list(coll.find({"source_file": {"$regex": "NVDA_AI_Infrastructure_Expansion_FY2026"}}))
print("mongo docs matching source_file:", len(docs))
for d in docs:
    print("ALL KEYS:", sorted(d.keys()))
    print("chunk_id:", d.get("chunk_id"))
    print("ticker:", d.get("ticker"))
    print("fiscal_year:", repr(d.get("fiscal_year")))
    print("doc_type:", d.get("doc_type"))
    print("section:", d.get("section"))
    print("contains_table:", d.get("contains_table"))
    print("chunk_type:", d.get("chunk_type"))
    print("raw_text len:", len(d.get("raw_text") or ""))
    print("=== RAW_TEXT (full) ===")
    print(d.get("raw_text") or "(EMPTY)")
c.close()