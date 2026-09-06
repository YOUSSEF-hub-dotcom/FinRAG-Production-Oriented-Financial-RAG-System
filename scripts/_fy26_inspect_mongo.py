"""Step 2-3: Inspect the actual stored Mongo chunk + Qdrant point for the
NVDA FY2026 supplemental filing, and dump the parsed PDF text content."""
import json
import sys

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
sys.path.insert(0, "/home/youssef/Financial_RAG/src/1_ingestion")

from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

PDF = "/home/youssef/Financial_RAG/data/NVDA/10-K/FY2026/NVDA_AI_Infrastructure_Expansion_FY2026.pdf"

# ---- 1. Dump PDF text via the SAME parser used by ingestion ----
print("========== PDF PARSED CONTENT ==========")
try:
    from app.api.parsers import APIFileParser
    data = open(PDF, "rb").read()
    parsed = APIFileParser().parse_file(data, "NVDA_AI_Infrastructure_Expansion_FY2026.pdf")
    print("parser text_content length:", len(parsed.get("text_content") or ""))
    print("--- FULL TEXT ---")
    print(parsed.get("text_content") or "(EMPTY)")
except Exception as e:
    print("parser error:", repr(e))

# ---- 2. Inspect Mongo chunks for this source file ----
print("\n========== MONGO CHUNKS ==========")
c = MongoClient(MONGODB_URI)
db = c[MONGODB_DB]
coll = db[MONGODB_COLLECTION]
# find by source_file name pattern (any fiscal_year)
docs = list(coll.find({"source_file": {"$regex": "NVDA_AI_Infrastructure_Expansion_FY2026"}}))
print("mongo docs matching source_file:", len(docs))
for d in docs:
    print(json.dumps({
        "chunk_id": d.get("chunk_id"),
        "ticker": d.get("ticker"),
        "fiscal_year": d.get("fiscal_year"),
        "doc_type": d.get("doc_type"),
        "section": d.get("section"),
        "contains_table": d.get("contains_table"),
        "chunk_type": d.get("chunk_type"),
        "source_file": d.get("source_file"),
        "text": (d.get("text") or "")[:1500],
    }, indent=2, default=str))
if not docs:
    # also check whether fiscal_year UNKNOWN chunks exist for NVDA
    print("No chunks matched. Searching NVDA chunks with fiscal_year UNKNOWN:")
    unknown = list(coll.find({"ticker": "NVDA", "fiscal_year": "UNKNOWN"}))
    print("NVDA UNKNOWN chunks:", len(unknown))
    for d in unknown[:10]:
        print(" -", d.get("chunk_id"), d.get("source_file"), d.get("section"))
c.close()
