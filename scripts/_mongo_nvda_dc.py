import pymongo
c = pymongo.MongoClient("mongodb://localhost:27017")
col = c["financial_rag"]["raw_chunks"]
docs = list(col.find(
    {"ticker": "NVDA", "fiscal_year": "2025",
     "$or": [{"raw_text": {"$regex": "Data Center", "$options": "i"}},
             {"raw_text": {"$regex": "data center", "$options": "i"}}]},
    {"section": 1, "chunk_type": 1, "table_index": 1, "raw_text": 1, "page": 1}
))
print("NVDA 2025 chunks mentioning Data Center:", len(docs))
dc_rev = [d for d in docs if ("115" in (d.get("raw_text") or "") or "revenue" in (d.get("raw_text") or "").lower())]
print("of those containing a revenue-ish number (115...):", len(dc_rev))
for d in docs[:8]:
    t = (d.get("raw_text") or "")
    print("\n--- section=", d.get("section"), "type=", d.get("chunk_type"), "page=", d.get("page"))
    print("   len=", len(t), "::", t[:240].replace("\n", " "))
