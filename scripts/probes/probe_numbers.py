import re
from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
col = client[MONGODB_DB][MONGODB_COLLECTION]

# Does the actual number exist in any chunk?
probes = {
    "416,161": ("AAPL", "2025"),   # total net sales FY2025
    "133,050": ("AAPL", "2025"),   # operating income FY2025
    "34,550": ("AAPL", "2025"),    # R&D expense FY2025
    "281,724": ("MSFT", "2025"),   # MSFT total revenue FY2025
    "215,938": ("NVDA", "2026"),   # NVDA total revenue FY2026
}

for num, (tick, yr) in probes.items():
    q = {"ticker": tick, "fiscal_year": yr, "raw_text": {"$regex": re.escape(num)}}
    hits = list(col.find(q, {"chunk_id": 1, "chunk_type": 1, "section": 1}))
    print(f"{num} ({tick} {yr}): {len(hits)} chunks")
    for h in hits[:3]:
        print("   ", h.get("chunk_id"), h.get("chunk_type"), "|", h.get("section"))

client.close()
