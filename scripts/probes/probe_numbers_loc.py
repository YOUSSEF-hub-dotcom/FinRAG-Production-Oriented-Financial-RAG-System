import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

targets = {
    "416161": "AAPL net sales",
    "133050": "AAPL operating expenses",
    "34550": "AAPL income before taxes",
    "281724": "MSFT revenue",
    "215938": "NVDA revenue",
}

for num, label in targets.items():
    print("="*80)
    print(f"TARGET {num} ({label})")
    hits = []
    for doc in col.find({}, {"chunk_id": 1, "chunk_type": 1, "raw_text": 1, "ticker": 1, "fiscal_year": 1, "section": 1}):
        idx = doc.get("raw_text", "").find(num)
        if idx >= 0:
            hits.append((doc["chunk_id"], doc["chunk_type"], doc.get("ticker"), doc.get("fiscal_year"), doc.get("section"), idx, len(doc.get("raw_text", ""))))
    print("hits:", len(hits))
    for h in hits[:8]:
        print("   ", h)
    # For first table hit, report position within 800 chars?
    first_table = next((h for h in hits if h[1] == "table"), None)
    if first_table:
        print("   first table hit at char idx", first_table[5], "of len", first_table[6], "-> within first 800?", first_table[5] < 800)
