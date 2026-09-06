import re
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

PH = re.compile(r"%%TABLE_(\d+)%%")

total_refs = 0
unresolved = []  # (text_chunk_id, placeholder, resolved table chunk id)
resolved = 0
by_ticker_year = {}

for doc in col.find({"chunk_type": "text", "raw_text": {"$regex": r"%%TABLE_"}}):
    ticker = doc.get("ticker")
    year = doc.get("fiscal_year")
    source = doc.get("source_file")
    key = (ticker, year, source)
    by_ticker_year.setdefault(key, 0)
    for m in PH.finditer(doc["raw_text"]):
        n = int(m.group(1))
        total_refs += 1
        by_ticker_year[key] += 1
        cid = f"{ticker}_tbl_{__import__('hashlib').md5(f'{ticker}:{year}:tbl:{source}:{n}'.encode()).hexdigest()[:12]}_{n:04d}"
        found = col.find_one({"chunk_id": cid})
        if found:
            resolved += 1
        else:
            unresolved.append((doc["chunk_id"], n, cid))

print("text chunks w/ placeholders resolved against deterministic _make_chunk_id")
print("total placeholders referenced:", total_refs)
print("resolved to existing table chunk:", resolved)
print("UNRESOLVED:", len(unresolved))
for u in unresolved[:30]:
    print("  ", u)

print("\nby (ticker, year, source):", dict(sorted(by_ticker_year.items())))
