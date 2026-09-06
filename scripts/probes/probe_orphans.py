import os
import re
import hashlib
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

PH = re.compile(r"%%TABLE_(\d+)%%")

def make_chunk_id(ticker, year, ctype, source, idx):
    h = hashlib.md5(f"{ticker}:{year}:{ctype}:{source}:{idx}".encode()).hexdigest()[:12]
    return f"{ticker}_{ctype}_{h}_{idx:04d}"

# group text chunks by (ticker, year, source); find all referenced indexes
referenced = {}
text_docs = list(col.find({"chunk_type": "text"}))
for td in text_docs:
    key = (td.get("ticker"), td.get("fiscal_year"), td.get("source_file"))
    refs = {int(m.group(1)) for m in PH.finditer(td.get("raw_text",""))}
    referenced.setdefault(key, set()).update(refs)

# group table chunks by (ticker, year, source); extract index from chunk_id suffix
tables = list(col.find({"chunk_type": "table"}))
by_key = {}
for tb in tables:
    key = (tb.get("ticker"), tb.get("fiscal_year"), tb.get("source_file"))
    m = re.search(r"_(\d{4})$", tb["chunk_id"])
    idx = int(m.group(1)) if m else -1
    by_key.setdefault(key, []).append((idx, tb["chunk_id"]))

for key in sorted(by_key):
    refs = referenced.get(key, set())
    total = len(by_key[key])
    orphans = [(i, cid) for i, cid in by_key[key] if i not in refs]
    print(f"{key}: tables={total} referenced_idx={sorted(refs)} orphan_count={len(orphans)}")
    if orphans:
        print("   orphans:", [(i, cid) for i, cid in orphans[:10]])
