"""Check what NVDA_tbl_bf56f12bfafa_0059 is and NVDA chunk inventory wrt FY2026."""
import sys
sys.path.insert(0, "/home/youssef/Financial_RAG")
from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

c = MongoClient(MONGODB_URI)
coll = c[MONGODB_DB][MONGODB_COLLECTION]

print("=== NVDA_tbl_bf56f12bfafa_0059 ===")
d = coll.find_one({"chunk_id": "NVDA_tbl_bf56f12bfafa_0059"})
if d:
    print("ticker:", d.get("ticker"), "fiscal_year:", repr(d.get("fiscal_year")),
          "section:", d.get("section"), "source_file:", d.get("source_file"))
    print("raw_text:", (d.get("raw_text") or "")[:600])
else:
    print("NOT FOUND")

print()
print("=== ALL NVDA chunks by fiscal_year + source_file ===")
from collections import Counter
years = Counter()
sources = Counter()
for d in coll.find({"ticker": "NVDA"}):
    years[d.get("fiscal_year")] += 1
    sources[d.get("source_file")] += 1
print("years:", dict(years))
print("sources:", dict(sources))

print()
print("=== NVDA chunks with fiscal_year FY2026 or 2026 ===")
for fd in ({"ticker": "NVDA", "fiscal_year": "2026"}, {"ticker": "NVDA", "fiscal_year": "FY2026"}):
    n = coll.count_documents(fd)
    print(fd, "->", n)
    for x in coll.find(fd).limit(5):
        print("   ", x.get("chunk_id"), x.get("source_file"), x.get("section"))
c.close()