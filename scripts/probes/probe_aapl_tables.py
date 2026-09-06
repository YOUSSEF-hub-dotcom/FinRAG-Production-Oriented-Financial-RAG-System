import re
from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
col = client[MONGODB_DB][MONGODB_COLLECTION]

# all AAPL 2025 table chunks
docs = list(col.find({"ticker": "AAPL", "fiscal_year": "2025", "chunk_type": "table"}, {"chunk_id": 1, "raw_text": 1}))
print("AAPL 2025 table chunks:", len(docs))

for d in docs:
    t = d["raw_text"] or ""
    if "Net sales" in t or "Operating income" in t or "Research and Development" in t or "416" in t or "416161" in t:
        print("\n=== ", d["chunk_id"], "len:", len(t))
        # find relevant lines
        for line in t.split("\n"):
            if any(k in line for k in ("Net sales", "Operating income", "Research", "416")):
                print("   ", line[:160])

# search any AAPL2025 chunk for 416161 / 416,161 variants
pat = re.compile(r"416[,\s]?161")
n = col.count_documents({"ticker": "AAPL", "fiscal_year": "2025", "raw_text": pat})
print("\nAAPL2025 chunks containing 416161:", n)

client.close()
