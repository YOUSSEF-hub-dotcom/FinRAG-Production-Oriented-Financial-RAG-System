from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
db = client["financial_rag"]
col = db["raw_chunks"]

for cid in ["AAPL_tbl_500b35644aa6_0014", "AAPL_tbl_6be6251e993f_0000", "AAPL_tbl_6be6251e993f_0001"]:
    doc = col.find_one({"chunk_id": cid})
    if doc:
        print(cid, "len(raw_text)=", len(doc["raw_text"]), "| ticker=", doc["ticker"], "| year=", doc["fiscal_year"], "| source=", doc.get("source_file"))

# overall stats for table chunks
tbls = list(col.find({"chunk_type": "table"}))
lens = sorted(len(d["raw_text"]) for d in tbls)
print("total table chunks:", len(tbls))
print("median len:", lens[len(lens)//2] if lens else 0, "max:", lens[-1] if lens else 0, "min:", lens[0] if lens else 0)
big = [d for d in tbls if len(d["raw_text"]) > 8000]
print("tables > 8000 chars:", len(big))
for d in big[:5]:
    print("  ", d["chunk_id"], len(d["raw_text"]))
