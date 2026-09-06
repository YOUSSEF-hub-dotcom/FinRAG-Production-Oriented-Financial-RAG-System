import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

for cid in ["AAPL_tbl_500b35644aa6_0014", "MSFT_tbl_xxx", "NVDA_tbl_xxx"]:
    doc = col.find_one({"chunk_id": cid})
    if doc:
        t = doc["raw_text"]
        print("="*80)
        print(cid, "len", len(t))
        print("--- first 800 chars ---")
        print(t[:800])
        print("--- contains 416,161 near start? index of '416,161':", t.find("416,161"))
