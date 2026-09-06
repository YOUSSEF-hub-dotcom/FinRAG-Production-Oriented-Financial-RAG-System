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

# For each (ticker, year) with placeholders, map every placeholder -> chunk_id and list
text_docs = list(col.find({"chunk_type": "text", "raw_text": {"$regex": r"%%TABLE_"}}))
for td in text_docs:
    ticker = td.get("ticker"); year = td.get("fiscal_year"); source = td.get("source_file")
    idxs = sorted(int(m.group(1)) for m in PH.finditer(td["raw_text"]))
    print(f"{td['chunk_id']} [{ticker} {year} {source}] placeholders={idxs}")

# check specific MSFT/NVDA table existence
for cid in ["MSFT_tbl_496d86e9a58f_0013", "NVDA_tbl_91c7ded32187_0020"]:
    d = col.find_one({"chunk_id": cid})
    print(cid, "exists=", d is not None, "| chunk_type=", d.get("chunk_type") if d else None)

# Does any text doc reference index 13 for MSFT/2025 or index 20 for NVDA/2026?
for td in text_docs:
    if td.get("ticker")=="MSFT" and td.get("fiscal_year")=="2025" and "%%TABLE_13%%" in td["raw_text"]:
        print("MSFT/2025 text references TABLE_13:", td["chunk_id"])
    if td.get("ticker")=="NVDA" and td.get("fiscal_year")=="2026" and "%%TABLE_20%%" in td["raw_text"]:
        print("NVDA/2026 text references TABLE_20:", td["chunk_id"])
