import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

cid = "AAPL_tbl_500b35644aa6_0014"
doc = col.find_one({"chunk_id": cid})
t = doc["raw_text"]

# Print chars 800..4200 to see structure around the 416161 number (at 3118)
print("len:", len(t))
print("---- 700..1400 ----")
print(t[700:1400])
print("---- 2900..3400 (around 416161 @3118) ----")
print(t[2900:3400])
