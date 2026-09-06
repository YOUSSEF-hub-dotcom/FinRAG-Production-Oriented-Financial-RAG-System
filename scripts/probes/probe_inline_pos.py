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

text_docs = list(col.find({"chunk_type": "text", "raw_text": {"$regex": r"%%TABLE_"}}))
tbl_docs = {d["chunk_id"]: d["raw_text"] for d in col.find({"chunk_type": "table"})}

# For each text chunk, inline tables and compute new char offset of each number occurrence.
# We report, per target number, the earliest position within ANY inlined text doc.
targets = ["416161","391035","383285","133050","209586","34550","93736","608","281724","101832","88136",
           "120810","106265","54649","87464","23455","1364","245122","109433","29510","215938","120067",
           "490","193479","22459","193737","115186","711","130497","81453","294"]

results = {t: [] for t in targets}
for td in text_docs:
    text = td["raw_text"]
    ticker = td.get("ticker"); year = td.get("fiscal_year"); source = td.get("source_file")
    def repl(m):
        n = int(m.group(1))
        cid = make_chunk_id(ticker, year, "tbl", source, n)
        return tbl_docs.get(cid, m.group(0))
    inlined = PH.sub(repl, text)
    flat = inlined.replace(",", "")
    for t in targets:
        idx = flat.find(t)
        if idx >= 0:
            results[t].append(idx)

print("worst-case (max) earliest offset needed per number, after inlining:")
worst_all = 0
for t, offsets in sorted(results.items(), key=lambda kv: min(kv[1]) if kv[1] else 0):
    if offsets:
        m = max(offsets)  # the doc where this number appears latest
        e = min(offsets)
        worst_all = max(worst_all, m)
        print(f"{t}: earliest={e}  latest={m}  (n={len(offsets)})")
    else:
        print(f"{t}: NOT FOUND after inline")
print("\nMAX latest offset across all numbers:", worst_all)
