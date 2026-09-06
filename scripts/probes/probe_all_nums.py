import os
import re
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
col = client["financial_rag"]["raw_chunks"]

# All numeric ground-truth values from test_dataset.csv (normalized without $/, . thousands separator)
targets = ["416161","391035","383285","133050","209586","34550","93736","608","281724","101832","88136",
           "120810","106265","54649","87464","23455","1364","245122","109433","29510","215938","120067",
           "490","193479","22459","193737","115186","711","130497","81453","294"]

def norm(s):
    return s.replace(",","").replace("$","").replace(".","").strip()

rows = list(col.find({}, {"chunk_id":1,"chunk_type":1,"raw_text":1,"ticker":1,"fiscal_year":1,"section":1}))

for t in targets:
    best = None
    in_text = False
    for d in rows:
        rt = d.get("raw_text","")
        # exact normalized match: look for t as a token (boundaries allow , or non-digit)
        if t in rt.replace(",",""):
            idx = rt.replace(",","").find(t)
            if best is None or idx < best[0]:
                best = (idx, len(rt), d["chunk_id"], d["ticker"], d["fiscal_year"])
        if t in rt and d["chunk_type"]=="text":
            in_text = True
    if best:
        print(f"{t}: earliest table offset={best[0]} in len={best[1]} ({best[2]} {best[3]} {best[4]}) in_text={in_text}")
    else:
        print(f"{t}: NOT FOUND")
