from pymongo import MongoClient
import re
mc = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=5000)
coll = mc["financial_rag"]["raw_chunks"]
docs = list(coll.find({"ticker":"MSFT","fiscal_year":"2025"}))
text = " ".join(d.get("raw_text","") for d in docs)
# Find operating income and revenue
for term in ["Operating income", "Total revenue", "operating margin"]:
    for m in re.finditer(term, text, re.I):
        print(repr(text[m.start()-80:m.end()+120])[:400])
        break
# Also check for margin in tables
for d in docs:
    rt=d.get("raw_text","")
    if "margin" in rt.lower() and ("%" in rt or "45" in rt):
        # find snippet with 45
        for m in re.finditer(r"45\.?\d*%?", rt):
            idx=m.start()
            print(d["chunk_id"], repr(rt[idx-60:idx+60]))
        # break after first
        if "45" in rt:
            break
# Check what the subset expected for MSFT operating margin
import pathlib
subset = pathlib.Path("scripts/_audit_subset.py").read_text()
for line in subset.splitlines():
    if "MSFT" in line and "margin" in line.lower():
        print("subset line:", line)
