#!/usr/bin/env python3
import re
from pymongo import MongoClient
coll = MongoClient("mongodb://localhost:27017")["financial_rag"]["raw_chunks"]

def find(ticker, nums, label):
    print("==", ticker, label, "==")
    for doc in coll.find({"ticker": ticker, "fiscal_year": "2025"}, {"chunk_id": 1, "raw_text": 1}):
        t = doc.get("raw_text", "")
        for n in nums:
            i = t.find(n)
            if i >= 0:
                ctx = t[max(0, i - 70): i + 70].replace("\n", " ")
                print("  %s num=%s: ...%s..." % (doc.get("chunk_id"), n, ctx))

find("NVDA", ["72880", "72881", "72,880", "72,881"], "net income candidates")
find("NVDA", ["12914", "12823", "12,914", "12,823"], "R&D candidates")

doc = coll.find_one({"chunk_id": "NVDA_tbl_f2a5a0b5238b_0021"}, {"raw_text": 1})
if doc:
    t = doc["raw_text"]
    for kw in ["Net income", "Research and development"]:
        i = t.find(kw)
        if i >= 0:
            print("  f2a5a0b5238b_0021", kw, "::", t[i - 10: i + 170].replace("\n", " "))
