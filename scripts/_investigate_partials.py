#!/usr/bin/env python3
import re
from pymongo import MongoClient

coll = MongoClient("mongodb://localhost:27017")["financial_rag"]["raw_chunks"]

def show(ticker, year, label, patterns):
    print("=" * 70)
    print(f"{ticker} FY{year} — {label}")
    docs = list(coll.find({"ticker": ticker, "fiscal_year": year},
                          {"chunk_id": 1, "chunk_type": 1, "raw_text": 1}))
    print(f"chunks scanned: {len(docs)}")
    for d in docs:
        t = d.get("raw_text", "") or ""
        for pat in patterns:
            for m in re.finditer(pat, t, re.I):
                s = m.start()
                ctx = t[max(0, s - 80): s + 120].replace("\n", " ")
                print(f"  [{d.get('chunk_id')}] ...{ctx}...")
                break

show("MSFT", "2025", "net income", [r"net income"])
show("NVDA", "2025", "net income", [r"net income"])
show("NVDA", "2025", "research and development", [r"research and development", r"r&d"])
