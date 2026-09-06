"""Mongo-only: find which table chunks contain each ground-truth number."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import pandas as pd
from pymongo import MongoClient

from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

MONEY_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text):
    out = set()
    for m in MONEY_RE.finditer(text):
        tok = re.sub(r"[, ]", "", m.group(0))
        if tok and tok.replace(".", "").isdigit():
            try:
                if float(tok) >= 10000 and "." not in tok:
                    out.add(tok)
            except ValueError:
                pass
    return sorted(out)


client = MongoClient(MONGODB_URI)
coll = client[MONGODB_DB][MONGODB_COLLECTION]

df = pd.read_csv("artifacts/test_dataset.csv").head(12)
filing_nums = {}
for _, row in df.iterrows():
    nums = extract_numbers(str(row["ground_truth"]))
    for n in nums:
        filing_nums.setdefault(n, [])

# For each target number, find its chunk.
targets = sorted({n for row in df.itertuples() for n in extract_numbers(str(row.ground_truth))})
print("targets:", targets)

for t in targets:
    hits = []
    for doc in coll.find(
        {"chunk_type": "table", "raw_text": {"$regex": re.escape(t)}},
        {"chunk_id": 1, "ticker": 1, "fiscal_year": 1, "raw_text": 1, "section": 1},
    ):
        text = doc.get("raw_text", "")
        hits.append(
            (doc.get("chunk_id"), doc.get("ticker"), doc.get("fiscal_year"),
             len(text), text.find(t), (doc.get("section") or "")[:40])
        )
    print(f"{t}: {hits}")
