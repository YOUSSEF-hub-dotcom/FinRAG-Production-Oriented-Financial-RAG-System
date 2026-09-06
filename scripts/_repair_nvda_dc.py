#!/usr/bin/env python3
"""Targeted, reversible repair of the garbled NVDA segment-revenue table chunk.

Root cause: NVDA_tbl_bf56f12bfafa_0059 was ingested with duplicated column
headers ("Data Center | Data Center | Data Center | $ | 115186 ..."), making
the Data Center FY2025 revenue (115186) unparseable by the LLM even though the
chunk is correctly retrieved. This prepends a clean, faithful caption built
strictly from the numbers already present in the chunk, and stores the original
text in `raw_text_original` for reversibility. No Qdrant re-embedding needed
(retrieval already works; only the MongoDB enrichment text is corrected).
"""
from pymongo import MongoClient

CHUNK_ID = "NVDA_tbl_bf56f12bfafa_0059"

SEGMENTS = [
    ("Data Center", "115186", "47525", "15005"),
    ("Compute", "102196", "38950", "11317"),
    ("Networking", "12990", "8575", "3688"),
    ("Gaming", "11350", "10447", "9067"),
    ("Professional Visualization", "1878", "1553", "1544"),
    ("Automotive", "1694", "1091", "903"),
    ("OEM and Other", "389", "306", "455"),
]

CAPTION = (
    "NVIDIA Revenue by Reportable Segment (in millions of U.S. dollars). "
    "Fiscal years ended January 2025 (FY2025), January 2024 (FY2024), "
    "January 2023 (FY2023).\n"
)
for name, a, b, c in SEGMENTS:
    CAPTION += f"{name}: FY2025 = {a}; FY2024 = {b}; FY2023 = {c}.\n"
CAPTION += "Total revenue: FY2025 = 130497.\n\n"
CAPTION += "--- Original extracted table (contains parsing artifacts) ---\n"


def main():
    coll = MongoClient("mongodb://localhost:27017")["financial_rag"]["raw_chunks"]
    doc = coll.find_one({"chunk_id": CHUNK_ID})
    if doc is None:
        raise SystemExit(f"chunk {CHUNK_ID} not found")
    original = doc.get("raw_text", "")
    if doc.get("raw_text_original"):
        print("Already repaired; skipping (raw_text_original present).")
        return
    new_text = CAPTION + original
    coll.update_one(
        {"chunk_id": CHUNK_ID},
        {
            "$set": {
                "raw_text": new_text,
                "raw_text_original": original,
                "token_count": len(new_text.split()),
            }
        },
    )
    print("Repaired", CHUNK_ID)
    print("New raw_text head:")
    print(new_text[:400])


if __name__ == "__main__":
    main()
