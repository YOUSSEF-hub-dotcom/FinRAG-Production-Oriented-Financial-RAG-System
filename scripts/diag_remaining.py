import re
from pymongo import MongoClient
import sys
sys.path.insert(0, "/home/youssef/Financial_RAG")
import run_audit as RA

mc = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=5000)
coll = mc["financial_rag"]["raw_chunks"]

def find_in_mongo(ticker, year, needle):
    docs = list(coll.find({"ticker": ticker, "fiscal_year": year}))
    hits = []
    for d in docs:
        rt = d.get("raw_text","")
        if needle.lower() in rt.lower():
            hits.append(d["chunk_id"])
    return hits, len(docs)

# AAPL 34,550 and 111,482
for needle in ["34,550","34550","111,482","111482"]:
    hits, total = find_in_mongo("AAPL","2025", needle)
    print(f"AAPL 2025 needle '{needle}': {len(hits)}/{total} hits, e.g. {hits[:2]}")

# MSFT 45.6
for needle in ["45.6","45.63","45.6%","45.63%"]:
    hits, total = find_in_mongo("MSFT","2025", needle)
    print(f"MSFT 2025 needle '{needle}': {len(hits)}/{total} hits")

# Check all numbers in AAPL 2025 docs
docs = list(coll.find({"ticker":"AAPL","fiscal_year":"2025"}))
all_text = " ".join(d.get("raw_text","") for d in docs)
nums = re.findall(r"\d[\d,]*\.?\d*", all_text)
# find R&D and cash flow numbers
for pat in [r"Research and Development[^0-9]*([\d,]+)", r"cash flow[^0-9]*([\d,]+)"]:
    m = re.search(pat, all_text, re.I)
    print(f"pat {pat}: {m.group(1) if m else 'not found'}")

# Check claim extraction for the PARTIAL answers
samples = [
    ("Apple spent 34550 on research and development in FY2025.", "AAPL 34550"),
    ("Apple's cash flow from operations in FY2025 was 111,482 (in millions).", "AAPL 111482"),
    ("Microsoft's operating margin for fiscal year 2025 was approximately 45.6%.", "MSFT 45.6%"),
    ("Microsoft's operating margin for fiscal year 2025 was approximately 45.63%.", "MSFT 45.63%"),
]
for ans, label in samples:
    claims = RA._extract_financial_claims(ans)
    print(f"\n{label}: claims={claims}")
    for c in claims:
        num = RA._claim_numeric_value(c)
        print(f"  claim '{c}' -> num '{num}' -> float {float(num) if num else None}")

# Check cross-ticker claim count for a typical compare answer
compare_ans = "Apple total net revenue $416,161 million, Microsoft $281,724 million, NVIDIA $130,497 million. Operating margins: Apple 31.96%, Microsoft 45.6%, NVIDIA 74.9%."
claims = RA._extract_financial_claims(compare_ans)
print(f"\nCompare answer claims ({len(claims)}): {claims}")

# Check memory answers from report
mem_answers = [
    ("Memory first company -> Apple (FAIL 0/1)", "Microsoft spent $32,488 million on Research and Development in FY2025.", "ALL"),
    ("Memory second -> Microsoft (FAIL 0/1)", "Microsoft spent $29,510 million on Research and Development in FY2024.", "ALL"),
]
for label, ans, ticker in mem_answers:
    claims = RA._extract_financial_claims(ans)
    # simulate verify with ALL ticker (should use live sources)
    # For ALL, evidence_docs is built from SUPPORTED_TICKERS each 10, but verify skips broadening for ALL
    docs_all = []
    for tk in ["AAPL","MSFT","NVDA"]:
        docs_all.extend(list(coll.find({"ticker":tk,"fiscal_year":"2025"}).limit(10)))
    st, stage, rc = RA.verify_factual_answer(ans, "ALL", "2025", "q", docs_all, [], True, live_sources=[])
    print(f"\n{label}: ans='{ans[:60]}' claims={claims} -> {st} {stage} {rc[:80]}")

# Check Amazon negative test
print("\n--- Amazon negative test ---")
# Amazon should have no docs
hits, total = find_in_mongo("AMZN","2025","")
print(f"AMZN docs: {len(list(coll.find({'ticker':'AMZN'})))}")
# Verify logic for Amazon
docs_empty = []
st, stage, rc = RA.verify_factual_answer("not available", "ALL", "2025", "q", [], [], True, live_sources=[])
print(f"Amazon not available with empty docs -> {st} {stage}")

# Check Qdrant direct open
from qdrant_client import QdrantClient
try:
    qc = QdrantClient(path="/home/youssef/Financial_RAG/data/qdrant_db")
    print("Qdrant direct open: success, count", qc.count(collection_name="financial_vectors").count)
    print("qdrant_introspectable:", RA.qdrant_introspectable())
except Exception as e:
    print("Qdrant direct open failed:", e)
    print("qdrant_introspectable:", RA.qdrant_introspectable())
