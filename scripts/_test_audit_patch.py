from run_audit import verify_factual_answer

mongo = [{"raw_text": "Apple Inc 2025 total net sales 416161 million"}]
qdrant_empty = []

# 1) Lock held (live server up): refusal must NOT be blamed on Qdrant Retrieval
s1, f1, r1 = verify_factual_answer(
    "The requested financial information is not available in the provided reports.",
    "AAPL", "2025", "What was Apple's total net revenue in FY2025?",
    mongo, qdrant_empty, qdrant_introspectable=False,
)
print("LOCK-HELD refusal ->", s1, "|", f1, "|", r1[:60])

# 2) Introspectable (server down, Qdrant genuinely empty): blame Qdrant Retrieval
s2, f2, r2 = verify_factual_answer(
    "The requested financial information is not available in the provided reports.",
    "AAPL", "2025", "What was Apple's total net revenue in FY2025?",
    mongo, qdrant_empty, qdrant_introspectable=True,
)
print("INTROSPECTABLE refusal ->", s2, "|", f2, "|", r2[:60])

# 3) Real answer with correct claim + mongo support -> PASS (introspectable irrelevant)
s3, f3, r3 = verify_factual_answer(
    "Apple's total net revenue in FY2025 was $416,161 million.",
    "AAPL", "2025", "What was Apple's total net revenue in FY2025?",
    mongo, qdrant_empty, qdrant_introspectable=False,
)
print("GOOD ANSWER ->", s3, "|", f3, "|", r3[:60])

assert f1 == "Generation", f1
assert f2 == "Qdrant Retrieval", f2
assert s3 == "PASS", s3
print("\nALL ASSERTIONS PASSED")
