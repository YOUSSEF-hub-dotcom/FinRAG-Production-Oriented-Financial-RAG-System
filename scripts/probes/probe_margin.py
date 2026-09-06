import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False,
                                enable_hybrid_retrieval=False,
                                enable_post_retrieval=False, top_k=3)
indexer = pipeline._mongo_indexer
IDS = ["AAPL_tbl_1a9ec4d66a80_0008", "AAPL_tbl_a48d2f2f32fc_0009"]

chunks = indexer.get_chunks_by_ids(IDS)
by_id = chunks if isinstance(chunks, dict) else {c["chunk_id"]: c for c in chunks}

for cid in IDS:
    c = by_id.get(cid)
    if not c:
        print(f"--- {cid}: NOT FOUND")
        continue
    text = c["raw_text"]
    print(f"--- {cid} len={len(text)}")
    print(text[:1600])
    print("===== decimal percents:", re.findall(r"\d+\.\d+\s*%", text))
    print("===== gross margin count:", text.lower().count("gross margin"))
    print()
