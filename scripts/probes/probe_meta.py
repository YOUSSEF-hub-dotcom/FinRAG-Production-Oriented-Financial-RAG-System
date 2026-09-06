"""Inspect metadata of the retrieved text chunk and the filter behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False,
                                enable_hybrid_retrieval=False,
                                enable_post_retrieval=False, top_k=3)

chunk = pipeline._mongo_indexer.get_chunk("MSFT_txt_e6952453a64e_0094")
print("TEXT CHUNK:", chunk.get("chunk_id"))
print("  ticker:", chunk.get("ticker"), "| fiscal_year:", repr(chunk.get("fiscal_year")),
      "| source_file:", chunk.get("source_file"), "| chunk_type:", chunk.get("chunk_type"))
print("  meta keys:", sorted(chunk.keys())[:40])

# Try filter with the chunk's actual fiscal_year value
fy = chunk.get("fiscal_year")
for val in [fy, str(fy)]:
    if val is None:
        continue
    chunks = pipeline._mongo_indexer.get_chunks_by_filter({"ticker": "MSFT", "fiscal_year": val})
    years = sorted({repr(c.get("fiscal_year")) for c in chunks})
    srcs = sorted({str(c.get("source_file"))[:60] for c in chunks})
    print(f"filter fiscal_year={val!r}: {len(chunks)} chunks, years={years}")
    print(f"  source_files={srcs}")

# Distinct fiscal_year values stored for MSFT
import pymongo
coll = pipeline._mongo_indexer._collection if hasattr(pipeline._mongo_indexer, "_collection") else None
print("distinct MSFT years:", pipeline._mongo_indexer.get_collection().distinct("fiscal_year", {"ticker": "MSFT"}) if hasattr(pipeline._mongo_indexer, "get_collection") else "n/a")
