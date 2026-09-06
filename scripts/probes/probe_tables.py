"""List table chunks per ticker/year with sizes and target-number hits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False,
                                enable_hybrid_retrieval=False,
                                enable_post_retrieval=False, top_k=3)

MSFT_TARGETS = ["281724", "101832", "88136", "120810", "106265", "54649", "87464"]
AAPL_TARGETS = ["416161", "391035", "133050", "34550", "93736", "383285", "209586"]

for ticker, year, targets in [("MSFT", "2025", MSFT_TARGETS), ("AAPL", "2025", AAPL_TARGETS), ("AAPL", "2024", AAPL_TARGETS)]:
    chunks = pipeline._mongo_indexer.get_chunks_by_filter({"ticker": ticker, "fiscal_year": str(year)})
    tables = [c for c in chunks if c.get("chunk_type") == "table" and c.get("raw_text")]
    tables.sort(key=lambda c: len(c.get("raw_text", "")), reverse=True)
    print(f"\n=== {ticker} FY{year}: {len(tables)} tables ===")
    for c in tables:
        text = c.get("raw_text", "")
        cid = c.get("chunk_id", "")
        snippet = " ".join(text.split())[:90]
        hits = [t for t in targets if t in text.replace(",", "")]
        print(f"{len(text):6d}ch {cid} hits={hits}")
        print(f"        {snippet}")
