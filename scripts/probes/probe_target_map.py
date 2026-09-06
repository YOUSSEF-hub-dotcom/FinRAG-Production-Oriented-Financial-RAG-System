"""Map every ground-truth figure to the table chunks containing it (current DB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False,
                                enable_hybrid_retrieval=False,
                                enable_post_retrieval=False, top_k=3)

TARGETS = {
    "AAPL": {
        "2025": ["416161", "133050", "209586", "46.9", "34550", "391035"],
        "2024": ["93736", "391035", "383285", "6.08", "46.2", "416161"],
    },
    "MSFT": {
        "2025": ["281724", "101832", "88136", "120810", "106265", "54649", "87464", "23455", "13.64"],
    },
}

for ticker, years in TARGETS.items():
    for year, targets in years.items():
        print(f"\n=== {ticker} FY{year} ===")
        chunks = pipeline._mongo_indexer.get_chunks_by_filter({"ticker": ticker, "fiscal_year": str(year)})
        tables = [c for c in chunks if c.get("chunk_type") == "table" and c.get("raw_text")]
        for target in targets:
            found = []
            for c in tables:
                text = c.get("raw_text", "").replace(",", "")
                if target in text:
                    found.append(f"{c.get('chunk_id')}[{len(c.get('raw_text',''))}]")
            print(f"  {target!r}: {found if found else 'NOT FOUND'}")
