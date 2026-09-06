"""Offsets of target figures within candidate tables (to pick safe truncation caps)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False,
                                enable_hybrid_retrieval=False,
                                enable_post_retrieval=False, top_k=3)

CASES = {
    "AAPL": {
        "2025": [("AAPL_tbl_fe088c5347c1_0014", ["416161", "391035", "133050", "34550", "93736"]),
                 ("AAPL_tbl_be747ab1a146_0019", ["209586", "416161"]),
                 ("AAPL_tbl_a48d2f2f32fc_0009", ["46.9", "46.2"]),
                 ("AAPL_tbl_4f56220c5753_0040", ["416161", "133050", "34550", "earnings per share", "net income", "total net sales"]),
                 ("AAPL_tbl_6743efffb92e_0007", ["209586"])],
        "2024": [("AAPL_tbl_d6ecc59ab1d8_0014", ["391035", "383285", "93736", "6.08", "earnings per share"]),
                 ("AAPL_tbl_8c4438369b68_0020", ["93736", "6.08", "391035", "total net sales", "net income"]),
                 ("AAPL_tbl_fd39e623d5c3_0009", ["46.2"])],
    },
    "MSFT": {
        "2025": [("MSFT_tbl_e79b44fb6a3b_0023", ["281724", "101832", "88136", "13.64"]),
                 ("MSFT_tbl_8840639df2be_0013", ["281724", "101832", "13.64"]),
                 ("MSFT_tbl_eec717e75be0_0071", ["120810", "106265", "54649", "87464", "gaming", "intelligent cloud", "total revenues"]),
                 ("MSFT_tbl_04a4b689267a_0073", ["23455", "gaming", "more personal computing"]),
                 ("MSFT_tbl_9a98a855fb58_0026", ["101832", "88136", "13.64", "earnings per share"])],
    },
}

for ticker, years in CASES.items():
    for year, specs in years.items():
        print(f"\n=== {ticker} FY{year} ===")
        chunks = {c.get("chunk_id"): c for c in pipeline._mongo_indexer.get_chunks_by_filter({"ticker": ticker, "fiscal_year": str(year)})}
        for cid, targets in specs:
            c = chunks.get(cid)
            if not c:
                print(f"  {cid}: NOT PRESENT")
                continue
            text = c.get("raw_text", "")
            low = text.lower()
            print(f"  {cid} [{len(text)}ch]")
            for t in targets:
                off = text.find(t) if t.isdigit() or "." in t else low.find(t.lower())
                if off < 0:
                    off = text.replace(",", "").find(t)
                print(f"      {t!r}: offset={off}")
