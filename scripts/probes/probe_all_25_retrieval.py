import asyncio
import re
import os
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "1_ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "4_retrieval"))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from hybrid_search import HybridSearchEngine
from post_retrieval import PostRetrievalPipeline

PH = re.compile(r"%%TABLE_(\d+)%%")

def make_chunk_id(ticker, year, ctype, source, idx):
    h = hashlib.md5(f"{ticker}:{year}:{ctype}:{source}:{idx}".encode()).hexdigest()[:12]
    return f"{ticker}_{ctype}_{h}_{idx:04d}"

def norm_num(s):
    return re.sub(r"[^0-9]", "", s)

def extract_ground_truth_numbers(gt):
    nums = []
    for m in re.finditer(r"\$?([0-9][0-9,]*\.?[0-9]*)", gt):
        raw = m.group(1).replace(",", "")
        if raw in ("2023", "2024", "2025", "2026"):
            continue
        if len(raw) >= 5 and raw.isdigit():
            nums.append(raw)
        elif raw.replace(".", "").isdigit() and "." in raw and len(raw) >= 3:
            nums.append(raw)
    return sorted(set(nums), key=len, reverse=True)

async def main():
    df = pd.read_csv("artifacts/test_dataset.csv")
    mongo = MongoDBIndexer()
    qdrant = QdrantIndexer()
    emb = EmbeddingEngine()
    hybrid = HybridSearchEngine(qdrant_indexer=qdrant, mongo_indexer=mongo, embed_fn=emb.embed, top_k=25)
    post = PostRetrievalPipeline()

    tbl_map = {}
    for d in mongo.get_chunks_by_filter({"chunk_type": "table"}, limit=100000):
        tbl_map[d["chunk_id"]] = d["raw_text"]

    stats = {"hybrid_has_num": 0, "inline_has_num": 0, "post_has_num": 0, "total": 0}
    rows = []
    for _, r in df.iterrows():
        q = str(r["question"]); gt = str(r["ground_truth"])
        nums = extract_ground_truth_numbers(gt)
        # find ticker/year from question
        ticker = None
        for t in ("AAPL", "Apple", "MSFT", "Microsoft", "NVDA", "NVIDIA"):
            if t.lower() in q.lower():
                ticker = {"appl": "AAPL", "apple": "AAPL", "msft": "MSFT", "microsoft": "MSFT", "nvda": "NVDA", "nvidia": "NVDA"}[t.lower()]
                break
        years = [y for y in ("2023","2024","2025","2026") if y in q]
        flt = {}
        if ticker: flt["ticker"] = ticker
        if years: flt["fiscal_year"] = years[0]

        results = await hybrid.asearch([q], flt)
        combined = " ".join(res.get("text","") for res in results)
        hybrid_hit = any(n in combined.replace(",","") for n in nums)

        # simulate inline resolution
        resolved = []
        for res in results:
            text = res.get("text","")
            if "%%TABLE_" in text:
                src = res.get("metadata",{}).get("source_file")
                tk = res.get("metadata",{}).get("ticker")
                yr = res.get("metadata",{}).get("fiscal_year")
                def repl(m):
                    n = int(m.group(1))
                    cid = make_chunk_id(tk, yr, "tbl", src, n)
                    return tbl_map.get(cid, m.group(0))
                text = PH.sub(repl, text)
            resolved.append({**res, "text": text})
        resolved_text = " ".join(res.get("text","") for res in resolved)
        inline_hit = any(n in resolved_text.replace(",","") for n in nums)

        post_res = await post.aprocess(q, results)
        post_text = " ".join(p.get("text","") for p in post_res)
        post_hit = any(n in post_text.replace(",","") for n in nums)
        post_table_count = sum(1 for p in post_res if p.get("chunk_type")=="table")

        stats["total"] += 1
        for key, hit in (("hybrid_has_num", hybrid_hit), ("inline_has_num", inline_hit), ("post_has_num", post_hit)):
            if hit: stats[key] += 1

        rows.append({
            "q": q[:55], "ticker": ticker, "year": years[0] if years else "?",
            "nums": nums,
            "hybrid_hit": hybrid_hit, "inline_hit": inline_hit,
            "post_hit": post_hit, "post_tables": post_table_count,
            "top3_types": [r.get("chunk_type") for r in results[:3]],
            "top3_scores": [round(r.get("rrf_score",0),4) for r in results[:3]],
        })
        print(f"{'Y' if post_hit else 'N'} post  {'Y' if inline_hit else 'N'} inline {'Y' if hybrid_hit else 'N'} hybrid | tbls in post={post_table_count} | {q[:55]} | nums={nums}")

    print("\n=== SUMMARY ===")
    print(stats)
    for key in ("hybrid_has_num", "inline_has_num", "post_has_num"):
        print(f"{key}: {stats[key]}/{stats['total']}")

asyncio.run(main())
