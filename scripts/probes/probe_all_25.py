import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")
sys.path.insert(0, "src/4_retrieval")

import asyncio
import re
import pandas as pd

from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from hybrid_search import HybridSearchEngine

embed = EmbeddingEngine()
qd = QdrantIndexer(path="data/qdrant_db", collection_name="financial_vectors")
mg = MongoDBIndexer()
eng = HybridSearchEngine(qdrant_indexer=qd, mongo_indexer=mg, embed_fn=embed.embed, top_k=25)

df = pd.read_csv("artifacts/test_dataset.csv")

def has_numeric_ground_truth(gt: str) -> bool:
    return bool(re.search(r"\d[\d,.]*", gt))

async def main():
    results = []
    for i, row in df.iterrows():
        q = str(row["question"])
        gt = str(row["ground_truth"])
        res = await eng.asearch([q], {})
        tables = [r for r in res if r.get("chunk_type") == "table"]
        # does any retrieved chunk contain a number present in ground truth?
        nums = set(re.findall(r"\d{4,6}(?:\.\d+)?", gt.replace(",", "")))
        hit = 0
        for r in res:
            t = (r.get("text") or "").replace(",", "")
            for n in nums:
                if n in t:
                    hit += 1
                    break
        results.append((i + 1, q[:50], len(res), len(tables), len(nums), hit))
        print(f"Q{i+1:02d} top25={len(res):2d} tables={len(tables):2d} gtNums={len(nums)} numHit={hit} | {q[:45]}")

    total_tables = sum(r[3] for r in results)
    total_hits = sum(r[5] for r in results)
    print("\nTOTAL: tables in top25 = %d, queries with number-hit = %d/25" % (total_tables, total_hits))

asyncio.run(main())
