import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")
sys.path.insert(0, "src/4_retrieval")
sys.path.insert(0, "src/5_generation")
sys.path.insert(0, "src/3_pre_retrieval")

import asyncio
from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from hybrid_search import HybridSearchEngine

embed = EmbeddingEngine()
qd = QdrantIndexer(path="data/qdrant_db", collection_name="financial_vectors")
mg = MongoDBIndexer()

eng = HybridSearchEngine(qdrant_indexer=qd, mongo_indexer=mg, embed_fn=embed.embed, top_k=25)

TARGET = "AAPL_tbl_500b35644aa6_0014"  # income statement table with 416161

async def main():
    q = "What were Apple's total net sales in fiscal year 2025?"
    res = await eng.asearch([q], {})
    cids = [r["chunk_id"] for r in res]
    print("total:", len(cids))
    print("income-statement table in top25:", TARGET in cids)
    # what table chunks are in top25?
    tabs = [r for r in res if r.get("chunk_type") == "table"]
    print("table chunks in top25:", len(tabs))
    for r in tabs[:10]:
        print("   ", r["chunk_id"], round(r.get("rrf_score", 0), 4))
    # top 5 by rrf
    print("\nTop 5 by rrf:")
    for r in res[:5]:
        print("   ", r["chunk_id"], r.get("chunk_type"), round(r.get("rrf_score", 0), 4))

asyncio.run(main())
