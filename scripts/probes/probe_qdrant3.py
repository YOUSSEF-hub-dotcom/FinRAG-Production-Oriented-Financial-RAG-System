import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")

from database_indexer import EmbeddingEngine, QdrantIndexer

embed = EmbeddingEngine()
qd = QdrantIndexer(path="data/qdrant_db", collection_name="financial_vectors")

q = "What were Apple's total net sales in fiscal year 2025?"
vec = embed.embed_single(q)

# dense search restricted to AAPL 2025, top 100
res = qd.search(vec, 100, "AAPL", "2025", None)
print("total dense hits (AAPL 2025):", len(res))
target = "AAPL_tbl_500b35644aa6_0014"
for i, r in enumerate(res):
    if r["chunk_id"] == target:
        print("TARGET at rank", i, "score", round(r["score"], 4))
        break
else:
    print("TARGET not in top 100")
# show how many tables in top 50
tabs = [r for r in res if r.get("contains_table")]
print("tables in top 100:", len(tabs))
for r in res[:10]:
    print("  ", r["chunk_id"], "table" if r.get("contains_table") else "text", round(r["score"], 4))
