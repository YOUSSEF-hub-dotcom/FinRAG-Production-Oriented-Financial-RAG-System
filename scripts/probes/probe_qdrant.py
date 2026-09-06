import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")

from database_indexer import EmbeddingEngine, QdrantIndexer

embed = EmbeddingEngine()
qd = QdrantIndexer(path="data/qdrant_db", collection_name="financial_vectors")
client = qd._connect()

for cid in ["AAPL_tbl_500b35644aa6_0014", "AAPL_tbl_fda9b8bc96ba_0006"]:
    pts = client.retrieve(collection_name="financial_vectors", ids=[cid], with_payload=True, with_vectors=False)
    print(cid, "->", len(pts), "vectors; payload:", pts[0].payload if pts else None)

q = "What were Apple's total net sales in fiscal year 2025?"
vec = embed.embed_single(q)
res = qd.search(vec, 25, None, None, None)
print("\nDirect dense search top 8:")
for r in res:
    print("  ", r["chunk_id"], r.get("contains_table"), round(r["score"], 4))

counts = {"table": 0, "text": 0}
scroll = client.scroll(collection_name="financial_vectors", limit=1000, with_payload=["contains_table"], with_vectors=False)
for p in scroll[0]:
    counts["table" if p.payload.get("contains_table") else "text"] += 1
print("\nQdrant chunk counts:", counts)
