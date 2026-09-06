import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")

from database_indexer import QdrantIndexer
from qdrant_client.models import FieldCondition, Filter, MatchValue

qd = QdrantIndexer(path="data/qdrant_db", collection_name="financial_vectors")
client = qd._connect()

def by_cid(cid):
    f = Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=cid))])
    res = client.scroll(collection_name="financial_vectors", limit=5, scroll_filter=f, with_payload=True, with_vectors=False)
    return res[0]

for cid in ["AAPL_tbl_500b35644aa6_0014", "AAPL_tbl_fda9b8bc96ba_0006", "AAPL_tbl_ca1cd5d2869d_0040"]:
    pts = by_cid(cid)
    print(cid, "->", len(pts), "points")
    if pts:
        p = pts[0]
        print("   id:", p.id)
        print("   payload:", p.payload)

# Scroll a sample of table chunks to see payload/section
f = Filter(must=[FieldCondition(key="contains_table", match=MatchValue(value=True))])
res = client.scroll(collection_name="financial_vectors", limit=5, scroll_filter=f, with_payload=True, with_vectors=False)
print("\nsample table points:")
for p in res[0]:
    print("   ", p.payload.get("chunk_id"), "section=", p.payload.get("section"))
