"""Step 3: Inspect the Qdrant point for the NVDA FY2026 chunk."""
import sys

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from qdrant_client import QdrantClient
from config.settings import QDRANT_PATH, QDRANT_COLLECTION

client = QdrantClient(path=QDRANT_PATH)
collections = client.get_collections()
print("collections:", [c.name for c in collections.collections])

if QDRANT_COLLECTION in [c.name for c in collections.collections]:
    info = client.get_collection(QDRANT_COLLECTION)
    print("collection points count:", info.points_count)
    # search by payload filter for the specific chunk_id
    from qdrant_client import models
    res = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value="NVDA_txt_4e2da079297d_0000"))]
        ),
        limit=10,
        with_payload=True,
        with_vectors=True,
    )
    points = res[0] if isinstance(res, tuple) else res.points
    print("points found for NVDA_txt_4e2da079297d_0000:", len(points))
    for p in points:
        print("point id:", p.id)
        print("payload:", p.payload)
        vec = p.vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()))
        dim = len(vec) if vec is not None else None
        print("vector dim:", dim, "sample:", vec[:5] if vec else None)
else:
    print("collection not found")
client.close()