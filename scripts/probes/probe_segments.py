import sys
from pathlib import Path

_ROOT = Path.home() / "Financial_RAG"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "1_ingestion"))

from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION, QDRANT_PATH, QDRANT_COLLECTION
from config.logging_config import get_logger

get_logger("x")

QUERY = "What are Apple's reportable business segments and how did Americas perform in fiscal year 2025?"

qdrant = QdrantIndexer(path=QDRANT_PATH, collection_name=QDRANT_COLLECTION)
mongo = MongoDBIndexer(uri=MONGODB_URI, db_name=MONGODB_DB, collection_name=MONGODB_COLLECTION)
emb = EmbeddingEngine()
qe = emb.embed_single(QUERY)
res = qdrant.search(qe, top_k=8, ticker="AAPL")
print("qdrant hits:", len(res))
for r in res:
    cid = r["chunk_id"]
    doc = mongo.get_chunks_by_ids([cid]).get(cid, {})
    text = doc.get("raw_text", "")
    segs = [s for s in ["americas", "europe", "greater china", "japan", "asia pacific"] if s in text.lower()]
    print(f"--- {cid} (score={r['score']:.4f}, chars={len(text)})")
    print("   segments in text:", segs)
    print("   head:", text[:120].replace("\n", " "))
try:
    qdrant.close()
except Exception:
    pass
try:
    mongo.close()
except Exception:
    pass
