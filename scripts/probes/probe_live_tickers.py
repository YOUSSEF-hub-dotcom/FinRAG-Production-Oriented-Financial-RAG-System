"""
Live diagnostic (Windows backend env). Inspect exact ticker values stored in
Qdrant (server + candidate local paths) and MongoDB.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION,
    SUPPORTED_TICKERS, DATA_DIR,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pymongo

print("QDRANT_HOST:", QDRANT_HOST, "PORT:", QDRANT_PORT)
print("QDRANT_COLLECTION:", QDRANT_COLLECTION)
print("DATA_DIR:", DATA_DIR)
print("SUPPORTED_TICKERS:", SUPPORTED_TICKERS)
print("MONGODB_URI:", MONGODB_URI)

def distinct_tickers(qclient, collection):
    distinct = {}
    off = None
    while True:
        pts, off = qclient.scroll(collection_name=collection, limit=2000,
                                  offset=off, with_payload=["ticker"], with_vectors=False)
        for p in pts:
            tk = (p.payload or {}).get("ticker", "<MISSING>")
            distinct[tk] = distinct.get(tk, 0) + 1
        if not off:
            break
    return distinct

# ---- Try Qdrant server ----
print("\n########## QDRANT SERVER ##########")
try:
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    cols = [c.name for c in qc.get_collections().collections]
    print("Server collections:", cols)
    if QDRANT_COLLECTION in cols:
        total = qc.count(collection_name=QDRANT_COLLECTION).count
        print("TOTAL POINTS:", total)
        d = distinct_tickers(qc, QDRANT_COLLECTION)
        print("DISTINCT ticker values:", d)
        print("PER-TICKER MatchValue filter:")
        for tk in SUPPORTED_TICKERS:
            f = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=tk))])
            print(f"   {tk!r:8} ->", qc.count(collection_name=QDRANT_COLLECTION, count_filter=f).count)
    else:
        print("Collection", QDRANT_COLLECTION, "NOT on server")
except Exception as e:
    print("Server connect error:", repr(e))

# ---- Try candidate local file paths ----
candidates = [
    Path(DATA_DIR) / "qdrant_db",
    Path(DATA_DIR) / "qdrant_db_test",
    Path(_PROJECT_ROOT) / "backups" / "qdrant_db_backup_pre_reingest",
]
for cand in candidates:
    print(f"\n########## QDRANT LOCAL PATH: {cand} ##########")
    if not cand.exists():
        print("  (does not exist)")
        continue
    try:
        qc = QdrantClient(path=str(cand))
        cols = [c.name for c in qc.get_collections().collections]
        print("  Local collections:", cols)
        if QDRANT_COLLECTION in cols:
            total = qc.count(collection_name=QDRANT_COLLECTION).count
            print("  TOTAL POINTS:", total)
            d = distinct_tickers(qc, QDRANT_COLLECTION)
            print("  DISTINCT ticker values:", d)
    except Exception as e:
        print("  Local connect error:", repr(e))

# ---- MongoDB ----
print("\n########## MONGODB ##########")
try:
    mc = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    mdb = mc[MONGODB_DB]
    mcol = mdb[MONGODB_COLLECTION]
    print("Total docs:", mcol.count_documents({}))
    print("Distinct ticker values:")
    for row in mcol.aggregate([{"$group": {"_id": "$ticker", "n": {"$sum": 1}}}], allowDiskUse=True):
        print(f"   {row['_id']!r:20} -> {row['n']}")
    print("Per supported ticker:")
    for tk in SUPPORTED_TICKERS:
        print(f"   {tk!r:8} ->", mcol.count_documents({"ticker": tk}))
except Exception as e:
    print("Mongo error:", repr(e))
