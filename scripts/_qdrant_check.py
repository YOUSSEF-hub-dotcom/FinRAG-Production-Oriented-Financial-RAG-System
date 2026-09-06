from qdrant_client import QdrantClient
import config.settings as s
print("QDRANT_PATH =", s.QDRANT_PATH)
print("QDRANT_COLLECTION =", s.QDRANT_COLLECTION)
c = QdrantClient(path=s.QDRANT_PATH)
cols = c.get_collections().collections
print("collections:", [x.name for x in cols])
name = s.QDRANT_COLLECTION
try:
    info = c.get_collection(name)
    print("status:", info.status)
    print("vectors size:", info.config.params.vectors.size, "distance:", info.config.params.vectors.distance)
    cnt = c.count(name).count
    print("POINT COUNT =", cnt)
except Exception as e:
    print("collection err:", repr(e))
    cnt = 0
if cnt:
    # sample payloads
    pts = c.scroll(name, limit=5, with_payload=True, with_vectors=False)[0]
    for p in pts:
        pl = p.payload
        print("payload sample:", {k: pl.get(k) for k in ("ticker","fiscal_year","section","chunk_type","source_file")})
    # count per ticker/fy in qdrant
    from collections import Counter
    cc = Counter()
    off=None
    while True:
        pts,off = c.scroll(name, limit=256, with_payload=True, with_vectors=False, offset=off)
        for p in pts:
            pl=p.payload or {}
            cc[(pl.get("ticker"), str(pl.get("fiscal_year")))] += 1
        if off is None or not pts: break
    print("Qdrant per (ticker,fy):", dict(cc))
