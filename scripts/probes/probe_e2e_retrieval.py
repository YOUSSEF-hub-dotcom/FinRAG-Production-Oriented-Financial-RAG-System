import os, shutil, sys, tempfile
from pathlib import Path

_ROOT = Path.home() / "Financial_RAG"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_ROOT / "src" / "5_generation"))

from config.logging_config import get_logger
from config.settings import DATA_DIR
get_logger("x")

from html_table_parser import parse_sec_filing
from cleaning import clean_financial_text
from metadata_extractor import extract_metadata
from hybrid_chunker import chunk_document
from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer

TEST_FILE = DATA_DIR / "AAPL" / "10-K" / "0000320193-25-000079" / "full-submission.txt"
QUERY = "What are Apple's reportable business segments and how did Americas perform in fiscal year 2025?"

parsed = parse_sec_filing(TEST_FILE)
cleaned = clean_financial_text(parsed["text_content"])
base_metadata = extract_metadata(file_path=TEST_FILE, content=parsed["raw_html"],
                                 chunk_text=cleaned[:2000], chunk_index=0)
chunks = chunk_document(text=cleaned, tables=parsed["tables"],
                        file_path=TEST_FILE, metadata_base=base_metadata)
print("TOTAL CHUNKS:", len(chunks))

qd = tempfile.mkdtemp(prefix="e2eprobe")
qdrant_indexer = QdrantIndexer(path=qd)
emb_engine = EmbeddingEngine()
embs = emb_engine.embed([c["text"] for c in chunks], batch_size=8)
qdrant_indexer.upsert_vectors(chunks, embs)

qe = emb_engine.embed_single(QUERY)
res = qdrant_indexer.search(qe, top_k=8, ticker="AAPL")
print("top-8 qdrant hits")
table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
print("TABLE CHUNKS:", len(table_chunks))
for c in table_chunks:
    t = c.get("text", "")
    segs = [s for s in ["americas", "europe", "greater china", "japan", "asia pacific"] if s in t.lower()]
    print(f"  table {c['chunk_id']} chars={len(t)} segs={segs} head={t[:80]!r}")

total_chars = 0
for rank, r in enumerate(res):
    cid = r["chunk_id"]
    docs = {c["chunk_id"]: c for c in chunks}
    doc = docs.get(cid, {})
    t = doc.get("text", "")
    segs = [s for s in ["americas", "europe", "greater china", "japan", "asia pacific"] if s in t.lower()]
    print(f"--- rank{rank} {cid} score={r['score']:.4f} chars={len(t)} segs={segs}")
    total_chars += len(t)

print("TOTAL top-8 chars:", total_chars)

# where does the segment table rank?
seg_table_id = None
for c in chunks:
    t = c.get("text", "")
    segs = [s for s in ["americas", "europe", "greater china", "japan", "asia pacific"] if s in t.lower()]
    if len(segs) == 5 and c.get("chunk_type") == "table":
        seg_table_id = c["chunk_id"]
print("SEGMENT TABLE ID:", seg_table_id)

# cosine rank of segment table among all
import numpy as np
seg_emb = None
for c, e in zip(chunks, embs):
    if c["chunk_id"] == seg_table_id:
        seg_emb = e
        break
if seg_emb is not None and seg_emb is not None:
    scores = [float(np.dot(se, qe) / (np.linalg.norm(se) * np.linalg.norm(qe) + 1e-12)) for se in embs]
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    rank_pos = [i for i in order if chunks[i]["chunk_id"] == seg_table_id][0]
    print("SEGMENT TABLE cosine rank:", rank_pos, "of", len(chunks))
    # print the top 15 ids
    for i in order[:15]:
        print("   ", chunks[i]["chunk_id"], chunks[i].get("chunk_type"), round(scores[i], 4))
try:
    qdrant_indexer.close()
except Exception:
    pass
shutil.rmtree(qd, ignore_errors=True)
