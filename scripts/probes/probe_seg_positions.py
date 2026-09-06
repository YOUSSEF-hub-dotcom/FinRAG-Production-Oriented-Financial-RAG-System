import os, shutil, sys, tempfile
from pathlib import Path

_ROOT = Path.home() / "Financial_RAG"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "1_ingestion"))
from config.logging_config import get_logger
from config.settings import DATA_DIR
get_logger("x")

from html_table_parser import parse_sec_filing
from cleaning import clean_financial_text
from metadata_extractor import extract_metadata
from hybrid_chunker import chunk_document
from database_indexer import EmbeddingEngine, QdrantIndexer

TEST_FILE = DATA_DIR / "AAPL" / "10-K" / "0000320193-25-000079" / "full-submission.txt"
QUERY = "What are Apple's reportable business segments and how did Americas perform in fiscal year 2025?"

parsed = parse_sec_filing(TEST_FILE)
cleaned = clean_financial_text(parsed["text_content"])
base_metadata = extract_metadata(file_path=TEST_FILE, content=parsed["raw_html"],
                                 chunk_text=cleaned[:2000], chunk_index=0)
chunks = chunk_document(text=cleaned, tables=parsed["tables"],
                        file_path=TEST_FILE, metadata_base=base_metadata)
qd = tempfile.mkdtemp(prefix="e2eprobe")
qdrant_indexer = QdrantIndexer(path=qd)
emb_engine = EmbeddingEngine()
embs = emb_engine.embed([c["text"] for c in chunks], batch_size=8)
qdrant_indexer.upsert_vectors(chunks, embs)
qe = emb_engine.embed_single(QUERY)
res = qdrant_indexer.search(qe, top_k=5, ticker="AAPL")
docs = {c["chunk_id"]: c for c in chunks}
segs = ["americas", "europe", "greater china", "japan", "asia pacific"]
for r in res:
    t = docs[r["chunk_id"]].get("text", "")
    low = t.lower()
    pos = {s: low.find(s) for s in segs}
    print(f"{r['chunk_id']} chars={len(t)} seg_positions={pos}")
qdrant_indexer.close()
shutil.rmtree(qd, ignore_errors=True)
