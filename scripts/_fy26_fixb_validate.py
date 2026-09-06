"""Fix B validation: run the REAL _ingest_document with store backends mocked
so the new fiscal_year threading can be verified without mutating the live
Mongo/Qdrant. Returns PASS/FAIL on the stored chunk metadata."""
import sys

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/app")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")

from pathlib import Path
import logging
logging.disable(logging.CRITICAL)

import app.api.main as m
import database_indexer

captured_chunks: list[dict] = []

class FakeEmbeddingEngine:
    def __init__(self, *a, **k):
        pass

    def embed(self, texts, batch_size=8):
        return [[0.0] * 768 for _ in texts]


class FakeMongoIndexer:
    def __init__(self, *a, **k):
        pass

    def upsert_chunks(self, chunks):
        captured_chunks.extend(chunks)
        return len(chunks)

    def close(self):
        pass


class FakeQdrantIndexer:
    def __init__(self, *a, **k):
        pass

    def upsert_vectors(self, chunks, embeddings):
        return len(chunks)

    def close(self):
        pass


database_indexer.EmbeddingEngine = FakeEmbeddingEngine
database_indexer.MongoDBIndexer = FakeMongoIndexer
database_indexer.QdrantIndexer = FakeQdrantIndexer

pdf = Path("/home/youssef/Financial_RAG/data/NVDA/10-K/FY2026/NVDA_AI_Infrastructure_Expansion_FY2026.pdf")
assert pdf.exists(), pdf

m._INGESTION_TASKS.clear()
m._INGESTION_TASKS["fixb-test"] = {"status": "queued"}

result = m._ingest_document(file_path=pdf, ticker="NVDA", fiscal_year="FY2026", task_id="fixb-test")

print("RESULT:", result)
print("captured chunks:", len(captured_chunks))
ok = len(captured_chunks) > 0
for c in captured_chunks:
    fy = c.get("metadata", {}).get("fiscal_year")
    ct = c.get("chunk_type")
    print(f"  chunk_id={c.get('chunk_id')} chunk_type={ct} fiscal_year={fy!r} text_len={len(c.get('text') or '')}")
    if fy != "2026":
        ok = False

print("FIXB_PASS" if ok else "FIXB_FAIL")