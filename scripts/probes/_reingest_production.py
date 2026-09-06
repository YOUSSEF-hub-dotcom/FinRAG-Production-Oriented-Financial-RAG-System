"""
Production re-ingestion: populates MongoDB + Qdrant with the 6 canonical SEC filings.

Flow per filing: parse -> clean -> metadata -> chunk -> embed (CUDA, batch 8) -> dual-storage upsert.

Usage:
    python _reingest_production.py            # ingest (idempotent after deterministic chunk IDs)
    python _reingest_production.py --reset    # wipe both production stores first, then ingest
"""

import gc
import sys
import time
from pathlib import Path

# --- Path Setup (mirrors test convention) ---
# This script lives at <root>/scripts/probes/_reingest_production.py, so the
# project root is three parents up. Both the project root (for `config`) and
# <root>/src/1_ingestion (for the flat ingestion imports) must be on the path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))

from config.logging_config import get_logger
from config.settings import (
    DATA_DIR,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

from cleaning import clean_financial_text
from database_indexer import EmbeddingEngine, MongoDBIndexer, QdrantIndexer
from html_table_parser import parse_sec_filing
from hybrid_chunker import chunk_document
from metadata_extractor import extract_metadata

logger = get_logger("ingestion.production")

# --- Canonical 6 filings: (ticker, fiscal_year, accession, path) ---
FILINGS = [
    ("AAPL", "2024", "0000320193-24-000123", DATA_DIR / "AAPL" / "10-K" / "0000320193-24-000123" / "full-submission.txt"),
    ("AAPL", "2025", "0000320193-25-000079", DATA_DIR / "AAPL" / "10-K" / "0000320193-25-000079" / "full-submission.txt"),
    ("MSFT", "2024", "0000950170-24-087843", DATA_DIR / "MSFT" / "10-K" / "0000950170-24-087843" / "full-submission.txt"),
    ("MSFT", "2025", "0000950170-25-100235", DATA_DIR / "MSFT" / "10-K" / "0000950170-25-100235" / "full-submission.txt"),
    ("NVDA", "2025", "0001045810-25-000023", DATA_DIR / "NVDA" / "10-K" / "0001045810-25-000023" / "full-submission.txt"),
    ("NVDA", "2026", "0001045810-26-000021", DATA_DIR / "NVDA" / "10-K" / "0001045810-26-000021" / "full-submission.txt"),
]


def reset_stores():
    """Wipe production MongoDB collection and Qdrant collection."""
    logger.info("=== RESET: wiping production stores ===")
    mongo = MongoDBIndexer()
    try:
        mongo.drop_collection()
        logger.info("MongoDB collection dropped: %s/%s.%s", MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION)
    finally:
        mongo.close()

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION in existing:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info("Qdrant collection deleted: %s", QDRANT_COLLECTION)
        else:
            logger.info("Qdrant collection did not exist: %s", QDRANT_COLLECTION)
    finally:
        client.close()


def embed_all(engine: EmbeddingEngine, texts: list[str], batch_size: int = 8) -> list[list[float]]:
    """Embed in batches with a batch_size=1 fallback on CUDA OOM."""
    import torch

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            all_embeddings.extend(engine.embed(batch, batch_size=batch_size))
        except Exception as exc:
            logger.warning("Batch embed failed (%s) — falling back to batch_size=1", exc)
            for t in batch:
                all_embeddings.extend(engine.embed([t], batch_size=1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_embeddings


def main() -> None:
    do_reset = "--reset" in sys.argv

    for ticker, year, _, path in FILINGS:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing filing: {path}")

    if do_reset:
        reset_stores()

    mongo = MongoDBIndexer()
    qdrant = QdrantIndexer()
    engine = EmbeddingEngine()
    qdrant.ensure_collection()

    try:
        logger.info("Pre-ingest counts — Mongo=%d, Qdrant=%d", mongo.count_documents(), qdrant.count_points())

        grand_total = 0
        for ticker, year, accession, path in FILINGS:
            t0 = time.time()
            logger.info("=== [%s] %s (%s) — starting ===", ticker, year, accession)

            parsed = parse_sec_filing(path)
            cleaned = clean_financial_text(parsed["text_content"])
            base_meta = extract_metadata(
                file_path=path,
                content=parsed["raw_html"],
                chunk_text=cleaned[:2000],
                chunk_index=0,
            )
            if base_meta["ticker"] != ticker or base_meta["fiscal_year"] != year:
                raise RuntimeError(
                    f"Metadata mismatch for {ticker} {year}: got {base_meta['ticker']} {base_meta['fiscal_year']}"
                )

            chunks = chunk_document(
                text=cleaned,
                tables=parsed["tables"],
                file_path=path,
                metadata_base=base_meta,
            )
            if not chunks:
                raise RuntimeError(f"No chunks produced for {ticker} {year}")

            texts = [c["text"] for c in chunks]
            embeddings = embed_all(engine, texts, batch_size=8)
            if len(embeddings) != len(chunks):
                raise RuntimeError(f"Embedding count {len(embeddings)} != chunk count {len(chunks)}")

            mongo_count = mongo.upsert_chunks(chunks)
            qdrant_count = qdrant.upsert_vectors(chunks, embeddings)

            elapsed = time.time() - t0
            grand_total += len(chunks)
            n_text = sum(1 for c in chunks if c.get("chunk_type") == "text")
            n_table = len(chunks) - n_text
            logger.info(
                "=== [%s] %s — %d chunks (%d text, %d table) | Mongo=%d Qdrant=%d | %.1fs | cumulative=%d ===",
                ticker, year, len(chunks), n_text, n_table, mongo_count, qdrant_count, elapsed, grand_total,
            )
            gc.collect()

        # --- Final verification ---
        print("\n" + "=" * 70)
        print("FINAL VERIFICATION")
        print("=" * 70)
        for t in ("AAPL", "MSFT", "NVDA"):
            print(f"  {t}: Mongo={mongo.count_documents(t)}, Qdrant={qdrant.count_points(t)}")
        mongo_total = mongo.count_documents()
        qdrant_total = qdrant.count_points()
        print(f"  TOTAL: Mongo={mongo_total}, Qdrant={qdrant_total}")
        print(f"  ALIGNMENT: {'MATCH' if mongo_total == qdrant_total else 'MISMATCH'}")

        # Cross-reference a Qdrant point back to MongoDB (close indexer client first
        # so the raw persistent client can lock the db dir).
        qdrant.close()
        from qdrant_client import QdrantClient

        raw = QdrantClient(path=str(QDRANT_PATH))
        try:
            hits, _ = raw.scroll(QDRANT_COLLECTION, limit=3, with_payload=["chunk_id"])
        finally:
            raw.close()
        if hits:
            sample_cid = hits[0].payload.get("chunk_id")
            doc = mongo.get_chunk(sample_cid)
            found = "FOUND" if doc else "NOT FOUND"
            print(f"  CROSS-REF: qdrant chunk_id={sample_cid} -> mongo {found}")
    finally:
        mongo.close()
        qdrant.close()

    print("=" * 70)


if __name__ == "__main__":
    main()
