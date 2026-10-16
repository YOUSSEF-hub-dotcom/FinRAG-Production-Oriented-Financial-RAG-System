"""
D6-B — Safe replacement of a logical filing's chunks across MongoDB + Qdrant.

This module implements a single, shared, ordered replacement routine used by
every ingestion path that knows a stable ``source_id`` (SEC accession or an
explicit persistent id). The ordering and verification are deliberate:

    1. Validate identity + incoming set (fail closed).
    2. Inspect existing records keyed to the source_id; refuse if any
       well-formed metadata conflicts with the incoming metadata.
    3. Upsert incoming chunks into BOTH stores.
    4. Verify EVERY expected chunk_id is present in BOTH stores — write
       failures / partial writes are never treated as success.
    5. Delete stale chunks, scoped strictly to the approved source_id.
    6. Re-verify the final keyed id sets and report any incomplete convergence.

Deletion is always scoped by ``source_id`` (and, for Qdrant, additionally by the
stale chunk_ids), so unkeyed legacy points and records belonging to other
filings are never removed. Identity is never guessed: a missing/ambiguous
``source_id`` raises :class:`IdentityError` instead of falling back to deletion.
"""

from __future__ import annotations

from typing import Any, Sequence

from config.logging_config import get_logger

logger = get_logger("ingestion.safe_replacement")

UNKNOWN = "UNKNOWN"

# Metadata compared by the conflict gate. A field is "well-formed" when it is
# non-empty and not the placeholder UNKNOWN.
_METADATA_FIELDS = ("ticker", "fiscal_year", "doc_type")


class ReplacementError(RuntimeError):
    """Base class for safe-replacement failures."""


class IdentityError(ReplacementError):
    """Raised when a valid logical-filing identity is not available."""


class MetadataConflictError(ReplacementError):
    """Raised when existing well-formed metadata conflicts with the incoming."""


class WriteVerificationError(ReplacementError):
    """Raised when an expected chunk_id is missing after writing."""


def _is_well_formed(value: Any) -> bool:
    return value is not None and str(value).strip() not in ("", UNKNOWN)


def _validate_source_id(source_id: Any) -> str:
    if not isinstance(source_id, str) or not source_id.strip():
        raise IdentityError(
            "Refusing replacement: source_id is empty (identity unresolved)"
        )
    source_id = source_id.strip()
    if source_id == UNKNOWN:
        raise IdentityError(
            "Refusing replacement: source_id is the UNKNOWN placeholder"
        )
    return source_id


def _incoming_metadata(chunk: dict) -> dict:
    metadata = chunk.get("metadata") or {}
    return {field: metadata.get(field) for field in _METADATA_FIELDS}


def _assert_no_metadata_conflict(
    existing_docs: Sequence[dict], incoming_meta: dict
) -> None:
    """Refuse replacement when an existing well-formed field differs.

    Existing fields that are empty/UNKNOWN never block a replacement; only a
    *well-formed, disagreeing* value does (a signal the source_id may be shared
    by an unrelated filing).
    """
    for doc in existing_docs:
        for field, incoming in incoming_meta.items():
            existing = doc.get(field)
            if _is_well_formed(existing) and str(existing) != str(incoming):
                raise MetadataConflictError(
                    f"Refusing replacement: existing {field}={existing!r} "
                    f"conflicts with incoming {field}={incoming!r}"
                )


def replace_source(
    source_id: str,
    chunks: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    mongo_indexer: Any,
    qdrant_indexer: Any,
) -> dict:
    """Safely replace all chunks for one logical filing (``source_id``).

    Args:
        source_id: Stable logical-filing identity (SEC accession or explicit
            persistent id). Must be non-empty and not ``UNKNOWN``.
        chunks: Incoming chunk dicts (must be non-empty, unique chunk_ids).
        embeddings: Vectors aligned 1:1 with ``chunks``.
        mongo_indexer: Object exposing ``upsert_chunks``,
            ``get_chunks_by_ids``, ``get_chunks_by_source_id``,
            ``get_chunk_ids_by_source_id`` and ``delete_stale_chunks``.
        qdrant_indexer: Object exposing ``upsert_vectors``,
            ``get_chunk_ids_by_source_id`` and ``delete_stale_by_source_id``.

    Returns:
        Summary dict including counts, ``converged`` and any ``unconverged`` ids.

    Raises:
        IdentityError: invalid/empty identity or empty incoming set.
        MetadataConflictError: existing well-formed metadata disagrees.
        WriteVerificationError: an expected chunk_id is missing post-write.
        ReplacementError: duplicate ids or chunk/embedding length mismatch.
    """
    sid = _validate_source_id(source_id)

    if not chunks:
        raise IdentityError(
            "Refusing replacement: incoming chunk set is empty"
        )
    if len(chunks) != len(embeddings):
        raise ReplacementError(
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
        )

    expected_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(expected_ids)) != len(expected_ids):
        raise ReplacementError("Refusing replacement: duplicate chunk_ids in incoming set")
    expected = set(expected_ids)
    incoming_meta = _incoming_metadata(chunks[0])

    # --- 1/2. Inspect current scope + conflict gate (BEFORE any write) --------
    existing_docs = mongo_indexer.get_chunks_by_source_id(sid)
    _assert_no_metadata_conflict(existing_docs, incoming_meta)
    existing_mongo_ids = {
        doc["chunk_id"] for doc in existing_docs if doc.get("chunk_id")
    }
    existing_qdrant_ids = qdrant_indexer.get_chunk_ids_by_source_id(sid)

    # --- 3. Upsert incoming into both stores --------------------------------
    mongo_written = mongo_indexer.upsert_chunks(chunks)
    qdrant_written = qdrant_indexer.upsert_vectors(chunks, embeddings)

    # --- 4. Verify EVERY expected id landed in BOTH stores ------------------
    mongo_present = set(mongo_indexer.get_chunks_by_ids(expected_ids))
    qdrant_present = qdrant_indexer.get_chunk_ids_by_source_id(sid)
    missing_mongo = expected - mongo_present
    missing_qdrant = expected - qdrant_present
    if missing_mongo or missing_qdrant:
        raise WriteVerificationError(
            "Refusing stale deletion: incomplete write "
            f"(missing_mongo={len(missing_mongo)}, "
            f"missing_qdrant={len(missing_qdrant)})"
        )

    # --- 5. Delete stale, scoped strictly to the approved source_id ---------
    stale_mongo = existing_mongo_ids - expected
    stale_qdrant = existing_qdrant_ids - expected
    removed_mongo = mongo_indexer.delete_stale_chunks(sid, sorted(stale_mongo))
    removed_qdrant = qdrant_indexer.delete_stale_by_source_id(sid, sorted(stale_qdrant))

    # --- 6. Re-verify final keyed id sets -----------------------------------
    mongo_final = mongo_indexer.get_chunk_ids_by_source_id(sid)
    qdrant_final = qdrant_indexer.get_chunk_ids_by_source_id(sid)
    missing = expected - (mongo_final & qdrant_final)
    extra = (mongo_final | qdrant_final) - expected
    converged = (
        not missing and not extra and mongo_final == expected and qdrant_final == expected
    )

    result = {
        "source_id": sid,
        "expected": len(expected),
        "mongo_written": mongo_written,
        "qdrant_written": qdrant_written,
        "mongo_present": len(mongo_present),
        "qdrant_present": len(qdrant_present),
        "stale_removed_mongo": removed_mongo,
        "stale_removed_qdrant": removed_qdrant,
        "converged": converged,
        "unconverged": {
            "missing": sorted(missing),
            "extra_mongo": sorted(mongo_final - expected),
            "extra_qdrant": sorted(qdrant_final - expected),
        },
    }

    if not converged:
        logger.warning(
            "Safe replacement for source_id=%s did not fully converge: %s",
            sid,
            result["unconverged"],
        )
    else:
        logger.info(
            "Safe replacement converged for source_id=%s (%d chunks, "
            "%d stale removed)",
            sid,
            len(expected),
            removed_mongo + removed_qdrant,
        )
    return result
