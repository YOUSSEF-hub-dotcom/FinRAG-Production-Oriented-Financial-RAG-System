"""
D6-C — Dry-run reconciliation and legacy-backfill assessment (READ-ONLY).

Compares MongoDB and Qdrant records and reports, without mutating anything:

  * records/points that carry a valid ``source_id`` (keyed) vs. legacy (unkeyed);
  * chunk_ids present in only one store (Mongo-only / Qdrant-only);
  * duplicate, ambiguous, or conflicting source identities and metadata;
  * candidate legacy records that could be backfilled from reliable provenance
    (the other store's keyed ``source_id`` for the same chunk_id, or a curated
    ``(ticker, fiscal_year, doc_type) -> accession`` map), split into
    ``confirmed`` and ``uncertain``.

This module never writes, deletes, or assigns ``source_id``. Applying a
backfill is a separate, explicitly-approved phase (D6-C apply) and is NOT
implemented here.

CLI (read-only):
    python reconcile_identity.py [--provenance map.json] [--json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Allow running as a standalone script from any CWD:
#   python src/1_ingestion/reconcile_identity.py [--json]
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.logging_config import get_logger

logger = get_logger("ingestion.reconcile")

UNKNOWN = "UNKNOWN"
_METADATA_FIELDS = ("ticker", "fiscal_year", "doc_type")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_valid_source_id(value: Any) -> bool:
    """A valid identity is a non-empty string that is not the UNKNOWN placeholder."""
    text = _norm(value)
    return text is not None and text != UNKNOWN


def _provenance_key(ticker: Any, fiscal_year: Any, doc_type: Any) -> tuple:
    return (
        (_norm(ticker) or "").upper(),
        _norm(fiscal_year) or "",
        (_norm(doc_type) or "").upper(),
    )


def _has_complete_metadata(record: Mapping) -> bool:
    return all(
        _norm(record.get(field)) not in (None, UNKNOWN) for field in _METADATA_FIELDS
    )


def normalize_provenance(
    raw: Mapping[Any, Any] | None,
) -> dict[tuple, list[str]]:
    """Normalize a provenance map into ``{(TICKER, YEAR, TYPE): [source_id, ...]}``.

    Accepts either ``{(ticker, year, doc_type): source_id}`` tuples/strings, a
    ``"TICKER|YEAR|TYPE"`` string key, or a list value for ambiguous keys.
    """
    normalized: dict[tuple, list[str]] = {}
    if not raw:
        return normalized
    for key, value in raw.items():
        if isinstance(key, (tuple, list)) and len(key) == 3:
            norm_key = _provenance_key(key[0], key[1], key[2])
        elif isinstance(key, str) and key.count("|") == 2:
            norm_key = _provenance_key(*key.split("|"))
        else:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        cleaned = [str(v).strip() for v in candidates if _norm(v)]
        if cleaned:
            normalized.setdefault(norm_key, [])
            for sid in cleaned:
                if sid not in normalized[norm_key]:
                    normalized[norm_key].append(sid)
    return normalized


def build_provenance_from_filings(entries: Iterable[Sequence[Any]]) -> dict[tuple, list[str]]:
    """Build provenance from ``(ticker, fiscal_year, accession)`` rows.

    Two rows sharing a ``(ticker, fiscal_year, doc_type)`` key would be
    ambiguous; here ``doc_type`` defaults to ``10-K`` (the canonical corpus),
    callers can extend the map manually for other form types.
    """
    raw: dict[tuple, list[str]] = {}
    for ticker, fiscal_year, accession in entries:
        key = _provenance_key(ticker, fiscal_year, "10-K")
        raw.setdefault(key, []).append(accession)
    return raw


def _classify_legacy_by_provenance(
    record: Mapping,
    provenance: Mapping[tuple, list[str]],
) -> tuple[str, str | None, str | None]:
    """Return ``(status, source_id, reason)`` for one legacy record.

    status ∈ {"confirmed", "ambiguous", "uncertain"}.
    """
    if not _has_complete_metadata(record):
        return "uncertain", None, "incomplete_metadata"
    key = _provenance_key(
        record.get("ticker"), record.get("fiscal_year"), record.get("doc_type")
    )
    entries = provenance.get(key)
    if not entries:
        return "uncertain", None, "no_provenance"
    unique = list(dict.fromkeys(entries))
    if len(unique) == 1:
        return "confirmed", unique[0], None
    return "ambiguous", None, "ambiguous_provenance"


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

def reconcile(
    mongo_indexer: Any,
    qdrant_indexer: Any,
    provenance: Mapping[Any, Any] | None = None,
    sample_limit: int = 20,
) -> dict:
    """Reconcile MongoDB against Qdrant. READ-ONLY — never mutates either store.

    Args:
        mongo_indexer: exposes ``iter_all_chunks()``.
        qdrant_indexer: exposes ``iter_all_points()``.
        provenance: optional curated ``(ticker, year, type) -> accession`` map.
        sample_limit: max ids retained per category in the report.

    Returns:
        A structured report dict (counts, categories, evidence, proposed actions).
    """
    prov = normalize_provenance(provenance)

    # --- Load both stores (read-only) ---------------------------------------
    mongo_records = list(mongo_indexer.iter_all_chunks())
    mongo_missing_chunk_id = [d for d in mongo_records if not _norm(d.get("chunk_id"))]
    mongo: dict[str, dict] = {
        str(d["chunk_id"]): d for d in mongo_records if _norm(d.get("chunk_id"))
    }

    qdrant_records = list(qdrant_indexer.iter_all_points())
    qdrant_missing_chunk_id = 0
    qdrant: dict[str, dict] = {}
    qdrant_duplicate_chunk_ids: list[str] = []
    for point in qdrant_records:
        chunk_id = _norm((point.get("payload") or {}).get("chunk_id"))
        if not chunk_id:
            qdrant_missing_chunk_id += 1
            continue
        if chunk_id in qdrant:
            qdrant_duplicate_chunk_ids.append(chunk_id)
        qdrant[chunk_id] = point.get("payload") or {}

    mongo_ids = set(mongo)
    qdrant_ids = set(qdrant)

    mongo_keyed = {cid for cid in mongo_ids if is_valid_source_id(mongo[cid].get("source_id"))}
    mongo_legacy = mongo_ids - mongo_keyed
    qdrant_keyed = {cid for cid in qdrant_ids if is_valid_source_id(qdrant[cid].get("source_id"))}
    qdrant_legacy = qdrant_ids - qdrant_keyed

    paired = mongo_ids & qdrant_ids
    mongo_only = mongo_ids - qdrant_ids
    qdrant_only = qdrant_ids - mongo_ids

    # --- Conflicts: same source_id, inconsistent well-formed metadata -------
    sid_metadata: dict[str, set] = {}
    for cid in mongo_keyed:
        doc = mongo[cid]
        sid = str(doc["source_id"]).strip()
        sid_metadata.setdefault(sid, set()).add(
            _provenance_key(doc.get("ticker"), doc.get("fiscal_year"), doc.get("doc_type"))
        )
    sid_metadata_conflicts = {
        sid: [list(t) for t in tuples]
        for sid, tuples in sid_metadata.items()
        if len(tuples) > 1
    }

    # --- Conflicts: paired chunk with differing source_id across stores -----
    source_id_mismatch: list[dict] = []
    for cid in paired:
        m_sid = _norm(mongo[cid].get("source_id"))
        q_sid = _norm(qdrant[cid].get("source_id"))
        if m_sid and q_sid and m_sid != q_sid:
            source_id_mismatch.append({"chunk_id": cid, "mongo": m_sid, "qdrant": q_sid})

    # --- Backfill assessment (per store, legacy only) -----------------------
    confirmed: list[dict] = []
    ambiguous: list[dict] = []
    uncertain: list[dict] = []

    def _assess(store: str, cid: str, record: Mapping, other_keyed: dict) -> None:
        other_sid = other_keyed.get(cid)
        if other_sid:
            confirmed.append(
                {"chunk_id": cid, "store": store, "source_id": other_sid,
                 "evidence": "other_store_keyed"}
            )
            return
        status, sid, reason = _classify_legacy_by_provenance(record, prov)
        if status == "confirmed":
            confirmed.append(
                {"chunk_id": cid, "store": store, "source_id": sid,
                 "evidence": "provenance_map"}
            )
        elif status == "ambiguous":
            ambiguous.append({"chunk_id": cid, "store": store, "reason": reason})
        else:
            uncertain.append({"chunk_id": cid, "store": store, "reason": reason})

    mongo_other_keyed = {cid: str(mongo[cid]["source_id"]).strip() for cid in mongo_keyed}
    qdrant_other_keyed = {cid: str(qdrant[cid]["source_id"]).strip() for cid in qdrant_keyed}

    for cid in sorted(mongo_legacy):
        _assess("mongo", cid, mongo[cid], qdrant_other_keyed)
    for cid in sorted(qdrant_legacy):
        _assess("qdrant", cid, qdrant[cid], mongo_other_keyed)

    def _sample(rows: list) -> list:
        return rows[:sample_limit]

    result = {
        "totals": {
            "mongo": len(mongo_records),
            "qdrant": len(qdrant_records),
            "mongo_indexed": len(mongo),
            "qdrant_indexed": len(qdrant),
        },
        "anomalies": {
            "mongo_missing_chunk_id": len(mongo_missing_chunk_id),
            "qdrant_missing_chunk_id": qdrant_missing_chunk_id,
            "qdrant_duplicate_chunk_ids": len(qdrant_duplicate_chunk_ids),
            "qdrant_duplicate_sample": _sample(qdrant_duplicate_chunk_ids),
        },
        "identity": {
            "mongo_keyed": len(mongo_keyed),
            "mongo_legacy": len(mongo_legacy),
            "qdrant_keyed": len(qdrant_keyed),
            "qdrant_legacy": len(qdrant_legacy),
        },
        "presence": {
            "paired": len(paired),
            "mongo_only": len(mongo_only),
            "qdrant_only": len(qdrant_only),
            "mongo_only_sample": _sample(sorted(mongo_only)),
            "qdrant_only_sample": _sample(sorted(qdrant_only)),
        },
        "conflicts": {
            "sid_metadata_conflicts": sid_metadata_conflicts,
            "source_id_mismatch": {
                "count": len(source_id_mismatch),
                "sample": _sample(source_id_mismatch),
            },
        },
        "backfill": {
            "confirmed_count": len(confirmed),
            "ambiguous_count": len(ambiguous),
            "uncertain_count": len(uncertain),
            "confirmed_sample": _sample(confirmed),
            "ambiguous_sample": _sample(ambiguous),
            "uncertain_sample": _sample(uncertain),
            "provenance_keys": len(prov),
        },
        "proposed_actions": _proposed_actions(
            mongo_only, qdrant_only, confirmed, ambiguous, uncertain, sid_metadata_conflicts,
            source_id_mismatch,
        ),
    }

    logger.info(
        "Reconcile (dry-run): mongo=%d qdrant=%d paired=%d confirmed_backfill=%d "
        "uncertain=%d ambiguous=%d",
        len(mongo), len(qdrant), len(paired), len(confirmed), len(uncertain), len(ambiguous),
    )
    return result


def _proposed_actions(
    mongo_only: set,
    qdrant_only: set,
    confirmed: list,
    ambiguous: list,
    uncertain: list,
    sid_conflicts: dict,
    sid_mismatch: list,
) -> list[str]:
    actions: list[str] = []
    if confirmed:
        actions.append(
            f"REVIEW: {len(confirmed)} legacy record(s) have reliable provenance for a "
            "one-way source_id backfill (separate approval required)."
        )
    if uncertain:
        actions.append(
            f"ESCALATE: {len(uncertain)} legacy record(s) lack reliable provenance; "
            "do NOT backfill automatically (need curated/operator mapping)."
        )
    if ambiguous:
        actions.append(
            f"ESCALATE: {len(ambiguous)} legacy record(s) map to multiple source_ids; "
            "resolve manually before any backfill."
        )
    if mongo_only:
        actions.append(
            f"INVESTIGATE: {len(mongo_only)} chunk(s) exist in Mongo but not Qdrant."
        )
    if qdrant_only:
        actions.append(
            f"INVESTIGATE: {len(qdrant_only)} point(s) exist in Qdrant but not Mongo."
        )
    if sid_conflicts:
        actions.append(
            f"INVESTIGATE: {len(sid_conflicts)} source_id(s) have conflicting metadata."
        )
    if sid_mismatch:
        actions.append(
            f"INVESTIGATE: {len(sid_mismatch)} paired chunk(s) disagree on source_id "
            "between stores."
        )
    if not actions:
        actions.append("No identity anomalies or backfill candidates detected.")
    return actions


def format_report(result: Mapping) -> str:
    """Human-readable rendering of a reconciliation report."""
    t = result["totals"]
    i = result["identity"]
    p = result["presence"]
    a = result["anomalies"]
    c = result["conflicts"]
    b = result["backfill"]

    lines = [
        "=" * 70,
        "IDENTITY RECONCILIATION — DRY RUN (read-only, no changes made)",
        "=" * 70,
        f"Mongo records : {t['mongo']} (indexed {t['mongo_indexed']}, "
        f"missing chunk_id {a['mongo_missing_chunk_id']})",
        f"Qdrant points : {t['qdrant']} (indexed {t['qdrant_indexed']}, "
        f"missing chunk_id {a['qdrant_missing_chunk_id']}, "
        f"duplicate chunk_id {a['qdrant_duplicate_chunk_ids']})",
        "",
        f"Keyed   — Mongo {i['mongo_keyed']} | Qdrant {i['qdrant_keyed']}",
        f"Legacy  — Mongo {i['mongo_legacy']} | Qdrant {i['qdrant_legacy']}",
        "",
        f"Paired {p['paired']} | Mongo-only {p['mongo_only']} | Qdrant-only {p['qdrant_only']}",
        "",
        "-- Conflicts --",
        f"source_id metadata conflicts : {len(c['sid_metadata_conflicts'])}",
        f"cross-store source_id mismatch: {c['source_id_mismatch']['count']}",
        "",
        "-- Backfill assessment --",
        f"confirmed : {b['confirmed_count']}",
        f"ambiguous : {b['ambiguous_count']}",
        f"uncertain : {b['uncertain_count']}",
        f"provenance keys loaded : {b['provenance_keys']}",
        "",
        "-- Proposed actions --",
    ]
    lines.extend(f"  * {action}" for action in result["proposed_actions"])
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Read-only CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Dry-run identity reconciliation. READ-ONLY: uses dedicated "
            "connection paths that create no indexes/collections/dirs and "
            "perform no writes, deletes, upserts, or migrations."
        )
    )
    parser.add_argument(
        "--provenance",
        help="Optional JSON file mapping 'TICKER|YEAR|TYPE' -> accession (or list).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--mongo-db", default=None)
    parser.add_argument("--mongo-collection", default=None)
    parser.add_argument("--qdrant-host", default=None)
    parser.add_argument("--qdrant-port", type=int, default=None)
    parser.add_argument("--qdrant-collection", default=None)
    args = parser.parse_args(argv)

    provenance = None
    if args.provenance:
        with open(args.provenance, "r", encoding="utf-8") as handle:
            provenance = json.load(handle)

    from reconcile_readers import open_mongo_reader, open_qdrant_reader

    mongo = open_mongo_reader(
        uri=args.mongo_uri,
        db_name=args.mongo_db,
        collection_name=args.mongo_collection,
    )
    qdrant = open_qdrant_reader(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection_name=args.qdrant_collection,
    )
    try:
        result = reconcile(mongo, qdrant, provenance, sample_limit=args.sample_limit)
    finally:
        mongo.close()
        qdrant.close()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
