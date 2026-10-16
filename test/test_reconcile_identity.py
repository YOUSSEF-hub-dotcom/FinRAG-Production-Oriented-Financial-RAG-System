"""
Focused offline tests for D6-C: dry-run identity reconciliation.

No services, no live DB, no Docker. Storage is simulated with read-only,
in-memory fakes (no write methods exist), so any mutation attempt would fail
loudly. The reconciler is pure and returns a report dict.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_src_root), str(_src_root / "1_ingestion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reconcile_identity import (  # noqa: E402
    build_provenance_from_filings,
    format_report,
    is_valid_source_id,
    normalize_provenance,
    reconcile,
)


ACC = "0000320193-24-000123"
ACC2 = "0000950170-24-087843"


# ===========================================================================
# Read-only fakes (intentionally have NO write methods)
# ===========================================================================

class FakeMongo:
    def __init__(self, docs=None):
        self.docs = [dict(d) for d in (docs or [])]

    def iter_all_chunks(self):
        return iter([dict(d) for d in self.docs])


class FakeQdrant:
    def __init__(self, points=None):
        self.points = [dict(p) for p in (points or [])]

    def iter_all_points(self):
        return iter([dict(p) for p in self.points])


def mdoc(cid, ticker="AAPL", year="2024", doc_type="10-K", sid=None):
    doc = {"chunk_id": cid, "ticker": ticker, "fiscal_year": year, "doc_type": doc_type}
    if sid is not None:
        doc["source_id"] = sid
    return doc


def qpt(cid, ticker="AAPL", year="2024", doc_type="10-K", sid=None):
    payload = {"chunk_id": cid, "ticker": ticker, "fiscal_year": year, "doc_type": doc_type}
    if sid is not None:
        payload["source_id"] = sid
    return {"point_id": f"p-{cid}", "payload": payload}


# ===========================================================================
# Helper-level tests
# ===========================================================================

class TestHelpers:

    def test_is_valid_source_id(self):
        assert is_valid_source_id(ACC)
        assert not is_valid_source_id("")
        assert not is_valid_source_id("  ")
        assert not is_valid_source_id("UNKNOWN")
        assert not is_valid_source_id(None)

    def test_normalize_provenance_pipe_keys_and_lists(self):
        prov = normalize_provenance(
            {"AAPL|2024|10-K": ACC, "MSFT|2024|10-K": [ACC2, "0000000000-00-000000"]}
        )
        assert prov[("AAPL", "2024", "10-K")] == [ACC]
        assert prov[("MSFT", "2024", "10-K")] == [ACC2, "0000000000-00-000000"]

    def test_build_provenance_from_filings(self):
        prov = build_provenance_from_filings([("AAPL", "2024", ACC), ("MSFT", "2024", ACC2)])
        assert prov[("AAPL", "2024", "10-K")] == [ACC]
        assert prov[("MSFT", "2024", "10-K")] == [ACC2]


# ===========================================================================
# Counting / categorisation
# ===========================================================================

class TestCategorization:

    def test_keyed_vs_legacy_counts(self):
        mongo = FakeMongo([mdoc("a", sid=ACC), mdoc("b")])
        qdrant = FakeQdrant([qpt("a", sid=ACC), qpt("b")])
        result = reconcile(mongo, qdrant)
        assert result["identity"] == {
            "mongo_keyed": 1,
            "mongo_legacy": 1,
            "qdrant_keyed": 1,
            "qdrant_legacy": 1,
        }

    def test_presence_categories(self):
        mongo = FakeMongo([mdoc("a", sid=ACC), mdoc("only_mongo", sid=ACC)])
        qdrant = FakeQdrant([qpt("a", sid=ACC), qpt("only_qdrant", sid=ACC)])
        result = reconcile(mongo, qdrant)
        assert result["presence"]["paired"] == 1
        assert result["presence"]["mongo_only"] == 1
        assert result["presence"]["qdrant_only"] == 1
        assert result["presence"]["mongo_only_sample"] == ["only_mongo"]
        assert result["presence"]["qdrant_only_sample"] == ["only_qdrant"]

    def test_totals_and_missing_chunk_id(self):
        mongo = FakeMongo([mdoc("a", sid=ACC), {"ticker": "AAPL"}])
        qdrant = FakeQdrant([qpt("a", sid=ACC), {"point_id": "z", "payload": {}}])
        result = reconcile(mongo, qdrant)
        assert result["totals"]["mongo"] == 2
        assert result["totals"]["qdrant"] == 2
        assert result["anomalies"]["mongo_missing_chunk_id"] == 1
        assert result["anomalies"]["qdrant_missing_chunk_id"] == 1


# ===========================================================================
# Backfill assessment
# ===========================================================================

class TestBackfillAssessment:

    def test_confirmed_via_other_store_keyed(self):
        mongo = FakeMongo([mdoc("a", sid=ACC)])
        qdrant = FakeQdrant([qpt("a")])  # legacy point, same chunk_id
        result = reconcile(mongo, qdrant)
        assert result["backfill"]["confirmed_count"] == 1
        assert result["backfill"]["confirmed_sample"][0]["evidence"] == "other_store_keyed"
        assert result["backfill"]["confirmed_sample"][0]["source_id"] == ACC
        assert result["backfill"]["confirmed_sample"][0]["store"] == "qdrant"

    def test_confirmed_via_provenance_map(self):
        mongo = FakeMongo([mdoc("c", ticker="AAPL", year="2024", doc_type="10-K")])
        qdrant = FakeQdrant([qpt("c", ticker="AAPL", year="2024", doc_type="10-K")])
        provenance = {"AAPL|2024|10-K": ACC}
        result = reconcile(mongo, qdrant, provenance)
        assert result["backfill"]["confirmed_count"] == 2
        assert all(
            row["evidence"] == "provenance_map"
            for row in result["backfill"]["confirmed_sample"]
        )

    def test_uncertain_when_no_provenance(self):
        mongo = FakeMongo([mdoc("c", ticker="AAPL", year="2024", doc_type="10-K")])
        qdrant = FakeQdrant()
        result = reconcile(mongo, qdrant)
        assert result["backfill"]["uncertain_count"] == 1
        assert result["backfill"]["uncertain_sample"][0]["reason"] == "no_provenance"

    def test_uncertain_when_incomplete_metadata(self):
        mongo = FakeMongo([mdoc("c", ticker="UNKNOWN", year="UNKNOWN", doc_type="UNKNOWN")])
        qdrant = FakeQdrant()
        provenance = {"UNKNOWN|UNKNOWN|UNKNOWN": ACC}
        result = reconcile(mongo, qdrant, provenance)
        assert result["backfill"]["uncertain_sample"][0]["reason"] == "incomplete_metadata"

    def test_ambiguous_when_multiple_provenance(self):
        mongo = FakeMongo([mdoc("c", ticker="MSFT", year="2024", doc_type="10-K")])
        qdrant = FakeQdrant()
        provenance = {"MSFT|2024|10-K": [ACC, ACC2]}
        result = reconcile(mongo, qdrant, provenance)
        assert result["backfill"]["ambiguous_count"] == 1
        assert result["backfill"]["ambiguous_sample"][0]["reason"] == "ambiguous_provenance"


# ===========================================================================
# Conflicts
# ===========================================================================

class TestConflicts:

    def test_sid_metadata_conflict(self):
        mongo = FakeMongo([
            mdoc("a", ticker="AAPL", year="2024", doc_type="10-K", sid=ACC),
            mdoc("b", ticker="MSFT", year="2024", doc_type="10-K", sid=ACC),
        ])
        qdrant = FakeQdrant([qpt("a", sid=ACC), qpt("b", sid=ACC)])
        result = reconcile(mongo, qdrant)
        assert ACC in result["conflicts"]["sid_metadata_conflicts"]
        assert len(result["conflicts"]["sid_metadata_conflicts"][ACC]) == 2

    def test_cross_store_source_id_mismatch(self):
        mongo = FakeMongo([mdoc("a", sid=ACC)])
        qdrant = FakeQdrant([qpt("a", sid=ACC2)])
        result = reconcile(mongo, qdrant)
        assert result["conflicts"]["source_id_mismatch"]["count"] == 1
        sample = result["conflicts"]["source_id_mismatch"]["sample"][0]
        assert sample == {"chunk_id": "a", "mongo": ACC, "qdrant": ACC2}

    def test_qdrant_duplicate_chunk_ids(self):
        qdrant = FakeQdrant([qpt("dup", sid=ACC), qpt("dup", sid=ACC)])
        result = reconcile(FakeMongo(), qdrant)
        assert result["anomalies"]["qdrant_duplicate_chunk_ids"] == 1


# ===========================================================================
# Dry-run guarantees
# ===========================================================================

class TestDryRunSafety:

    def test_reconcile_does_not_mutate_inputs(self):
        mongo = FakeMongo([mdoc("a", sid=ACC), mdoc("b")])
        qdrant = FakeQdrant([qpt("a", sid=ACC), qpt("b")])
        mongo_before = [dict(d) for d in mongo.docs]
        qdrant_before = [dict(p) for p in qdrant.points]

        reconcile(mongo, qdrant, {"AAPL|2024|10-K": ACC})

        assert mongo.docs == mongo_before
        assert qdrant.points == qdrant_before

    def test_fakes_have_no_write_methods(self):
        # A structural guarantee that the reconciler cannot upsert/delete.
        for fake in (FakeMongo(), FakeQdrant()):
            for name in ("upsert_chunks", "upsert_vectors", "delete_stale_chunks",
                         "delete_stale_by_source_id", "drop_collection"):
                assert not hasattr(fake, name)

    def test_report_proposes_actions_without_applying(self):
        mongo = FakeMongo([mdoc("b", ticker="AAPL", year="2024", doc_type="10-K")])
        qdrant = FakeQdrant()
        result = reconcile(mongo, qdrant, {"AAPL|2024|10-K": ACC})
        assert result["proposed_actions"]
        assert any("REVIEW" in a or "ESCALATE" in a for a in result["proposed_actions"])


# ===========================================================================
# Report rendering
# ===========================================================================

class TestReport:

    def test_format_report_contains_key_sections(self):
        mongo = FakeMongo([mdoc("a", sid=ACC)])
        qdrant = FakeQdrant([qpt("a", sid=ACC)])
        text = format_report(reconcile(mongo, qdrant))
        assert "DRY RUN" in text
        assert "Keyed" in text and "Legacy" in text
        assert "Backfill assessment" in text
        assert "no changes made" in text
