"""
Focused offline tests for D6-B: safe replacement of a filing's chunks.

No services, no live DB, no Docker. Storage is simulated with in-memory fakes
that mirror the MongoDBIndexer / QdrantIndexer contract, including the ability
to simulate partial writes (silent per-item failure) and unkeyed legacy points.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_src_root), str(_src_root / "1_ingestion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safe_replacement import (  # noqa: E402
    IdentityError,
    MetadataConflictError,
    ReplacementError,
    WriteVerificationError,
    replace_source,
)


SID = "0000320193-24-000123"
OTHER_SID = "0000950170-24-087843"


# ===========================================================================
# In-memory storage fakes
# ===========================================================================

class FakeMongo:
    def __init__(self, docs=None, events=None):
        self.docs = {d["chunk_id"]: dict(d) for d in (docs or [])}
        self.events = events if events is not None else []
        self.delete_calls = []

    def _doc(self, chunk, source_id=None):
        meta = dict(chunk.get("metadata", {}))
        if source_id is not None:
            meta["source_id"] = source_id
        return {"chunk_id": chunk["chunk_id"], **meta}

    def get_chunks_by_source_id(self, sid):
        self.events.append("mongo.inspect")
        return [dict(d) for d in self.docs.values() if d.get("source_id") == sid]

    def get_chunk_ids_by_source_id(self, sid):
        return {cid for cid, d in self.docs.items() if d.get("source_id") == sid}

    def get_chunks_by_ids(self, ids):
        return {cid: dict(self.docs[cid]) for cid in ids if cid in self.docs}

    def upsert_chunks(self, chunks):
        self.events.append("mongo.upsert")
        written = 0
        for c in chunks:
            self.docs[c["chunk_id"]] = self._doc(c)
            written += 1
        return written

    def delete_stale_chunks(self, sid, ids):
        if not ids:
            return 0
        self.events.append("mongo.delete")
        self.delete_calls.append((sid, list(ids)))
        n = 0
        for cid in ids:
            d = self.docs.get(cid)
            if d and d.get("source_id") == sid:
                del self.docs[cid]
                n += 1
        return n


class FakeQdrant:
    def __init__(self, points=None, events=None):
        self.points = {cid: dict(p) for cid, p in (points or {}).items()}
        self.events = events if events is not None else []
        self.delete_calls = []
        self.fail_ids = set()  # ids silently dropped on upsert (partial write)
        self.fail_all = False

    def get_chunk_ids_by_source_id(self, sid):
        return {cid for cid, p in self.points.items() if p.get("source_id") == sid}

    def upsert_vectors(self, chunks, embeddings):
        self.events.append("qdrant.upsert")
        if self.fail_all:
            return 0
        written = 0
        for c in chunks:
            if c["chunk_id"] in self.fail_ids:
                continue
            self.points[c["chunk_id"]] = dict(c.get("metadata", {}))
            written += 1
        return written

    def delete_stale_by_source_id(self, sid, ids):
        if not ids:
            return 0
        self.events.append("qdrant.delete")
        self.delete_calls.append((sid, list(ids)))
        n = 0
        for cid in ids:
            p = self.points.get(cid)
            if p and p.get("source_id") == sid:
                del self.points[cid]
                n += 1
        return n


def _chunk(cid, ticker="AAPL", year="2024", doc_type="10-K", sid=SID):
    return {
        "chunk_id": cid,
        "text": f"text-{cid}",
        "metadata": {
            "ticker": ticker,
            "fiscal_year": year,
            "doc_type": doc_type,
            "source_id": sid,
            "section": "General",
            "contains_table": False,
        },
    }


def _keyed_doc(cid, ticker="AAPL", year="2024", doc_type="10-K", sid=SID):
    return {
        "chunk_id": cid,
        "ticker": ticker,
        "fiscal_year": year,
        "doc_type": doc_type,
        "source_id": sid,
    }


# ===========================================================================
# Operation ordering
# ===========================================================================

class TestOperationOrdering:

    def test_inspect_then_upsert_then_delete(self):
        events = []
        mongo = FakeMongo([_keyed_doc("stale1")], events)
        qdrant = FakeQdrant({"stale1": {"source_id": SID, "chunk_id": "stale1"}}, events)
        chunks = [_chunk("new1")]
        replace_source(SID, chunks, [[0.0]] * 1, mongo, qdrant)

        assert events.index("mongo.inspect") < events.index("mongo.upsert")
        assert events.index("mongo.upsert") < events.index("mongo.delete")
        assert events.index("qdrant.upsert") < events.index("qdrant.delete")
        assert "stale1" not in mongo.docs
        assert "stale1" not in qdrant.points
        assert "new1" in mongo.docs and "new1" in qdrant.points

    def test_no_delete_when_no_stale(self):
        mongo = FakeMongo()
        qdrant = FakeQdrant()
        replace_source(SID, [_chunk("a")], [[0.0]], mongo, qdrant)
        assert mongo.delete_calls == []
        assert qdrant.delete_calls == []


# ===========================================================================
# Identity / keep-set validation (fail closed)
# ===========================================================================

class TestIdentityValidation:

    @pytest.mark.parametrize("bad", ["", "   ", "UNKNOWN", None, 123])
    def test_rejects_bad_source_id(self, bad):
        with pytest.raises(IdentityError):
            replace_source(bad, [_chunk("a")], [[0.0]], FakeMongo(), FakeQdrant())

    def test_rejects_empty_chunk_set(self):
        with pytest.raises(IdentityError):
            replace_source(SID, [], [], FakeMongo(), FakeQdrant())

    def test_rejects_duplicate_chunk_ids(self):
        mongo, qdrant = FakeMongo(), FakeQdrant()
        chunks = [_chunk("dup"), _chunk("dup")]
        with pytest.raises(ReplacementError):
            replace_source(SID, chunks, [[0.0], [0.0]], mongo, qdrant)
        assert mongo.docs == {} and qdrant.points == {}

    def test_rejects_length_mismatch(self):
        with pytest.raises(ReplacementError):
            replace_source(SID, [_chunk("a"), _chunk("b")], [[0.0]], FakeMongo(), FakeQdrant())


# ===========================================================================
# Metadata-conflict gate (refuse before any write)
# ===========================================================================

class TestMetadataConflict:

    def test_refuses_conflicting_ticker(self):
        mongo = FakeMongo([_keyed_doc("old", ticker="AAPL")])
        qdrant = FakeQdrant({"old": {"source_id": SID}})
        with pytest.raises(MetadataConflictError):
            replace_source(SID, [_chunk("new", ticker="MSFT")], [[0.0]], mongo, qdrant)
        assert mongo.delete_calls == [] and qdrant.delete_calls == []
        assert "old" in mongo.docs  # nothing mutated

    def test_refuses_conflicting_doc_type(self):
        mongo = FakeMongo([_keyed_doc("old", doc_type="10-K")])
        qdrant = FakeQdrant()
        with pytest.raises(MetadataConflictError):
            replace_source(SID, [_chunk("new", doc_type="10-K/A")], [[0.0]], mongo, qdrant)

    def test_allows_when_existing_unknown(self):
        mongo = FakeMongo([_keyed_doc("old", ticker="UNKNOWN", year="UNKNOWN", doc_type="UNKNOWN")])
        qdrant = FakeQdrant({"old": {"source_id": SID}})
        summary = replace_source(SID, [_chunk("new", ticker="AAPL")], [[0.0]], mongo, qdrant)
        assert summary["converged"] is True

    def test_allows_when_incoming_unknown_and_existing_empty(self):
        mongo = FakeMongo([_keyed_doc("old", ticker="", year="", doc_type="")])
        qdrant = FakeQdrant({"old": {"source_id": SID}})
        summary = replace_source(SID, [_chunk("new", ticker="UNKNOWN")], [[0.0]], mongo, qdrant)
        assert summary["converged"] is True

    def test_conflict_leaves_existing_intact(self):
        mongo = FakeMongo([_keyed_doc("old", ticker="AAPL"), _keyed_doc("old2", ticker="AAPL")])
        qdrant = FakeQdrant({"old": {"source_id": SID}})
        with pytest.raises(MetadataConflictError):
            replace_source(SID, [_chunk("new", ticker="NVDA")], [[0.0]], mongo, qdrant)
        assert set(mongo.docs) == {"old", "old2"}


# ===========================================================================
# Partial-write failure (must not delete, must not report success)
# ===========================================================================

class TestPartialWrite:

    def test_qdrant_partial_write_raises_and_skips_delete(self):
        events = []
        mongo = FakeMongo([_keyed_doc("stale")], events)
        qdrant = FakeQdrant({"stale": {"source_id": SID}}, events)
        qdrant.fail_ids = {"new2"}  # silently drop one point
        chunks = [_chunk("new1"), _chunk("new2")]
        with pytest.raises(WriteVerificationError):
            replace_source(SID, chunks, [[0.0], [0.0]], mongo, qdrant)
        assert "qdrant.delete" not in events
        assert "mongo.delete" not in events
        assert "stale" in qdrant.points  # stale NOT deleted after a failed write

    def test_qdrant_total_failure_raises(self):
        mongo = FakeMongo()
        qdrant = FakeQdrant()
        qdrant.fail_all = True
        with pytest.raises(WriteVerificationError):
            replace_source(SID, [_chunk("a")], [[0.0]], mongo, qdrant)

    def test_mongo_missing_after_write_raises(self):
        mongo = FakeMongo()
        # simulate Mongo silently failing to persist one id
        original = mongo.upsert_chunks

        def dropping(chunks):
            result = original(chunks)
            mongo.docs.pop(chunks[-1]["chunk_id"], None)
            return result

        mongo.upsert_chunks = dropping
        qdrant = FakeQdrant()
        chunks = [_chunk("a"), _chunk("b")]
        with pytest.raises(WriteVerificationError):
            replace_source(SID, chunks, [[0.0], [0.0]], mongo, qdrant)


# ===========================================================================
# Stale-deletion scope + legacy safety
# ===========================================================================

class TestDeletionScope:

    def test_only_stale_ids_deleted_within_scope(self):
        mongo = FakeMongo([_keyed_doc("keep"), _keyed_doc("stale")])
        qdrant = FakeQdrant(
            {
                "keep": {"source_id": SID, "chunk_id": "keep"},
                "stale": {"source_id": SID, "chunk_id": "stale"},
            }
        )
        summary = replace_source(SID, [_chunk("keep")], [[0.0]], mongo, qdrant)
        assert "stale" not in mongo.docs
        assert "stale" not in qdrant.points
        assert "keep" in mongo.docs and "keep" in qdrant.points
        assert summary["stale_removed_mongo"] == 1
        assert summary["stale_removed_qdrant"] == 1
        # delete calls were scoped to the approved source_id
        assert all(sid == SID for sid, _ in mongo.delete_calls)
        assert all(sid == SID for sid, _ in qdrant.delete_calls)

    def test_other_source_id_untouched(self):
        mongo = FakeMongo([_keyed_doc("mine", sid=SID), _keyed_doc("theirs", sid=OTHER_SID)])
        qdrant = FakeQdrant(
            {
                "mine": {"source_id": SID},
                "theirs": {"source_id": OTHER_SID},
            }
        )
        replace_source(SID, [_chunk("new")], [[0.0]], mongo, qdrant)
        assert "theirs" in mongo.docs
        assert "theirs" in qdrant.points

    def test_unkeyed_legacy_never_deleted(self):
        mongo = FakeMongo([_keyed_doc("stale")])
        # Qdrant holds an UNKEYED legacy point sharing the stale chunk_id.
        qdrant = FakeQdrant({"stale": {"chunk_id": "stale"}})
        summary = replace_source(SID, [_chunk("new")], [[0.0]], mongo, qdrant)
        assert "stale" in qdrant.points  # legacy point preserved
        assert "stale" not in mongo.docs  # keyed Mongo stale removed
        assert "new" in mongo.docs and "new" in qdrant.points
        assert summary["converged"] is True

    def test_mongo_stale_scoped_by_source_id(self):
        # Same chunk_id stored under a different source_id must not be deleted.
        mongo = FakeMongo([_keyed_doc("shared", sid=OTHER_SID)])
        qdrant = FakeQdrant()
        replace_source(SID, [_chunk("new")], [[0.0]], mongo, qdrant)
        assert "shared" in mongo.docs


# ===========================================================================
# Retry safety / idempotency
# ===========================================================================

class TestRetrySafety:

    def test_second_run_is_idempotent(self):
        mongo = FakeMongo()
        qdrant = FakeQdrant()
        chunks = [_chunk("a"), _chunk("b")]
        first = replace_source(SID, chunks, [[0.0], [0.0]], mongo, qdrant)
        assert first["converged"] is True

        mongo.delete_calls.clear()
        qdrant.delete_calls.clear()
        second = replace_source(SID, chunks, [[0.0], [0.0]], mongo, qdrant)
        assert second["converged"] is True
        assert mongo.delete_calls == [] and qdrant.delete_calls == []
        assert set(mongo.docs) == {"a", "b"}
        assert set(qdrant.points) == {"a", "b"}

    def test_shrinking_set_removes_tail(self):
        mongo = FakeMongo()
        qdrant = FakeQdrant()
        replace_source(SID, [_chunk("a"), _chunk("b"), _chunk("c")], [[0.0]] * 3, mongo, qdrant)
        summary = replace_source(SID, [_chunk("a")], [[0.0]], mongo, qdrant)
        assert summary["converged"] is True
        assert set(mongo.docs) == {"a"}
        assert set(qdrant.points) == {"a"}


# ===========================================================================
# Qdrant payload propagation (D6-B scope item 2)
# ===========================================================================

class TestPayloadPropagation:

    def test_source_id_in_qdrant_payload_fields(self):
        from database_indexer import _QDRANT_PAYLOAD_FIELDS

        assert "source_id" in _QDRANT_PAYLOAD_FIELDS
