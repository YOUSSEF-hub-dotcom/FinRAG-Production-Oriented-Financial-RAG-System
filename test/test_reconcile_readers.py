"""
Focused offline tests for D6-C1: strictly read-only reconciliation connections.

The fakes RECORD any write-capable operation (create_index, drop, insert,
update, delete, create_collection, upsert, ...) instead of raising, so a test
can prove that the read-only connect path never invoked one.
"""

import logging
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_src_root), str(_src_root / "1_ingestion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reconcile_identity import reconcile  # noqa: E402
from reconcile_readers import (  # noqa: E402
    MongoReadOnlyReader,
    QdrantReadOnlyReader,
    ReadOnlyConnectionError,
    open_mongo_reader,
    open_qdrant_reader,
)


COLLECTION = "raw_chunks"
QCOLLECTION = "financial_vectors"


# ===========================================================================
# MongoDB fakes — record writes, never raise
# ===========================================================================

class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self.batch = None

    def batch_size(self, n):
        self.batch = n
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self, docs, calls):
        self._docs = docs
        self.calls = calls

    def find(self, flt, projection=None):
        self.calls.append(("find", flt, projection))
        return FakeCursor(self._docs)

    # --- write-capable operations: record and return benignly --------------
    def create_index(self, *a, **k):
        self.calls.append("WRITE:create_index")

    def insert_many(self, *a, **k):
        self.calls.append("WRITE:insert_many")

    def insert_one(self, *a, **k):
        self.calls.append("WRITE:insert_one")

    def update_one(self, *a, **k):
        self.calls.append("WRITE:update_one")

    def delete_many(self, *a, **k):
        self.calls.append("WRITE:delete_many")

    def drop(self, *a, **k):
        self.calls.append("WRITE:drop")


class FakeMongoDB:
    def __init__(self, collections, calls):
        self._collections = collections
        self.calls = calls

    def list_collection_names(self):
        self.calls.append("list_collection_names")
        return list(self._collections)

    def __getitem__(self, name):
        self.calls.append(("db_getitem", name))
        return self._collections[name]

    def create_collection(self, *a, **k):
        self.calls.append("WRITE:create_collection")


class FakeAdmin:
    def __init__(self, calls):
        self.calls = calls

    def command(self, name):
        self.calls.append(("admin_command", name))
        return {"ok": 1}


class FakeMongoClient:
    def __init__(self, collections, calls, uri=None, **kwargs):
        self.calls = calls
        self.uri = uri
        self.kwargs = kwargs
        self.admin = FakeAdmin(calls)
        self._db = FakeMongoDB(collections, calls)

    def __getitem__(self, name):
        self.calls.append(("client_getitem", name))
        return self._db

    def close(self):
        self.calls.append("close")

    def drop_database(self, *a, **k):
        self.calls.append("WRITE:drop_database")


def make_mongo_factory(collections, calls):
    def factory(uri, **kwargs):
        calls.append(("MongoClient", uri))
        return FakeMongoClient(collections, calls, uri=uri, **kwargs)
    return factory


def has_write(calls):
    return any(isinstance(c, str) and c.startswith("WRITE:") for c in calls)


# ===========================================================================
# Qdrant fakes — record writes, never raise
# ===========================================================================

class FakeQdrantClient:
    def __init__(self, collection_names, points, calls, **kwargs):
        self._names = collection_names
        self._points = points
        self.calls = calls
        self.kwargs = kwargs

    def get_collections(self):
        self.calls.append("get_collections")
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self._names])

    def scroll(self, collection_name, limit=256, with_payload=True, with_vectors=False, offset=None):
        self.calls.append(("scroll", collection_name, offset))
        return list(self._points.get(collection_name, [])), None

    def close(self):
        self.calls.append("close")

    # --- write-capable operations ----------------------------------------
    def create_collection(self, *a, **k):
        self.calls.append("WRITE:create_collection")

    def delete_collection(self, *a, **k):
        self.calls.append("WRITE:delete_collection")

    def upsert(self, *a, **k):
        self.calls.append("WRITE:upsert")

    def delete(self, *a, **k):
        self.calls.append("WRITE:delete")


def make_qdrant_factory(collection_names, points, calls):
    def factory(**kwargs):
        calls.append(("QdrantClient", kwargs))
        return FakeQdrantClient(collection_names, points, calls, **kwargs)
    return factory


# ===========================================================================
# MongoDB read-only reader
# ===========================================================================

class TestMongoReadOnlyReader:

    def test_connects_and_reads_without_writes(self):
        calls = []
        docs = [{"chunk_id": "a", "ticker": "AAPL"}, {"chunk_id": "b"}]
        reader = MongoReadOnlyReader.connect(
            "mongodb://localhost:27017",
            "fin",
            COLLECTION,
            client_factory=make_mongo_factory({COLLECTION: FakeCollection(docs, calls)}, calls),
        )
        assert list(reader.iter_all_chunks()) == docs
        assert not has_write(calls)  # no create_index / insert / update / delete
        assert "list_collection_names" in calls
        assert ("admin_command", "ping") in calls

    def test_missing_collection_raises_and_closes(self):
        calls = []
        with pytest.raises(ReadOnlyConnectionError):
            MongoReadOnlyReader.connect(
                "mongodb://localhost:27017",
                "fin",
                "does_not_exist",
                client_factory=make_mongo_factory({COLLECTION: FakeCollection([], calls)}, calls),
            )
        assert "close" in calls
        assert not has_write(calls)

    def test_ping_failure_raises(self):
        class FailingAdmin:
            def command(self, name):
                raise RuntimeError("server down")

        calls = []

        class FailingClient(FakeMongoClient):
            def __init__(self, **kwargs):
                super().__init__({COLLECTION: FakeCollection([], calls)}, calls, **kwargs)
                self.admin = FailingAdmin()

        with pytest.raises(ReadOnlyConnectionError):
            MongoReadOnlyReader.connect(
                "mongodb://localhost:27017",
                "fin",
                COLLECTION,
                client_factory=lambda uri, **kw: FailingClient(**kw),
            )

    def test_password_not_logged(self, caplog):
        calls = []
        secret = "sup3r-s3cret-pw"
        with caplog.at_level(logging.INFO):
            MongoReadOnlyReader.connect(
                "mongodb://localhost:27017",
                "fin",
                COLLECTION,
                username="readonly_user",
                password=secret,
                client_factory=make_mongo_factory({COLLECTION: FakeCollection([], calls)}, calls),
            )
        assert secret not in caplog.text


# ===========================================================================
# Qdrant read-only reader
# ===========================================================================

class TestQdrantReadOnlyReader:

    def test_connects_and_reads_without_writes(self):
        calls = []
        points = {
            QCOLLECTION: [
                SimpleNamespace(id="p1", payload={"chunk_id": "a"}),
                SimpleNamespace(id="p2", payload={"chunk_id": "b"}),
            ]
        }
        reader = QdrantReadOnlyReader.connect(
            "localhost",
            6333,
            QCOLLECTION,
            client_factory=make_qdrant_factory([QCOLLECTION], points, calls),
        )
        got = list(reader.iter_all_points())
        assert [p["payload"]["chunk_id"] for p in got] == ["a", "b"]
        assert not has_write(calls)
        assert "get_collections" in calls

    def test_refuses_local_path_mode(self):
        calls = []
        with pytest.raises(ReadOnlyConnectionError):
            QdrantReadOnlyReader.connect(
                "localhost",
                6333,
                QCOLLECTION,
                path="/some/local/qdrant",
                client_factory=make_qdrant_factory([QCOLLECTION], {}, calls),
            )
        # factory must not even be constructed in path mode
        assert calls == []

    def test_missing_collection_raises_and_closes(self):
        calls = []
        with pytest.raises(ReadOnlyConnectionError):
            QdrantReadOnlyReader.connect(
                "localhost",
                6333,
                "nope",
                client_factory=make_qdrant_factory([QCOLLECTION], {}, calls),
            )
        assert "close" in calls
        assert not has_write(calls)

    def test_iter_all_points_paginates_with_offset(self):
        calls = []
        pages = {
            None: ([SimpleNamespace(id="p1", payload={"chunk_id": "a"})], "off1"),
            "off1": ([SimpleNamespace(id="p2", payload={"chunk_id": "b"})], "off2"),
            "off2": ([SimpleNamespace(id="p3", payload={"chunk_id": "c"})], None),
        }

        class PagingClient:
            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=QCOLLECTION)])

            def scroll(self, collection_name, limit, with_payload, with_vectors, offset):
                calls.append(offset)
                return pages[offset]

            def close(self):
                calls.append("close")

        reader = QdrantReadOnlyReader.connect(
            "localhost", 6333, QCOLLECTION, client_factory=lambda **kw: PagingClient()
        )
        got = [p["payload"]["chunk_id"] for p in reader.iter_all_points(batch_size=2)]
        assert got == ["a", "b", "c"]
        assert calls == [None, "off1", "off2"]
        assert not has_write(calls)

    def test_iter_all_points_supports_legacy_list_return(self):
        calls = []

        class LegacyClient:
            def scroll(self, **kwargs):
                calls.append(kwargs.get("offset"))
                return [SimpleNamespace(id="p1", payload={"chunk_id": "a"})]

        reader = QdrantReadOnlyReader(LegacyClient(), QCOLLECTION)
        got = list(reader.iter_all_points())
        assert [p["payload"]["chunk_id"] for p in got] == ["a"]
        assert calls == [None]


# ===========================================================================
# Structural guarantees
# ===========================================================================

class TestStructuralReadOnly:

    def test_readers_expose_no_write_methods(self):
        for cls in (MongoReadOnlyReader, QdrantReadOnlyReader):
            for name in (
                "upsert_chunks",
                "upsert_vectors",
                "delete_stale_chunks",
                "delete_stale_by_source_id",
                "drop_collection",
                "create_index",
                "create_collection",
            ):
                assert not hasattr(cls, name), f"{cls.__name__}.{name} should not exist"

    def test_reconcile_runs_over_readers(self):
        calls = []
        docs = [{"chunk_id": "a", "ticker": "AAPL", "source_id": "ACC"}]
        mongo = MongoReadOnlyReader(FakeCollection(docs, calls))
        qclient = FakeQdrantClient(
            [QCOLLECTION],
            {QCOLLECTION: [SimpleNamespace(id="p", payload={"chunk_id": "a", "source_id": "ACC"})]},
            calls,
        )
        qdrant = QdrantReadOnlyReader(qclient, QCOLLECTION)
        result = reconcile(mongo, qdrant)
        assert result["totals"]["mongo"] == 1
        assert result["totals"]["qdrant"] == 1
        assert result["presence"]["paired"] == 1
        assert not has_write(calls)


# ===========================================================================
# Factories (settings defaults + overrides)
# ===========================================================================

class TestFactories:

    def test_open_mongo_reader_forwards_overrides(self, monkeypatch):
        captured = {}

        def fake_connect(cls, uri, db_name, collection_name, **kwargs):
            captured.update(uri=uri, db=db_name, coll=collection_name)
            return "mongo-reader"

        monkeypatch.setattr(MongoReadOnlyReader, "connect", classmethod(fake_connect))
        assert open_mongo_reader("U", "D", "C") == "mongo-reader"
        assert captured == {"uri": "U", "db": "D", "coll": "C"}

    def test_open_qdrant_reader_forwards_overrides(self, monkeypatch):
        captured = {}

        def fake_connect(cls, host, port, collection_name, **kwargs):
            captured.update(host=host, port=port, coll=collection_name)
            return "qdrant-reader"

        monkeypatch.setattr(QdrantReadOnlyReader, "connect", classmethod(fake_connect))
        assert open_qdrant_reader("H", 1234, "C") == "qdrant-reader"
        assert captured == {"host": "H", "port": 1234, "coll": "C"}
