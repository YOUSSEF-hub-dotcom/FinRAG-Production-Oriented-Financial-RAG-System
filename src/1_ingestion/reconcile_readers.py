"""
D6-C1 — Dedicated READ-ONLY connection factories for live reconciliation.

These readers intentionally do NOT reuse the application ingestion connection
paths (``MongoDBIndexer`` / ``QdrantIndexer``), because those paths have hidden
write side effects on connect:

  * ``MongoDBIndexer._connect`` runs ``create_index("chunk_id", unique=True)``.
  * ``QdrantIndexer._connect`` runs ``os.makedirs(path)`` and may open embedded
    persistence (which can run migrations).

Instead, the read-only readers here:

  * call NO ``create_index`` / ``create_collection`` / ``os.makedirs`` / upsert /
    delete / migration;
  * open MongoDB / Qdrant via a plain ping + collection-existence check;
  * expose ONLY ``iter_all_chunks`` / ``iter_all_points`` (and ``close``), so a
    caller physically cannot mutate through them;
  * refuse local-path Qdrant mode (embedded persistence is not guaranteed
    read-only on open) and require a Qdrant server;
  * accept optional credentials for programmatically-supplied read-only users
    and never log or print them.

Operation audit for the reconciliation CLI: ``ping``, ``list_collection_names``,
``find`` (Mongo); ``get_collections``, ``scroll`` (Qdrant); plus pure in-memory
report assembly. All are reads.
"""

from __future__ import annotations

from typing import Any, Iterator

from config.logging_config import get_logger

logger = get_logger("ingestion.reconcile_readers")


class ReadOnlyConnectionError(RuntimeError):
    """Raised when a strictly read-only connection cannot be established."""


class MongoReadOnlyReader:
    """Read-only MongoDB chunk reader (no index/collection writes)."""

    def __init__(self, collection: Any, client: Any = None):
        self._collection = collection
        self._client = client

    @classmethod
    def connect(
        cls,
        uri: str,
        db_name: str,
        collection_name: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout_ms: int = 5000,
        client_factory: Any = None,
    ) -> "MongoReadOnlyReader":
        """Open a read-only MongoDB connection and verify the collection exists.

        No index is created and no collection is created. Raises
        :class:`ReadOnlyConnectionError` if the server is unreachable or the
        collection/database cannot be read.
        """
        if client_factory is None:
            import pymongo

            client_factory = pymongo.MongoClient

        kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": timeout_ms}
        if username is not None:
            kwargs["username"] = username
            kwargs["password"] = password

        try:
            client = client_factory(uri, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise ReadOnlyConnectionError(f"MongoDB connection failed: {exc}") from exc

        try:
            client.admin.command("ping")
            db = client[db_name]
            names = db.list_collection_names()
            if collection_name not in names:
                raise ReadOnlyConnectionError(
                    f"MongoDB collection '{db_name}.{collection_name}' not found"
                )
        except ReadOnlyConnectionError:
            _safe_close(client)
            raise
        except Exception as exc:
            _safe_close(client)
            raise ReadOnlyConnectionError(
                f"MongoDB read-only connection failed for '{db_name}.{collection_name}': {exc}"
            ) from exc

        logger.info(
            "Read-only MongoDB connection opened: %s (no index creation)",
            f"{db_name}.{collection_name}",
        )
        return cls(db[collection_name], client=client)

    def iter_all_chunks(self, batch_size: int = 1000) -> Iterator[dict]:
        cursor = self._collection.find({}, {"_id": 0}).batch_size(max(1, int(batch_size)))
        for doc in cursor:
            yield doc

    def close(self) -> None:
        if self._client is not None:
            _safe_close(self._client)
            self._client = None


class QdrantReadOnlyReader:
    """Read-only Qdrant point reader (server mode only; no migrations)."""

    def __init__(self, client: Any, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        collection_name: str,
        *,
        api_key: str | None = None,
        path: str | None = None,
        timeout: int | None = None,
        client_factory: Any = None,
    ) -> "QdrantReadOnlyReader":
        """Open a read-only Qdrant (server) connection and verify the collection.

        Refuses ``path`` (local embedded persistence) because opening it may run
        migrations. Raises :class:`ReadOnlyConnectionError` on failure.
        """
        if path:
            raise ReadOnlyConnectionError(
                "Read-only Qdrant refuses local path mode (embedded persistence may "
                "run migrations on open); connect to a Qdrant server instead."
            )

        if client_factory is None:
            from qdrant_client import QdrantClient

            client_factory = QdrantClient

        kwargs: dict[str, Any] = {"host": host, "port": port}
        if api_key:
            kwargs["api_key"] = api_key
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            client = client_factory(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise ReadOnlyConnectionError(f"Qdrant connection failed: {exc}") from exc

        try:
            names = [c.name for c in client.get_collections().collections]
            if collection_name not in names:
                raise ReadOnlyConnectionError(
                    f"Qdrant collection '{collection_name}' not found"
                )
        except ReadOnlyConnectionError:
            _safe_close(client)
            raise
        except Exception as exc:
            _safe_close(client)
            raise ReadOnlyConnectionError(
                f"Qdrant read-only connection failed for '{collection_name}': {exc}"
            ) from exc

        logger.info(
            "Read-only Qdrant connection opened: %s:%s (collection %s, no migration)",
            host,
            port,
            collection_name,
        )
        return cls(client, collection_name)

    def iter_all_points(self, batch_size: int = 256) -> Iterator[dict]:
        offset = None
        while True:
            result = self._client.scroll(
                collection_name=self._collection_name,
                limit=max(1, int(batch_size)),
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            # qdrant-client >= 1.8 returns (points, next_offset); older returns
            # a bare list. Support both to stay robust across client versions.
            if isinstance(result, tuple):
                points, offset = result
            else:  # pragma: no cover - legacy client shape
                points, offset = result, None
            for point in points:
                yield {"point_id": str(point.id), "payload": point.payload or {}}
            if offset is None:
                break

    def close(self) -> None:
        if self._client is not None:
            _safe_close(self._client)
            self._client = None


def _safe_close(client: Any) -> None:
    try:
        client.close()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def open_mongo_reader(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    **kwargs: Any,
) -> MongoReadOnlyReader:
    """Factory: read-only MongoDB reader from settings unless overridden."""
    from config.settings import MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI

    return MongoReadOnlyReader.connect(
        uri or MONGODB_URI,
        db_name or MONGODB_DB,
        collection_name or MONGODB_COLLECTION,
        **kwargs,
    )


def open_qdrant_reader(
    host: str | None = None,
    port: int | None = None,
    collection_name: str | None = None,
    **kwargs: Any,
) -> QdrantReadOnlyReader:
    """Factory: read-only Qdrant reader from settings unless overridden."""
    from config.settings import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

    return QdrantReadOnlyReader.connect(
        host or QDRANT_HOST,
        port or QDRANT_PORT,
        collection_name or QDRANT_COLLECTION,
        **kwargs,
    )
