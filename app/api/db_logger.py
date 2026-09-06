"""
Asynchronous MongoDB audit-logging service for the Financial RAG API.

Stores a rich, **unstructured / dynamic** execution trace for every RAG request
in the ``rag_audit_logs`` collection of the configured MongoDB database. Each
document captures the full request -> retrieval -> LLM -> response lifecycle so
that quality, cost, latency and guardrail behaviour can be audited after the
fact.

Design notes:
  * Built on ``pymongo.AsyncMongoClient`` (pymongo >= 4.6) so inserts never
    block the event loop.
  * The logger is **fire-and-forget**: it is invoked from FastAPI
    ``BackgroundTasks`` (or the Arq worker) so the HTTP response is returned
    immediately. All failures are swallowed and logged -- audit logging must
    never break a user's answer.
  * A ``collection`` may be injected (tests, alternate backends) to avoid
    touching a real database.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from config.logging_config import get_logger
from config.settings import MONGODB_DB, MONGODB_URI

from app.api.schemas import AuditLogEvent

logger = get_logger("api.db_logger")

# Collection that persists the dynamic RAG execution traces.
AUDIT_COLLECTION_NAME: str = "rag_audit_logs"


class MongoAuditLogger:
    """Async MongoDB writer for RAG audit events (``rag_audit_logs``)."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: str = AUDIT_COLLECTION_NAME,
        collection: Any = None,
    ) -> None:
        """Create the logger.

        Args:
            uri: MongoDB connection URI (defaults to ``MONGODB_URI``).
            db_name: Database name (defaults to ``MONGODB_DB``).
            collection_name: Target collection (defaults to ``rag_audit_logs``).
            collection: Pre-built collection object (injected for tests / custom
                backends). When provided, no connection is opened.
        """
        self._uri = uri or MONGODB_URI
        self._db_name = db_name or MONGODB_DB
        self._collection_name = collection_name
        self._client: Optional[AsyncMongoClient] = None
        self._collection = collection  # injected (tests) -> no real I/O

    async def connect(self) -> None:
        """Lazily open the async MongoDB connection (no-op if injected)."""
        if self._collection is not None:
            return
        if self._client is None:
            self._client = AsyncMongoClient(self._uri, serverSelectionTimeoutMS=5000)

    async def _get_collection(self):
        """Return the active collection, connecting on first use if needed."""
        if self._collection is not None:
            return self._collection
        await self.connect()
        db = self._client[self._db_name]
        return db[self._collection_name]

    async def log_event(self, event: "AuditLogEvent | dict") -> Optional[str]:
        """Persist a single audit event to MongoDB.

        Args:
            event: An :class:`AuditLogEvent` (preferred) or a plain dict with the
                same dynamic schema.

        Returns:
            The inserted document id as a string, or ``None`` on failure.
        """
        try:
            if isinstance(event, AuditLogEvent):
                doc: dict[str, Any] = event.model_dump(mode="json")
            else:
                doc = dict(event)

            # Normalise timestamp to a real BSON datetime for TTL / sorting.
            ts = doc.get("timestamp")
            if isinstance(ts, str):
                try:
                    doc["timestamp"] = datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    )
                except Exception:
                    doc["timestamp"] = datetime.now(timezone.utc)
            elif not isinstance(ts, datetime):
                doc["timestamp"] = datetime.now(timezone.utc)

            collection = await self._get_collection()
            result = await collection.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as exc:
            logger.error("Audit log insert failed (Mongo): %s", exc, exc_info=True)
            return None
        except Exception as exc:  # never crash the request pipeline
            logger.error("Audit log insert failed (unexpected): %s", exc, exc_info=True)
            return None

    async def get_recent_logs(
        self, limit: int = 50, user_identifier: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Fetch the most recent audit logs (newest first).

        Args:
            limit: Maximum number of documents to return.
            user_identifier: Optional filter by ``user_identifier``.

        Returns:
            List of audit log documents (``_id`` stringified).
        """
        collection = await self._get_collection()
        filt: dict[str, Any] = {}
        if user_identifier:
            filt["user_identifier"] = user_identifier
        cursor = collection.find(filt).sort("timestamp", -1).limit(limit)
        out: list[dict[str, Any]] = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            out.append(doc)
        return out

    async def close(self) -> None:
        """Close the underlying async client if one was opened."""
        if self._client is not None:
            # pymongo.AsyncMongoClient.close() is a coroutine.
            try:
                await self._client.close()
            except TypeError:
                self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton (lazy) + accessor
# ---------------------------------------------------------------------------
_audit_logger: Optional[MongoAuditLogger] = None


def get_audit_logger(collection: Any = None) -> MongoAuditLogger:
    """Return a MongoAuditLogger, reusing a process-wide singleton.

    Args:
        collection: When supplied, a *fresh* logger wrapping that collection is
            returned (used by tests to inject a fake collection).

    Returns:
        A configured :class:`MongoAuditLogger` instance.
    """
    global _audit_logger
    if collection is not None:
        return MongoAuditLogger(collection=collection)
    if _audit_logger is None:
        _audit_logger = MongoAuditLogger()
    return _audit_logger


async def _enqueue_audit_via_arq(event_dict: dict[str, Any]) -> None:
    """Best-effort Arq enqueue of an audit event (sync wrapper for bg tasks)."""
    # Imported lazily to avoid a module-level cycle with app.api.worker.
    from app.api.worker import enqueue_audit_log

    try:
        await enqueue_audit_log(event_dict)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Audit Arq enqueue failed: %s", exc)
