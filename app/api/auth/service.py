"""
MongoDB-backed user persistence for the Financial RAG AuthN/AuthZ layer.

Stores accounts in the "users" collection of the configured database. The
collection is injectable so the test-suite can run fully offline against a fake
collection. Password hashing/verification is delegated to :mod:`app.api.auth.jwt`
(bcrypt via "pwdlib").

Document shape
--------------
{
  "_id": ObjectId,
  "user_id": str,          # stable UUID, used as JWT "sub"
  "email": str,            # lower-cased, unique index
  "full_name": str,
  "hashed_password": str,  # bcrypt hash (never returned to clients)
  "role": "user" | "admin",
  "is_active": bool,
  "created_at": datetime,
  "updated_at": datetime
}
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError

from config.logging_config import get_logger
from config.settings import AUTH_USERS_COLLECTION, MONGODB_DB, MONGODB_URI

from app.api.auth.jwt import hash_password, verify_password

logger = get_logger("api.auth.service")

# Allowed roles -- RBAC is a strict allow-list.
VALID_ROLES: tuple[str, ...] = ("user", "admin")


def _new_user_id() -> str:
    """Generate a stable, URL-safe user identifier (JWT "sub")."""
    return uuid.uuid4().hex


def _to_public(doc: Optional[dict]) -> Optional[dict]:
    """Strip internal fields (password hash, Mongo "_id") before returning."""
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("hashed_password", None)
    doc.pop("_id", None)
    return doc


class UserService:
    """Async CRUD for user accounts backed by MongoDB (or an injected collection)."""

    def __init__(
        self,
        collection: Any = None,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._uri = uri or MONGODB_URI
        self._db_name = db_name or MONGODB_DB
        self._collection_name = collection_name or AUTH_USERS_COLLECTION
        self._client: Optional[AsyncMongoClient] = None
        self._collection = collection  # injected (tests) => no real I/O
        self._index_ensured = False

    async def _get_collection(self):
        """Return the active collection, connecting lazily when needed."""
        if self._collection is not None:
            return self._collection
        if self._client is None:
            self._client = AsyncMongoClient(self._uri, serverSelectionTimeoutMS=5000)
        coll = self._client[self._db_name][self._collection_name]
        if not self._index_ensured:
            try:
                await coll.create_index("email", unique=True, name="email_unique")
                logger.info("Ensured unique index on users.email")
            except Exception as exc:
                logger.warning("Could not ensure unique email index: %s", exc)
            self._index_ensured = True
        return coll

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
        role: str = "user",
        is_active: bool = True,
    ) -> dict:
        """Persist a new user (default role "user"). Raises on duplicate email."""
        coll = await self._get_collection()
        doc = {
            "user_id": _new_user_id(),
            "email": email.lower().strip(),
            "full_name": full_name,
            "hashed_password": hash_password(password),
            "role": role if role in VALID_ROLES else "user",
            "is_active": is_active,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        try:
            await coll.insert_one(doc)
        except DuplicateKeyError:
            raise
        return _to_public(doc)

    async def get_by_email(self, email: str) -> Optional[dict]:
        coll = await self._get_collection()
        return await coll.find_one({"email": email.lower().strip()})

    async def get_by_id(self, user_id: str) -> Optional[dict]:
        coll = await self._get_collection()
        return await coll.find_one({"user_id": user_id})

    async def authenticate(self, email: str, password: str) -> Optional[dict]:
        """Verify credentials; return the full user doc or "None".

        Fails closed: unknown user, disabled account, or bad password all yield
        "None" (the caller maps this to 401 without leaking which failed).
        """
        doc = await self.get_by_email(email)
        if doc is None or not doc.get("is_active", True):
            return None
        if not verify_password(password, doc.get("hashed_password", "")):
            return None
        return doc

    async def update_role(
        self, user_id: str, role: str, is_active: Optional[bool] = None
    ) -> Optional[dict]:
        """Promote/demote a user's role and/or enable/disable the account."""
        from pymongo import ReturnDocument

        coll = await self._get_collection()
        update: dict[str, Any] = {
            "role": role if role in VALID_ROLES else "user",
            "updated_at": datetime.now(timezone.utc),
        }
        if is_active is not None:
            update["is_active"] = bool(is_active)
        result = await coll.find_one_and_update(
            {"user_id": user_id},
            {"$set": update},
            return_document=ReturnDocument.AFTER,
        )
        return _to_public(result)

    async def change_password(self, user_id: str, new_password: str) -> bool:
        """Set a new bcrypt hash for "user_id"; returns True if a row changed."""
        coll = await self._get_collection()
        result = await coll.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "hashed_password": hash_password(new_password),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        return result.modified_count > 0

    async def list_users(self, limit: int = 100) -> list[dict]:
        coll = await self._get_collection()
        out: list[dict] = []
        cursor = coll.find({}).sort("created_at", -1).limit(limit)
        async for doc in cursor:
            out.append(_to_public(doc))
        return out

    async def count_admins(self) -> int:
        coll = await self._get_collection()
        return await coll.count_documents({"role": "admin"})

    async def close(self) -> None:
        """Close the underlying async client if one was opened."""
        if self._client is not None:
            try:
                await self._client.close()
            except TypeError:
                self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton + accessor (injectable for tests)
# ---------------------------------------------------------------------------

_user_service: Optional[UserService] = None


def get_user_service(collection: Any = None) -> UserService:
    """Return a UserService, reusing a process-wide singleton.

    Args:
        collection: When supplied, a *fresh* service wrapping that collection is
            returned (tests inject a fake collection to avoid real MongoDB).
    """
    global _user_service
    if collection is not None:
        return UserService(collection=collection)
    if _user_service is None:
        _user_service = UserService()
    return _user_service
