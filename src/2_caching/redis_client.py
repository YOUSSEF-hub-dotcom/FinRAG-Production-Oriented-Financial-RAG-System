"""
Redis connection manager for the semantic caching layer.

Centralizes lazy Redis connections — both synchronous (pipeline fast-path)
and asynchronous (background guardrail write) — with graceful degradation so
the whole system keeps working when Redis is unavailable (the cache silently
disables itself and every operation returns a miss / no-op).
"""

import asyncio
from typing import Any

from config.logging_config import get_logger
from config.settings import REDIS_URL

logger = get_logger("caching.redis")


class RedisClient:
    """Lazy Redis client manager (sync + async) with graceful failure handling.

    State semantics (per connection flavour):
      - None  -> not connected yet
      - False -> Redis is unavailable (connection refused / timeout)
      - object -> connected redis client
    """

    def __init__(
        self,
        url: str = REDIS_URL,
        socket_connect_timeout: int = 3,
        decode_responses: bool = True,
    ):
        self._url = url
        self._socket_connect_timeout = socket_connect_timeout
        self._decode_responses = decode_responses
        self._client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------------
    # Synchronous flavour
    # ------------------------------------------------------------------

    def get(self) -> Any:
        """Return a connected sync Redis client, or False when unavailable."""
        if self._client is not None:
            return self._client
        try:
            import redis

            client = redis.from_url(
                self._url,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=self._decode_responses,
            )
            client.ping()
            self._client = client
            logger.info("Redis connected (sync): %s", self._url)
        except Exception as exc:
            logger.warning("Redis unavailable (cache disabled): %s", exc)
            self._client = False
        return self._client

    # ------------------------------------------------------------------
    # Asynchronous flavour
    # ------------------------------------------------------------------

    async def aget(self) -> Any:
        """Return a connected async Redis client, or False when unavailable."""
        if self._async_client is not None:
            return self._async_client
        try:
            import redis.asyncio as asyncio_redis

            client = asyncio_redis.from_url(
                self._url,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=self._decode_responses,
            )
            await client.ping()
            self._async_client = client
            logger.info("Redis connected (async): %s", self._url)
        except Exception as exc:
            logger.warning("Redis unavailable (cache disabled, async): %s", exc)
            self._async_client = False
        return self._async_client

    # ------------------------------------------------------------------
    # Availability + teardown
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when a live sync Redis connection can be established."""
        return self.get() is not False

    async def aavailable(self) -> bool:
        """True when a live async Redis connection can be established."""
        return (await self.aget()) is not False

    def close(self) -> None:
        """Close both the sync and async connections, if any.

        The async connection is closed synchronously (best effort) only when
        no event loop is currently running; inside a running loop the async
        connection is left for the event-loop shutdown, which is the only
        safe option from a synchronous context.
        """
        if self._client and self._client is not False:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

        if self._async_client and self._async_client is not False:
            self._close_async_best_effort()
        self._async_client = None

    def _close_async_best_effort(self) -> None:
        """Try to await the async connection close; ignore every failure."""
        try:
            asyncio.get_running_loop()
            return
        except RuntimeError:
            pass
        try:
            asyncio.run(self._async_client.aclose())
        except Exception:
            pass

    async def aclose(self) -> None:
        """Close the async connection from an async context (awaits aclose)."""
        if self._async_client and self._async_client is not False:
            try:
                await self._async_client.aclose()
            except Exception:
                pass
        self._async_client = None
