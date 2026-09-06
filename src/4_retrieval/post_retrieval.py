"""
Module 4 -- Retrieval Stage: Post-Retrieval Pipeline (Entry Point).

Composes the three post-retrieval stages into a single callable pipeline:

    Hybrid Search (40 chunks)
        -> Cross-Encoder Reranker (Top 8)
        -> Table Shield (Async Clean)
        -> Cylinder Reorder (dynamic: best at head, 2nd-best at tail)

Public API:
    - PostRetrievalPipeline: configurable orchestrator (callable class).
    - run_post_retrieval: functional one-shot entry point.

Stages (from `src/4_retrieval/` Part 2):
    - reranker.CrossEncoderReranker        -- BAAI/bge-reranker-large, top_n=8
    - table_shield.TableShield             -- strict table guard + async LLM clean
    - cylinder_reorder.cylinder_reorder    -- Lost-in-the-Middle staggered reorder
"""

import asyncio
from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import (
    POST_RETRIEVAL_INPUT_CHUNKS,
    POST_RETRIEVAL_RERANK_TOP_N,
)

from cylinder_reorder import cylinder_reorder
from reranker import CrossEncoderReranker
from table_shield import TableShield

logger = get_logger("retrieval.post_retrieval")


class PostRetrievalPipeline:
    """
    Rerank -> Table Shield -> Cylinder Reorder orchestrator.

    Consumes the 40 fully-fetched chunks emitted by HybridSearchEngine and
    returns the top-8 cleaned, table-shielded, cylinder-reordered chunks,
    preserving 100% of the original metadata on every chunk.

    Args:
        reranker: CrossEncoderReranker instance (or compatible `rerank(query,
            chunks, top_n=...) -> list[dict]`). Defaults to a fresh instance.
        table_shield: TableShield instance (or compatible async
            `shield(chunks) -> list[dict]`). Defaults to a fresh instance.
        input_chunks: Expected number of chunks entering the pipeline
            (default 40 -- the hybrid search RRF cap).
        top_n: Number of chunks kept after re-ranking (default 8).
        pattern: Cylinder reorder pattern as 0-based indices. When None the
            dynamic per-length cylinder layout is used (best at head,
            second-best at tail). Default None.
    """

    def __init__(
        self,
        reranker: Optional[Any] = None,
        table_shield: Optional[Any] = None,
        input_chunks: int = POST_RETRIEVAL_INPUT_CHUNKS,
        top_n: int = POST_RETRIEVAL_RERANK_TOP_N,
        pattern: Optional[tuple[int, ...]] = None,
    ):
        self._top_n = max(1, int(top_n))
        self._input_chunks = max(1, int(input_chunks))
        self._pattern = pattern
        self._reranker = reranker or CrossEncoderReranker(top_n=self._top_n)
        self._table_shield = table_shield or TableShield()

        logger.info(
            "PostRetrievalPipeline ready: input=%d top_n=%d pattern=%s",
            self._input_chunks,
            self._top_n,
            self._pattern,
        )

    def warm_up(self) -> None:
        """Pre-load the shared cross-encoder reranker (the same instance that
        serves requests) so the first query does not pay its model-load cost.
        Raises on failure so startup warm-up is honest about it."""
        self._reranker.warm_up()

    async def aprocess(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Asynchronous end-to-end post-retrieval processing.

        Args:
            query: Original user query (used by the cross-encoder reranker).
            chunks: Raw chunks from HybridSearchEngine (typically 40).

        Returns:
            Top-8 cleaned, table-shielded, cylinder-reordered chunks with
            complete metadata preserved. When fewer than two chunks are
            available the cylinder stage is skipped defensively so the
            pipeline never crashes on sparse corpora.
        """
        if not chunks:
            logger.warning("PostRetrievalPipeline called with empty chunk list")
            return []

        # Stage 1: Cross-Encoder Reranker -> top_n (default 8).
        reranked = self._reranker.rerank(query, chunks, top_n=self._top_n)
        if not reranked:
            logger.warning("Reranker returned no chunks -- post-retrieval aborted")
            return []
        return await self._postprocess(query, chunks, reranked)

    async def aprocess_scored(
        self,
        query: str,
        chunks: list[dict],
        scores: list[float],
    ) -> list[dict]:
        """End-to-end post-retrieval using precomputed cross-encoder `scores`
        (aligned to `chunks`) instead of running a fresh predict.

        This is the batched helper used by parallel multi-ticker retrieval: a
        single shared GPU predict scores every ticker's chunks at once, and this
        method then applies the IDENTICAL per-ticker selection + rescue + table
        shield + cylinder reorder that `aprocess` performs, so results are
        byte-for-byte equivalent.
        """
        if not chunks:
            logger.warning("PostRetrievalPipeline called with empty chunk list")
            return []

        reranked = self._reranker.rerank_with_scores(
            chunks,
            scores,
            top_n=self._top_n,
        )
        if not reranked:
            logger.warning("Reranker returned no chunks -- post-retrieval aborted")
            return []
        return await self._postprocess(query, chunks, reranked)

    async def _postprocess(
        self,
        query: str,
        chunks: list[dict],
        reranked: list[dict],
    ) -> list[dict]:
        """Post-rerank stages shared by `aprocess` and `aprocess_scored`:
        rescue guarantee -> table shield -> cylinder reorder."""
        # Stage 1b: Guarantee rescue. Segment/table rescue (see
        # HybridSearchEngine._rescue_table_chunks) marks the authoritative
        # segment table chunk with `rescued_table=True`, but the cross-encoder
        # reranker can still drop it because table chunks carry junk headers.
        # Force any rescued chunk into the top_n so the LLM sees the figures.
        rescued = [c for c in chunks if c.get("rescued_table")]
        if rescued:
            kept_ids = {c.get("chunk_id") for c in reranked}
            promoted = [c for c in rescued if c.get("chunk_id") not in kept_ids]
            reranked = promoted + reranked

        # Stage 2: Table Shield (tables passthrough; non-tables async clean).
        shielded = await self._table_shield.shield(reranked)

        # Stage 3: Cylinder Reorder -- dynamic pattern built for the actual
        # count (best at head, 2nd-best at tail). A single surviving chunk
        # (sparse corpus) degrades gracefully to shielded order.
        if len(shielded) >= 2:
            reordered = cylinder_reorder(shielded, self._pattern)
        else:
            logger.warning(
                "Cylinder reorder skipped: got %d chunks -- returning shielded order",
                len(shielded),
            )
            reordered = shielded

        logger.info(
            "Post-retrieval complete: %d chunks in -> %d chunks out",
            len(chunks),
            len(reordered),
        )
        return reordered

    def process(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Synchronous entry point (manages its own event loop via asyncio.run).

        Raises:
            RuntimeError: When called from inside an already-running event
                loop -- use the awaitable `aprocess` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aprocess(query, chunks))
        raise RuntimeError(
            "PostRetrievalPipeline.process() cannot run inside a live event "
            "loop; use await aprocess() instead."
        )

    __call__ = process


async def run_post_retrieval(
    query: str,
    chunks: list[dict],
    top_n: int = POST_RETRIEVAL_RERANK_TOP_N,
    pattern: Optional[tuple[int, ...]] = None,
    reranker: Optional[Any] = None,
    table_shield: Optional[Any] = None,
) -> list[dict]:
    """
    Functional one-shot entry point for the post-retrieval pipeline.

    Args:
        query: Original user query.
        chunks: Raw chunks from HybridSearchEngine (typically 40).
        top_n: Number of chunks kept after re-ranking (default 8).
        pattern: Optional custom cylinder reorder pattern (default None uses
            the dynamic per-length cylinder layout).
        reranker: Optional custom reranker instance.
        table_shield: Optional custom table shield instance.

    Returns:
        Top-8 cleaned, table-shielded, cylinder-reordered chunks.
    """
    pipeline = PostRetrievalPipeline(
        reranker=reranker,
        table_shield=table_shield,
        top_n=top_n,
        pattern=pattern,
    )
    return await pipeline.aprocess(query, chunks)
