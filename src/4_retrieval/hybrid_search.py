"""
Module 4 -- Retrieval Stage: Hybrid Search Engine.

Part 1 implements the core parallel retrieval architecture:

  1. STRICT PRE-FILTERING (Qdrant): every dense search is scoped by the Module 3
     `metadata_filter` (ticker / fiscal_year / section) BEFORE any similarity
     computation, narrowing the search to the isolated chunk scope.
  2. PARALLEL HYBRID SEARCH (asyncio.gather): all (possibly expanded) queries
     are processed concurrently; for each query both branches run:
       - Dense retrieval: nomic-embed-text-v1.5 embeddings are produced for ALL
         queries in a SINGLE GPU/CPU batch pass, then queried against Qdrant
         with candidate_k = 40.
       - Sparse retrieval (BM25): exact keyword / number matching over the
         pre-filtered chunk corpus with candidate_k = 40.
  3. ENSEMBLE & RECIPROCAL RANK FUSION (RRF): dense + sparse ranked lists from
     every query are fused with RRF (k=60, balanced 1.0/1.0 weights) so exact
     financial numbers, metrics and section titles stay competitive with
     semantic similarity; the final output is capped at top_k = 40 chunk IDs.
  4. FULL-TEXT & TABLE FETCHING (MongoDB): the top 40 chunk_ids are
     materialised with their complete raw text / markdown table blocks and
     metadata.

Output: a `list[dict]` of the top-40 fully-fetched chunks ready for
post-retrieval processing (reranking, table shield, cylinder reordering).
"""

import asyncio
import re
from typing import Any, Callable, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from config.logging_config import get_logger
from config.settings import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_SPARSE_CORPUS_LIMIT,
    HYBRID_SPARSE_WEIGHT,
    HYBRID_TOP_K,
)

logger = get_logger("retrieval.hybrid_search")

# Batch embedder contract: list[str] -> list[list[float]].
EmbedFn = Callable[[list[str]], list[list[float]]]

# Token pattern keeps decimals intact ("724.5", "1.2", "0.75") so BM25 can match
# exact financial figures, not just their integer prefixes.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Metadata keys used for strict pre-filtering and output enrichment.
_FILTER_FIELDS = ("ticker", "fiscal_year", "section")
_METADATA_KEYS = (
    "ticker",
    "fiscal_year",
    "section",
    "doc_type",
    "contains_table",
    "page_number",
    "source_file",
)

# Per-branch candidate count (dense + sparse) kept deliberately larger than the
# final fused cap so RRF has a wide overlap pool to rank across.
DEFAULT_CANDIDATE_K = 40

# Segment / reported-segment query signals. When a query asks about a company's
# reported segments (e.g. "Data Center revenue") the figures live in a segment
# table chunk whose BM25/dense rank is often dominated by narrative chunks that
# merely mention the segment name. These keywords trigger a table-chunk rescue
# boost (see `_rescue_table_chunks`) so the table survives into the fused top-k.
_SEGMENT_KEYWORDS = (
    "data center",
    "segment",
    "intelligent cloud",
    "gaming",
    "automotive",
    "professional visualization",
    "revenue by segment",
    "by segment",
)
# Boost magnitude: RRF scores land ~0.016-0.05, so 0.5 guarantees the rescued
# table chunk ranks at the top of the fused list and reaches the reranker.
_TABLE_RESCUE_BOOST = 0.5


class HybridSearchEngine:
    """
    Parallel dense + sparse retrieval fused with Reciprocal Rank Fusion.

    Args:
        qdrant_indexer: QdrantIndexer instance used for dense vector search.
        mongo_indexer: MongoDBIndexer instance used for the sparse corpus and
            for full-text/table fetching.
        embed_fn: Batch embedder (list[str] -> list[list[float]]) backed by
            nomic-embed-text-v1.5; called ONCE per search for all queries.
        top_k: Final fused result count (default 40).
        candidate_k: Per-branch (dense/BM25) candidate count (default 40).
        rrf_k: RRF constant (default 60).
        dense_weight: RRF weight for the dense lists (default 1.0).
        sparse_weight: RRF weight for the sparse lists (default 1.0) -- balanced
            so exact numbers / section titles are not drowned out by semantic
            similarity.
        sparse_corpus_limit: Max chunks fetched for the pre-filtered BM25 corpus.
    """

    def __init__(
        self,
        qdrant_indexer: Any,
        mongo_indexer: Any,
        embed_fn: Optional[EmbedFn] = None,
        top_k: int = HYBRID_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = HYBRID_RRF_K,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        sparse_weight: float = HYBRID_SPARSE_WEIGHT,
        sparse_corpus_limit: int = HYBRID_SPARSE_CORPUS_LIMIT,
    ):
        if embed_fn is None:
            raise ValueError("HybridSearchEngine requires an embed_fn for dense retrieval")
        self._qdrant_indexer = qdrant_indexer
        self._mongo_indexer = mongo_indexer
        self._embed_fn = embed_fn
        self._top_k = max(1, int(top_k))
        self._candidate_k = max(1, int(candidate_k))
        self._rrf_k = max(1, int(rrf_k))
        self._dense_weight = float(dense_weight)
        self._sparse_weight = float(sparse_weight)
        self._sparse_corpus_limit = max(1, int(sparse_corpus_limit))
        logger.info(
            "HybridSearchEngine ready: top_k=%d candidate_k=%d rrf_k=%d dense_w=%.2f "
            "sparse_w=%.2f corpus_limit=%d",
            self._top_k,
            self._candidate_k,
            self._rrf_k,
            self._dense_weight,
            self._sparse_weight,
            self._sparse_corpus_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        queries: list[str],
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Synchronous entry point.

        Runs the async pipeline via `asyncio.run()`. Raises RuntimeError when
        called from inside an already-running event loop (use `asearch` there).

        Args:
            queries: Search queries (single or expanded) from Module 3.
            metadata_filter: Optional dict with ticker / fiscal_year / section.

        Returns:
            List of top-k fully-fetched chunk dicts.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asearch(queries, metadata_filter))
        raise RuntimeError(
            "HybridSearchEngine.search() cannot run inside a live event loop; "
            "use the awaitable asearch() instead."
        )

    async def asearch(
        self,
        queries: list[str],
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Asynchronous entry point (use inside a running event loop).

        Args:
            queries: Search queries (single or expanded) from Module 3.
            metadata_filter: Optional dict with ticker / fiscal_year / section.

        Returns:
            List of top-k fully-fetched chunk dicts.
        """
        q = [str(x).strip() for x in (queries or []) if str(x).strip()]
        if not q:
            logger.warning("Hybrid search skipped: no non-empty queries supplied")
            return []

        # 1. Strict pre-filtering -- normalise the Module 3 filter so both Qdrant
        #    payloads and MongoDB documents match (fiscal_year stored as str).
        flt = self._normalise_filter(metadata_filter)
        qdrant_filter = self._build_qdrant_filter(flt)
        logger.info(
            "Hybrid search start: queries=%d filter=%s qdrant_pre_filter=%s",
            len(q),
            flt,
            "active" if qdrant_filter is not None else "none",
        )

        # Per-entity mode: when tickers list has >1 entry, search each entity
        # separately and merge for guaranteed balanced representation.
        tickers = flt.get("tickers")
        if tickers and isinstance(tickers, list) and len(tickers) > 1:
            return await self._search_per_entity(q, flt, tickers)

        # 2. Sparse branch -- fetch the strictly pre-filtered corpus (BM25 scope).
        corpus_docs = await asyncio.to_thread(
            self._mongo_indexer.get_chunks_by_filter,
            flt,
            self._sparse_corpus_limit,
        )
        logger.info("Pre-filtered BM25 corpus: %d chunks", len(corpus_docs))
        bm25, corpus_ids, corpus_tokens = self._build_bm25(corpus_docs)

        # 3. Dense branch -- embed ALL queries in a single GPU/CPU batch pass.
        try:
            embeddings = self._embed_fn(q)
        except Exception as exc:
            logger.error("Batch embedding failed: %s", exc)
            embeddings = []
        if len(embeddings) != len(q):
            logger.error(
                "Embed batch size mismatch: expected %d got %d -- dense disabled",
                len(q),
                len(embeddings),
            )
            embeddings = []

        # 4. Parallel hybrid search -- every query runs dense + sparse concurrently.
        dense_tasks = [self._dense_search(emb, flt, qdrant_filter, self._candidate_k) for emb in embeddings]
        sparse_tasks = [
            self._sparse_search(bm25, corpus_ids, corpus_tokens, query, self._candidate_k)
            for query in q
        ]
        results = await asyncio.gather(*dense_tasks, *sparse_tasks)
        dense_lists = results[: len(dense_tasks)]
        sparse_lists = results[len(dense_tasks):]

        # 5. Ensemble -- Reciprocal Rank Fusion across all queries and both families.
        fused = self._reciprocal_rank_fusion(
            dense_lists,
            sparse_lists,
            k=self._rrf_k,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
        )
        fused = self._rescue_table_chunks(fused, q, corpus_docs)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
        logger.info(
            "RRF fused %d unique candidates -> top %d chunk ids",
            len(fused),
            len(ranked),
        )
        if not ranked:
            logger.warning("Hybrid search produced zero fused candidates")
            return []

        # 6. Full-text & table fetching from MongoDB (top-k chunk_ids).
        top_ids = [cid for cid, _ in ranked]
        docs_map = await asyncio.to_thread(self._mongo_indexer.get_chunks_by_ids, top_ids)

        best_dense = self._best_scores(dense_lists)
        best_sparse = self._best_scores(sparse_lists)

        output: list[dict] = []
        for cid, score in ranked:
            doc = docs_map.get(cid)
            if doc is None:
                logger.debug("Chunk %s missing in MongoDB -- skipped", cid)
                continue
            output.append(
                {
                    "chunk_id": cid,
                    "text": doc.get("raw_text", ""),
                    "chunk_type": doc.get("chunk_type", "text"),
                    "token_count": doc.get("token_count", 0),
                    "metadata": {k: doc.get(k) for k in _METADATA_KEYS},
                    "rrf_score": round(score, 6),
                    "dense_score": best_dense.get(cid),
                    "sparse_score": best_sparse.get(cid),
                    "rescued_table": bool(fused.get(cid, 0.0) >= _TABLE_RESCUE_BOOST),
                }
            )

        logger.info("Hybrid search complete: %d chunks returned", len(output))
        return output

    # ------------------------------------------------------------------
    # Per-entity search (cross-company queries)
    # ------------------------------------------------------------------

    async def _search_per_entity(
        self,
        queries: list[str],
        base_flt: dict,
        tickers: list[str],
    ) -> list[dict]:
        """
        Search each entity separately and merge for balanced representation.

        When a cross-company query targets multiple tickers, a single fused
        search can let one company's chunks dominate the top-k.  This method
        runs the full hybrid pipeline (dense + BM25 + RRF) independently for
        each ticker, takes the top results from each, then merges and de-dupes
        by chunk_id (keeping the best score).
        """
        per_entity_k = max(2, self._top_k // max(1, len(tickers)))
        all_results: list[dict] = []

        for ticker in tickers:
            entity_flt = {k: v for k, v in base_flt.items() if k != "tickers"}
            entity_flt["ticker"] = ticker
            qdrant_filter = self._build_qdrant_filter(entity_flt)

            logger.info("Per-entity search: ticker=%s", ticker)

            corpus_docs = await asyncio.to_thread(
                self._mongo_indexer.get_chunks_by_filter,
                entity_flt,
                self._sparse_corpus_limit,
            )
            bm25, corpus_ids, corpus_tokens = self._build_bm25(corpus_docs)

            try:
                embeddings = self._embed_fn(queries)
            except Exception as exc:
                logger.error("Per-entity embedding failed for %s: %s", ticker, exc)
                embeddings = []

            if len(embeddings) != len(queries):
                embeddings = []

            dense_tasks = [
                self._dense_search(emb, entity_flt, qdrant_filter, self._candidate_k)
                for emb in embeddings
            ]
            sparse_tasks = [
                self._sparse_search(bm25, corpus_ids, corpus_tokens, query, self._candidate_k)
                for query in queries
            ]
            results = await asyncio.gather(*dense_tasks, *sparse_tasks)
            dense_lists = results[: len(dense_tasks)]
            sparse_lists = results[len(dense_tasks):]

            fused = self._reciprocal_rank_fusion(
                dense_lists, sparse_lists,
                k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
            )
            fused = self._rescue_table_chunks(fused, queries, corpus_docs)
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[
                : per_entity_k
            ]

            if not ranked:
                logger.info("Per-entity: ticker=%s returned 0 results", ticker)
                continue

            top_ids = [cid for cid, _ in ranked]
            docs_map = await asyncio.to_thread(self._mongo_indexer.get_chunks_by_ids, top_ids)
            best_dense = self._best_scores(dense_lists)
            best_sparse = self._best_scores(sparse_lists)

            for cid, score in ranked:
                doc = docs_map.get(cid)
                if doc is None:
                    continue
                all_results.append(
                    {
                        "chunk_id": cid,
                        "text": doc.get("raw_text", ""),
                        "chunk_type": doc.get("chunk_type", "text"),
                        "token_count": doc.get("token_count", 0),
                        "metadata": {k: doc.get(k) for k in _METADATA_KEYS},
                        "rrf_score": round(score, 6),
                        "dense_score": best_dense.get(cid),
                        "sparse_score": best_sparse.get(cid),
                    }
                )

        merged: dict[str, dict] = {}
        for r in all_results:
            cid = r["chunk_id"]
            if cid not in merged or r["rrf_score"] > merged[cid]["rrf_score"]:
                merged[cid] = r

        final = sorted(merged.values(), key=lambda r: r["rrf_score"], reverse=True)[
            : self._top_k
        ]
        logger.info(
            "Per-entity search complete: %d entities -> %d merged chunks",
            len(tickers),
            len(final),
        )
        return final

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    async def _dense_search(
        self,
        embedding: list[float],
        metadata_filter: dict,
        qdrant_filter: Optional[Filter],
        top_k: int,
    ) -> list[dict]:
        """Query Qdrant with strict pre-filtering for a single query embedding."""
        try:
            results = await asyncio.to_thread(
                self._qdrant_indexer.search,
                embedding,
                top_k,
                metadata_filter.get("ticker"),
                metadata_filter.get("fiscal_year"),
                metadata_filter.get("section"),
            )
            return results or []
        except Exception as exc:
            logger.error("Dense (Qdrant) search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Sparse retrieval (BM25)
    # ------------------------------------------------------------------

    def _build_bm25(
        self, corpus_docs: list[dict]
    ) -> tuple[Optional[BM25Okapi], list[str], list[set[str]]]:
        """
        Tokenize the pre-filtered corpus and build the BM25Okapi index.

        Returns (bm25_index, chunk_id list, per-doc token sets) so the sparse
        search can rank by score while keeping only docs that actually share at
        least one query token. The token overlap gate is required because the
        ATIRE BM25 variant (rank_bm25) floors negative IDF values, which can
        push MATCHING docs to negative scores in small corpora -- filtering by
        `score > 0` would wrongly drop legitimate exact-number / section-title
        hits (financial-number safety).
        """
        tokenized = [self._tokenize(d.get("raw_text", "")) for d in corpus_docs]
        token_sets = [set(tokens) for tokens in tokenized]
        ids = [d["chunk_id"] for d in corpus_docs]
        if not ids or not any(tokens for tokens in tokenized):
            logger.warning("BM25 corpus empty after pre-filtering")
            return None, ids, token_sets
        return BM25Okapi(tokenized), ids, token_sets

    async def _sparse_search(
        self,
        bm25: Optional[BM25Okapi],
        corpus_ids: list[str],
        corpus_tokens: list[set[str]],
        query: str,
        top_k: int,
    ) -> list[dict]:
        """Exact keyword/number matching via BM25 over the pre-filtered corpus."""

        def _score() -> list[dict]:
            if bm25 is None:
                return []
            query_terms = set(self._tokenize(query))
            scores = bm25.get_scores(self._tokenize(query))
            matched = [
                (cid, float(score))
                for cid, score, toks in zip(corpus_ids, scores, corpus_tokens)
                if query_terms and toks & query_terms
            ]
            ranked = sorted(matched, key=lambda pair: pair[1], reverse=True)
            return [
                {"chunk_id": cid, "score": score}
                for cid, score in ranked[:top_k]
            ]

        try:
            return await asyncio.to_thread(_score)
        except Exception as exc:
            logger.error("Sparse (BM25) search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_lists: list[list[dict]],
        sparse_lists: list[list[dict]],
        k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, float]:
        """
        Combine every ranked list (dense + sparse, across all queries) with RRF.

        score(chunk) = sum over lists of weight * 1 / (k + rank + 1).
        Balanced weights keep exact-number BM25 hits competitive with dense hits.
        """
        fused: dict[str, float] = {}
        for ranked, weight in ((dense_lists, dense_weight), (sparse_lists, sparse_weight)):
            for hits in ranked:
                for rank, hit in enumerate(hits):
                    chunk_id = hit.get("chunk_id")
                    if not chunk_id:
                        continue
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * (
                        1.0 / (k + rank + 1)
                    )
        return fused

    @staticmethod
    def _best_scores(ranked_lists: list[list[dict]]) -> dict[str, float]:
        """Best per-chunk score across all lists of one retrieval family."""
        best: dict[str, float] = {}
        for hits in ranked_lists:
            for hit in hits:
                chunk_id = hit.get("chunk_id")
                score = float(hit.get("score", 0.0))
                if chunk_id and (chunk_id not in best or score > best[chunk_id]):
                    best[chunk_id] = score
        return best

    # ------------------------------------------------------------------
    # Table / segment rescue
    # ------------------------------------------------------------------

    def _rescue_table_chunks(
        self,
        fused: dict[str, float],
        queries: list[str],
        corpus_docs: list[dict],
    ) -> dict[str, float]:
        """
        Boost the best-matching table chunk for segment/metric-intent queries.

        Reported-segment figures (e.g. "Data Center revenue") live in a segment
        table chunk whose BM25/dense rank is frequently dominated by narrative
        chunks that merely mention the segment name. For queries that signal
        segment intent we run a BM25 pass restricted to ``contains_table``
        chunks and boost the strongest match so it survives into the fused
        top-k and reaches the post-retrieval reranker.
        """
        qtext = " ".join(str(x) for x in (queries or [])).lower()
        if not any(kw in qtext for kw in _SEGMENT_KEYWORDS):
            return fused
        table_docs = [d for d in (corpus_docs or []) if d.get("contains_table")]
        if not table_docs:
            return fused
        bm25, ids, toks = self._build_bm25(table_docs)
        if bm25 is None:
            return fused
        best_cid, best_score = None, 0.0
        for q in queries:
            qt = set(self._tokenize(q))
            if not qt:
                continue
            for cid, sc, t in zip(ids, bm25.get_scores(self._tokenize(q)), toks):
                if qt & t and sc > best_score:
                    best_score, best_cid = sc, cid
        if best_cid and best_score > 0:
            fused[best_cid] = max(fused.get(best_cid, 0.0), _TABLE_RESCUE_BOOST)
            logger.info(
                "Table/segment rescue: boosted chunk %s (bm25=%.3f)",
                best_cid, best_score,
            )
        return fused

    # ------------------------------------------------------------------
    # Filter / token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_filter(metadata_filter: Optional[dict]) -> dict:
        """
        Normalise a Module 3 metadata_filter for strict pre-filtering.

        - Keeps only the supported keys (ticker / fiscal_year / section).
        - Uppercases ticker, stringifies fiscal_year (Qdrant payload + MongoDB
          both store fiscal_year as a string).
        - Passes through tickers list for cross-company OR filtering.
        """
        normalised: dict = {}
        for key in _FILTER_FIELDS:
            value = (metadata_filter or {}).get(key)
            if value is None or str(value).strip() == "":
                continue
            if key == "ticker":
                normalised[key] = str(value).strip().upper()
            else:
                normalised[key] = str(value).strip()
        tickers = (metadata_filter or {}).get("tickers")
        if tickers and isinstance(tickers, list) and len(tickers) > 1:
            normalised["tickers"] = [str(t).strip().upper() for t in tickers if str(t).strip()]
        return normalised

    @staticmethod
    def _build_qdrant_filter(metadata_filter: dict) -> Optional[Filter]:
        """Build a Qdrant Filter from the normalised metadata_filter (or None).

        When tickers list is present, uses Filter(should=[...]) for OR matching
        across multiple companies; otherwise uses Filter(must=[...]) for exact match.
        """
        tickers = metadata_filter.get("tickers")
        if tickers and isinstance(tickers, list) and len(tickers) > 1:
            ticker_conditions = [
                FieldCondition(key="ticker", match=MatchValue(value=t))
                for t in tickers
            ]
            other_conditions = []
            for key in _FILTER_FIELDS:
                if key == "ticker":
                    continue
                value = metadata_filter.get(key)
                if value:
                    other_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if other_conditions:
                return Filter(
                    must=other_conditions,
                    should=ticker_conditions,
                )
            return Filter(should=ticker_conditions)
        conditions = []
        for key in _FILTER_FIELDS:
            value = metadata_filter.get(key)
            if value:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase tokenization preserving decimal numbers for exact matching."""
        return _TOKEN_PATTERN.findall(str(text).lower())
