"""
Test Suite for Module 4 Part 1: Hybrid Search Engine (dense + BM25 + RRF).

Covers:
  - Strict Qdrant pre-filtering: ticker / fiscal_year / section normalisation
    and forwarding to every dense search call (fiscal_year coerced to str).
  - Parallel hybrid search: single batch embed call for ALL queries, one
    Qdrant dense search per query, one pre-filtered sparse corpus fetch.
  - Balanced RRF: an exact-number BM25 hit outranks a pure semantic hit so
    financial figures stay competitive (financial-number/section-title safety).
  - Full-text & table fetching from MongoDB with missing-doc skipping.
  - Graceful degradation: empty input -> [], embed failure/size-mismatch
    disables the dense branch only, empty corpus -> [], never raises.
  - Public API: sync search() raises RuntimeError inside a live loop, the
    awaitable asearch() is the loop-safe entry point.

All external systems (Qdrant, MongoDB, embedding model) are fakes; no network,
GPU or live database is required.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

from hybrid_search import HybridSearchEngine  # noqa: E402


# ============================================================================
# Deterministic test doubles (real logic, no live infra)
# ============================================================================

def _mongo_docs() -> dict[str, dict]:
    """Mongo-style chunk documents (fiscal_year stored as str)."""
    return {
        "aapl-0001": {
            "chunk_id": "aapl-0001",
            "raw_text": (
                "Apple Inc reported revenue of 724.5 million dollars in fiscal "
                "2024. Revenue grew steadily during the year."
            ),
            "chunk_type": "text",
            "token_count": 24,
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Item 7",
            "doc_type": "10-K",
            "contains_table": False,
            "page_number": "20",
            "source_file": "aapl_10k_2024.pdf",
        },
        "aapl-0002": {
            "chunk_id": "aapl-0002",
            "raw_text": (
                "Operating cash flow and capital expenditures analysis for the "
                "fiscal year. Liquidity remains strong."
            ),
            "chunk_type": "text",
            "token_count": 18,
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Item 7A",
            "doc_type": "10-K",
            "contains_table": False,
            "page_number": "22",
            "source_file": "aapl_10k_2024.pdf",
        },
        "aapl-0003": {
            "chunk_id": "aapl-0003",
            "raw_text": "| Metric | Value |\n| Revenue | 724.5 |\n| Net income | 97.0 |",
            "chunk_type": "table",
            "token_count": 15,
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Item 8",
            "doc_type": "10-K",
            "contains_table": True,
            "page_number": "30",
            "source_file": "aapl_10k_2024.pdf",
        },
        "msft-0001": {
            "chunk_id": "msft-0001",
            "raw_text": (
                "Microsoft Azure cloud revenue reached 80.5 million dollars in "
                "fiscal 2024."
            ),
            "chunk_type": "text",
            "token_count": 14,
            "ticker": "MSFT",
            "fiscal_year": "2024",
            "section": "Item 7",
            "doc_type": "10-K",
            "contains_table": False,
            "page_number": "25",
            "source_file": "msft_10k_2024.pdf",
        },
    }


def _dense_results() -> list[dict]:
    """Fixed semantic ranking: aapl-0002 on top, the exact-number chunk 2nd."""
    return [
        {"chunk_id": "aapl-0002", "score": 0.91},
        {"chunk_id": "aapl-0001", "score": 0.90},
        {"chunk_id": "aapl-0003", "score": 0.85},
    ]


class FakeQdrantIndexer:
    """Records every dense search call and returns a fixed ranked list."""

    def __init__(self, results: list[dict] | None = None):
        self.calls: list[dict] = []
        self._results = results if results is not None else _dense_results()

    def search(
        self,
        embedding,
        top_k: int = 25,
        ticker: str | None = None,
        fiscal_year: str | None = None,
        section: str | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "embedding": embedding,
                "top_k": top_k,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "section": section,
            }
        )
        # Mirror strict pre-filtering: a ticker-scoped search only returns
        # chunks belonging to that ticker (chunk ids use a lower-case prefix).
        results = self._results
        if ticker:
            results = [r for r in results if r["chunk_id"].startswith(ticker.lower())]
        return results[:top_k]


class FakeMongoIndexer:
    """Real metadata filtering over an in-memory document dict."""

    def __init__(self, docs: dict[str, dict] | None = None):
        self._docs = docs if docs is not None else _mongo_docs()
        self.filter_calls: list[dict] = []
        self.ids_calls: list[list[str]] = []

    def get_chunks_by_filter(self, metadata_filter: dict | None = None, limit: int = 500):
        self.filter_calls.append({"filter": metadata_filter, "limit": limit})
        flt = metadata_filter or {}
        matched = [
            doc
            for doc in self._docs.values()
            if all(
                flt.get(k) is None or doc.get(k) == flt.get(k)
                for k in ("ticker", "fiscal_year", "section")
            )
        ]
        return matched[: max(1, int(limit))]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        self.ids_calls.append(list(chunk_ids))
        return {cid: self._docs[cid] for cid in chunk_ids if cid in self._docs}


class FakeEmbedder:
    """Deterministic batch embedder that records each (single) call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)) * 0.01 + i * 0.001] for i, t in enumerate(texts)]


def _make_engine(
    top_k: int = 25,
    candidate_k: int = 25,
    embed_fn=None,
    qdrant=None,
    mongo=None,
    **kwargs,
) -> HybridSearchEngine:
    qdrant = qdrant if qdrant is not None else FakeQdrantIndexer()
    mongo = mongo if mongo is not None else FakeMongoIndexer()
    embed_fn = embed_fn if embed_fn is not None else FakeEmbedder()
    return HybridSearchEngine(
        qdrant_indexer=qdrant,
        mongo_indexer=mongo,
        embed_fn=embed_fn,
        top_k=top_k,
        candidate_k=candidate_k,
        **kwargs,
    )


_QUERY = "Apple revenue 724.5 million"


# ============================================================================
# Constructor & basic guards
# ============================================================================

class TestConstructorAndGuards:
    def test_requires_embed_fn(self):
        with pytest.raises(ValueError):
            HybridSearchEngine(
                qdrant_indexer=FakeQdrantIndexer(),
                mongo_indexer=FakeMongoIndexer(),
            )

    def test_empty_input_returns_empty_list(self):
        engine = _make_engine()
        assert engine.search([]) == []
        assert engine.search(None) == []

    def test_whitespace_only_queries_return_empty_list(self):
        engine = _make_engine()
        assert engine.search(["   ", ""]) == []
        assert engine.search([" valid ", "   "])  # valid queries survive stripping


# ============================================================================
# Strict Qdrant pre-filtering
# ============================================================================

class TestStrictPrefiltering:
    def test_metadata_filter_normalised_and_forwarded(self):
        qdrant = FakeQdrantIndexer()
        engine = _make_engine(
            qdrant=qdrant,
            mongo=FakeMongoIndexer(_mongo_docs()),
        )
        engine.search(
            [_QUERY],
            {"ticker": " aapl ", "fiscal_year": 2024, "section": "item 7"},
        )
        assert len(qdrant.calls) == 1
        call = qdrant.calls[0]
        assert call["ticker"] == "AAPL"
        assert call["fiscal_year"] == "2024"  # int coerced to str
        assert call["section"] == "item 7"
        assert call["top_k"] == 25

    def test_unknown_filter_keys_are_dropped(self):
        qdrant = FakeQdrantIndexer()
        engine = _make_engine(qdrant=qdrant)
        engine.search([_QUERY], {"ticker": "aapl", "bogus": "x", "page_number": 3})
        call = qdrant.calls[0]
        assert call["ticker"] == "AAPL"
        assert call["fiscal_year"] is None
        assert call["section"] is None

    def test_no_filter_sends_none_to_qdrant(self):
        qdrant = FakeQdrantIndexer()
        engine = _make_engine(qdrant=qdrant)
        engine.search([_QUERY])
        call = qdrant.calls[0]
        assert call["ticker"] is None
        assert call["fiscal_year"] is None
        assert call["section"] is None

    def test_ticker_scope_limits_sparse_corpus(self):
        qdrant = FakeQdrantIndexer()
        mongo = FakeMongoIndexer(_mongo_docs())
        engine = _make_engine(qdrant=qdrant, mongo=mongo)
        result = engine.search([_QUERY], {"ticker": "MSFT"})
        assert len(mongo.filter_calls) == 1
        assert mongo.filter_calls[0]["filter"] == {"ticker": "MSFT"}
        assert qdrant.calls[0]["ticker"] == "MSFT"
        assert [o["chunk_id"] for o in result] == ["msft-0001"]

    def test_filter_unit_normalisation(self):
        normalized = HybridSearchEngine._normalise_filter(
            {"ticker": " aapl ", "fiscal_year": 2024, "section": "Item 7", "junk": 1}
        )
        assert normalized == {"ticker": "AAPL", "fiscal_year": "2024", "section": "Item 7"}

    def test_build_qdrant_filter_matches_field_conditions(self):
        qf = HybridSearchEngine._build_qdrant_filter(
            {"ticker": "AAPL", "fiscal_year": "2024"}
        )
        assert isinstance(qf, Filter)
        assert len(qf.must) == 2
        for cond in qf.must:
            assert isinstance(cond, FieldCondition)
            assert isinstance(cond.match, MatchValue)
        assert HybridSearchEngine._build_qdrant_filter({}) is None


# ============================================================================
# Parallel hybrid search & batch embedding
# ============================================================================

class TestParallelSearch:
    def test_embed_called_once_for_all_queries(self):
        embed = FakeEmbedder()
        qdrant = FakeQdrantIndexer()
        engine = _make_engine(embed_fn=embed, qdrant=qdrant)
        engine.search(["q1", "q2", "q3"], {"ticker": "AAPL"})
        assert embed.calls == [["q1", "q2", "q3"]]  # SINGLE batch pass
        assert len(qdrant.calls) == 3                # one dense search per query
        assert len({id(c["embedding"]) for c in qdrant.calls}) <= 3

    def test_sparse_corpus_fetched_once(self):
        mongo = FakeMongoIndexer(_mongo_docs())
        engine = _make_engine(mongo=mongo)
        engine.search(["a", "b"], {"ticker": "AAPL"})
        assert len(mongo.filter_calls) == 1

    def test_sparse_corpus_limit_forwarded(self):
        mongo = FakeMongoIndexer(_mongo_docs())
        engine = _make_engine(mongo=mongo, sparse_corpus_limit=2)
        engine.search([_QUERY], {"ticker": "AAPL"})
        assert mongo.filter_calls[0]["limit"] == 2


# ============================================================================
# Balanced RRF: exact-number / section-title safety
# ============================================================================

class TestBalancedRRF:
    def test_exact_number_bm25_hit_outranks_pure_semantic_hit(self):
        engine = _make_engine()
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        chunks = [o["chunk_id"] for o in result]
        # aapl-0001 (contains "724.5") is dense rank 2 but sparse rank 1, so the
        # balanced RRF lifts it above dense rank-1 aapl-0002.
        assert chunks[0] == "aapl-0001"
        assert chunks[1] == "aapl-0003"
        assert chunks[2] == "aapl-0002"
        assert [o["rrf_score"] for o in result] == sorted(
            [o["rrf_score"] for o in result], reverse=True
        )

    def test_output_schema(self):
        engine = _make_engine()
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        for chunk in result:
            assert set(chunk.keys()) == {
                "chunk_id",
                "text",
                "chunk_type",
                "token_count",
                "metadata",
                "rrf_score",
                "dense_score",
                "sparse_score",
            }
            assert {"ticker", "fiscal_year", "section"} <= set(chunk["metadata"])

    def test_sparse_score_reported_for_exact_hit(self):
        engine = _make_engine()
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        top = result[0]
        assert top["dense_score"] == 0.90
        assert top["sparse_score"] is not None and top["sparse_score"] > 0.0

    def test_top_k_caps_output(self):
        engine = _make_engine(top_k=2)
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        assert len(result) == 2

    def test_rrf_math_static(self):
        fused = HybridSearchEngine._reciprocal_rank_fusion(
            [[{"chunk_id": "c1", "score": 1.0}]],
            [[{"chunk_id": "c2", "score": 2.0}]],
            k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
        )
        assert fused == {"c1": 1 / 61, "c2": 1 / 61}

    def test_chunk_in_both_families_scores_higher(self):
        fused = HybridSearchEngine._reciprocal_rank_fusion(
            [[{"chunk_id": "c1", "score": 1.0}, {"chunk_id": "c2", "score": 0.5}]],
            [[{"chunk_id": "c1", "score": 3.0}]],
            k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
        )
        assert fused["c1"] > fused["c2"]

    def test_best_scores_picks_max_across_lists(self):
        best = HybridSearchEngine._best_scores(
            [
                [{"chunk_id": "c1", "score": 0.5}, {"chunk_id": "c2", "score": 0.9}],
                [{"chunk_id": "c1", "score": 0.8}],
            ]
        )
        assert best == {"c1": 0.8, "c2": 0.9}


# ============================================================================
# Full-text & table fetching from MongoDB
# ============================================================================

class TestMongoFetching:
    def test_table_chunk_materialised(self):
        engine = _make_engine()
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        table = next(o for o in result if o["chunk_id"] == "aapl-0003")
        assert table["chunk_type"] == "table"
        assert "| Metric | Value |" in table["text"]  # markdown table preserved
        assert table["metadata"]["contains_table"] is True

    def test_missing_mongo_docs_are_skipped(self):
        qdrant = FakeQdrantIndexer(
            results=[
                {"chunk_id": "aapl-ghost-0001", "score": 0.99},
                {"chunk_id": "aapl-0001", "score": 0.90},
            ]
        )
        mongo = FakeMongoIndexer(_mongo_docs())
        engine = _make_engine(qdrant=qdrant, mongo=mongo)
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        assert "aapl-ghost-0001" not in [o["chunk_id"] for o in result]
        assert "aapl-0001" in [o["chunk_id"] for o in result]


# ============================================================================
# Graceful degradation (never raises)
# ============================================================================

class TestGracefulDegradation:
    def test_embed_exception_disables_dense_only(self):
        def bad_embed(_texts):
            raise RuntimeError("embedder down")

        qdrant = FakeQdrantIndexer()
        engine = _make_engine(embed_fn=bad_embed, qdrant=qdrant)
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        assert qdrant.calls == []            # dense never ran
        assert len(result) >= 1              # sparse survived
        assert result[0]["dense_score"] is None

    def test_embed_size_mismatch_disables_dense(self):
        engine = _make_engine(
            embed_fn=lambda texts: [[0.1]] * (len(texts) - 1)
        )
        result = engine.search([_QUERY, "revenue"], {"ticker": "AAPL"})
        assert len(result) >= 1

    def test_empty_sparse_corpus_returns_empty(self):
        mongo = FakeMongoIndexer(docs={})
        engine = _make_engine(mongo=mongo)
        assert engine.search([_QUERY], {"ticker": "AAPL"}) == []

    def test_qdrant_exception_keeps_sparse(self):
        class FailingQdrant(FakeQdrantIndexer):
            def search(self, *args, **kwargs):
                raise ConnectionError("qdrant down")

        engine = _make_engine(qdrant=FailingQdrant())
        result = engine.search([_QUERY], {"ticker": "AAPL"})
        assert len(result) >= 1  # sparse branch still returns results


# ============================================================================
# Async / sync API contract
# ============================================================================

class TestAsyncContract:
    def test_asearch_awaitable_returns_results(self):
        engine = _make_engine()
        result = asyncio.run(engine.asearch([_QUERY], {"ticker": "AAPL"}))
        assert result and result[0]["chunk_id"] == "aapl-0001"

    def test_sync_search_raises_inside_live_loop(self):
        engine = _make_engine()

        async def _call_sync():
            return engine.search([_QUERY])

        with pytest.raises(RuntimeError):
            asyncio.run(_call_sync())

    def test_sync_search_returns_results_outside_loop(self):
        engine = _make_engine()
        assert engine.search([_QUERY], {"ticker": "AAPL"})[0]["chunk_id"] == "aapl-0001"


# ============================================================================
# Tokenizer: exact financial-number matching
# ============================================================================

class TestTokenizer:
    def test_preserves_decimal_figures(self):
        tokens = HybridSearchEngine._tokenize(
            "Revenue was 724.5 million and 1.2x growth in FY2024 (0.75 ratio)"
        )
        assert "724.5" in tokens
        assert "1.2" in tokens
        assert "0.75" in tokens
        assert "fy2024" in tokens  # "FY2024" lowercased as one token
        assert "724" not in tokens

    def test_lowercases_text(self):
        assert HybridSearchEngine._tokenize("Apple") == ["apple"]
