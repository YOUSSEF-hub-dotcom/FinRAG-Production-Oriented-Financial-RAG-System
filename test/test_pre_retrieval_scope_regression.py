"""Regression tests for the pre-retrieval scope initialisation bug.

Bug: on the ``enable_pre_retrieval=True`` path, ``FinancialRAGPipeline.query()``
and ``FinancialRAGPipeline.query_stream()`` referenced ``resolved_ticker`` /
``inherited_year`` before assignment (those names are only assigned on the
legacy, pre-retrieval-disabled path). Route 2 (GENERAL) and Route 3 (REWRITE)
therefore raised ``UnboundLocalError``.

These tests pin the fix: both routes must execute end-to-end (retrieval ->
generation) without raising, for the sync and the streaming APIs.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    str(_ROOT),
    str(_ROOT / "src"),
    str(_ROOT / "src" / "1_ingestion"),
    str(_ROOT / "src" / "3_pre_retrieval"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pre_retrieval_schemas import PreRetrievalResult  # noqa: E402


def _make_qdrant_results(count: int = 2) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i:03d}",
            "score": 0.90 - i * 0.05,
            "payload": {
                "chunk_id": f"chunk_{i:03d}",
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Item 7",
                "doc_type": "10-K",
                "contains_table": False,
            },
        }
        for i in range(count)
    ]


def _make_mongo_docs(results: list[dict]) -> dict[str, dict]:
    return {
        r["chunk_id"]: {
            "chunk_id": r["chunk_id"],
            "raw_text": (
                f"Apple reported financial details in chunk {r['chunk_id']}. "
                "Revenue grew steadily and operating expenses remained stable."
            ),
        }
        for r in results
    }


@pytest.fixture
def pr_pipeline():
    """Pipeline with pre-retrieval enabled and all externals mocked."""
    with (
        patch("pipeline.PreRetrievalOrchestrator") as MockOrch,
        patch("pipeline.EmbeddingEngine") as MockEmbed,
        patch("pipeline.QdrantIndexer") as MockQdrant,
        patch("pipeline.MongoDBIndexer") as MockMongo,
        patch("pipeline.FinancialRAGGenerator") as MockGen,
        patch("pipeline.AsyncGuardrail") as MockGuard,
        patch("pipeline.SemanticCache") as MockCache,
        patch("pipeline.mlflow.set_experiment"),
        patch("pipeline.mlflow.start_run"),
        patch("pipeline.mlflow.end_run"),
        patch("pipeline.mlflow.log_param"),
        patch("pipeline.mlflow.log_metric"),
    ):
        embed = MockEmbed.return_value
        embed.embed.return_value = [[0.1] * 768] * 2
        embed.embed_single.return_value = [0.1] * 768

        qdrant = MockQdrant.return_value
        qdrant.search.return_value = _make_qdrant_results(2)
        qdrant.count_points.return_value = 110

        mongo = MockMongo.return_value
        mongo.get_chunks_by_ids.return_value = _make_mongo_docs(
            _make_qdrant_results(2)
        )
        mongo.count_documents.return_value = 110

        gen = MockGen.return_value
        gen.generate.return_value = {
            "raw_output": "{}",
            "parsed": None,
            "model_used": "test-model",
            "fallback_triggered": False,
            "ttft_ms": 1.0,
        }

        async def _stream(query, retrieved_docs, **kwargs):
            yield "token1"
            yield "token2"

        gen.stream_tokens = _stream

        cache = MockCache.return_value
        cache.get.return_value = None

        guard = MockGuard.return_value
        guard.check.return_value = {"passed": True}

        from pipeline import FinancialRAGPipeline

        pipeline = FinancialRAGPipeline(
            qdrant_path="/tmp/test_pr_scope_regression",
            mongo_db="test_db",
            mongo_collection="test_collection",
            top_k=3,
            enable_cache=False,
            enable_guardrail=False,
            enable_pre_retrieval=True,
        )
        yield {
            "pipeline": pipeline,
            "pre_retrieval": MockOrch.return_value,
            "generator": gen,
            "qdrant": qdrant,
        }


_GENERAL = PreRetrievalResult(
    queries=[],
    metadata_filter={},
    is_safe=True,
    action="GENERAL",
    standalone_query="Hi!",
)

_REWRITE = PreRetrievalResult(
    queries=["What is Apple's revenue in 2024?", "top-line revenue for AAPL in 2024"],
    metadata_filter={"ticker": "AAPL", "fiscal_year": 2024},
    is_safe=True,
    action="REWRITE",
    standalone_query="What is Apple's revenue in 2024?",
    embeddings=[[0.1] * 768, [0.1] * 768],
)


class TestPreRetrievalScopeQuery:
    """Route 2 / Route 3 on the synchronous ``query()`` path."""

    def test_route2_general_does_not_raise(self, pr_pipeline):
        pr_pipeline["pre_retrieval"].process.return_value = _GENERAL
        result = pr_pipeline["pipeline"].query("Hi!")
        assert isinstance(result, dict)
        assert pr_pipeline["generator"].generate.called
        pr_pipeline["qdrant"].search.assert_not_called()

    def test_route3_rewrite_does_not_raise(self, pr_pipeline):
        pr_pipeline["pre_retrieval"].process.return_value = _REWRITE
        result = pr_pipeline["pipeline"].query("What is Apple's revenue in 2024?")
        assert isinstance(result, dict)
        assert pr_pipeline["generator"].generate.called
        assert pr_pipeline["qdrant"].search.call_count >= 1


class TestPreRetrievalScopeStream:
    """Route 2/3 must also work on the streaming path."""

    @staticmethod
    def _drain(pipeline, query: str) -> list[str]:
        async def run() -> list[str]:
            out: list[str] = []
            async for tok in pipeline.query_stream(query):
                out.append(tok)
            return out

        return asyncio.run(run())

    def test_route2_general_stream_does_not_raise(self, pr_pipeline):
        pr_pipeline["pre_retrieval"].process.return_value = _GENERAL
        tokens = self._drain(pr_pipeline["pipeline"], "Hi!")
        assert tokens == ["token1", "token2"]

    def test_route3_rewrite_stream_does_not_raise(self, pr_pipeline):
        pr_pipeline["pre_retrieval"].process.return_value = _REWRITE
        tokens = self._drain(
            pr_pipeline["pipeline"], "What is Apple's revenue in 2024?"
        )
        assert tokens == ["token1", "token2"]
