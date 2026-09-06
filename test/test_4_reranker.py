"""
Unit tests for Module 4 Part 2: Post-Retrieval Engine - Reranker.

Validates:
  - CrossEncoderReranker initialisation with CUDA/CPU auto-detection.
  - Re-rank precision (score attachment, sorting, top_n capping).
  - Graceful degradation when cross-encoder fails.
  - Functional rerank_chunks entry point.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "text": "Apple reported Q3 revenue of 724.5 million dollars."},
    {"chunk_id": "c2", "text": "Microsoft Azure revenue reached 80.5 million in Q3."},
    {"chunk_id": "c3", "text": "NVIDIA GPU sales grew 150% sequentially."},
    {"chunk_id": "c4", "text": "Tesla earnings beat estimates with $3.2B profit."},
    {"chunk_id": "c5", "text": "Meta ad revenue increased 10% YoY."},
]


# ============================================================================
# Constructor & device detection
# ============================================================================

class TestCrossEncoderRerankerConstructor:
    """Validate CrossEncoderReranker initialisation and device detection."""

    def test_device_auto_detect_cuda(self, monkeypatch):
        """Ensure device detection prefers CUDA when available."""
        import reranker as mod
        monkeypatch.setattr(mod, "_AUTO_DEVICE", "cuda")
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker._device == "cuda"

    def test_device_auto_detect_cpu(self, monkeypatch):
        """Ensure device falls back to CPU when CUDA unavailable."""
        import reranker as mod
        monkeypatch.setattr(mod, "_AUTO_DEVICE", "cpu")
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker._device == "cpu"

    def test_explicit_device_cuda(self):
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(device="cuda")
        assert reranker._device == "cuda"

    def test_explicit_device_cpu(self):
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(device="cpu")
        assert reranker._device == "cpu"

    def test_top_n_defaults(self):
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker._top_n == 8

    def test_top_n_custom(self):
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)
        assert reranker._top_n == 3

    def test_top_n_validation(self):
        from reranker import CrossEncoderReranker
        assert CrossEncoderReranker(top_n=0)._top_n == 1
        assert CrossEncoderReranker(top_n=-5)._top_n == 1


# ============================================================================
# Re-rank logic
# ============================================================================

class TestCrossEncoderRerankerRerankLogic:
    """Validate core re-ranking functionality."""

    def test_empty_chunks(self):
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_success(self, MockCE):
        """Successful cross-encoder prediction attaches scores and sorts."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        result = reranker.rerank("Apple revenue", _SAMPLE_CHUNKS)

        assert len(result) == 5
        assert all("rerank_score" in c for c in result)
        # Should be sorted descending by score
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_top_n_capping(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)
        result = reranker.rerank("query", _SAMPLE_CHUNKS)
        assert len(result) == 3
        # Top 3 scores should be 0.9, 0.8, 0.7
        scores = [c["rerank_score"] for c in result]
        assert scores == [0.9, 0.8, 0.7]

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_overrides_top_n_param(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=2)
        result = reranker.rerank("query", _SAMPLE_CHUNKS, top_n=4)
        assert len(result) == 4

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_handles_prediction_failure(self, MockCE):
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Model error")
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", _SAMPLE_CHUNKS)
        assert len(result) == 5
        assert all(c.get("rerank_score") == 0.0 for c in result)
        assert result[0]["chunk_id"] == "c1"

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_preserves_metadata(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [
            {"chunk_id": "c1", "text": "text1", "metadata": {"ticker": "AAPL", "page": 1}},
            {"chunk_id": "c2", "text": "text2", "metadata": {"ticker": "MSFT", "page": 2}},
        ]
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", chunks)
        # Original order preserved (both scored equally by mock)
        assert result[0]["metadata"]["ticker"] == "AAPL"
        assert result[0]["metadata"]["page"] == 1


# ============================================================================
# Callable interface
# ============================================================================

class TestCrossEncoderRerankerCallInterface:
    """Validate callable and sync wrapper interfaces."""

    def test_callable_is_rerank(self):
        """__call__ is the rerank method at class level."""
        from reranker import CrossEncoderReranker
        assert CrossEncoderReranker.__call__ is CrossEncoderReranker.rerank


# ============================================================================
# Functional entry point
# ============================================================================

class TestRerankChunksFunctional:
    """Validate rerank_chunks functional interface."""

    @patch("reranker.CrossEncoderReranker")
    def test_rerank_chunks_creates_reranker(self, MockReranker):
        from reranker import rerank_chunks
        mock_instance = Mock()
        mock_instance.rerank.return_value = [{"text": "clean", "rerank_score": 0.9}]
        MockReranker.return_value = mock_instance

        result = rerank_chunks(
            "Apple revenue",
            _SAMPLE_CHUNKS,
            top_n=3,
            model_name="test/model",
            device="cpu",
        )

        MockReranker.assert_called_once_with(model_name="test/model", device="cpu", top_n=3)
        mock_instance.rerank.assert_called_once_with("Apple revenue", _SAMPLE_CHUNKS, top_n=3)
        assert result == [{"text": "clean", "rerank_score": 0.9}]

    @patch("reranker.CrossEncoderReranker")
    def test_rerank_chunks_defaults(self, MockReranker):
        from reranker import rerank_chunks
        mock_instance = Mock()
        mock_instance.rerank.return_value = []
        MockReranker.return_value = mock_instance

        rerank_chunks("query", _SAMPLE_CHUNKS)

        MockReranker.assert_called_once_with(model_name="BAAI/bge-reranker-large", device=None, top_n=8)
        mock_instance.rerank.assert_called_once_with("query", _SAMPLE_CHUNKS, top_n=8)


# ============================================================================
# Integration tests with deterministic mocks
# ============================================================================

class TestRerankerIntegration:
    """Integration tests with mocked CrossEncoder for deterministic behaviour."""

    @patch("sentence_transformers.CrossEncoder")
    def test_integration_full_pipeline(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.95, 0.85, 0.75, 0.65, 0.55]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)
        result = reranker.rerank("Apple revenue", _SAMPLE_CHUNKS)

        assert len(result) == 3
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["rerank_score"] == 0.95
        assert result[1]["chunk_id"] == "c2"
        assert result[1]["rerank_score"] == 0.85
        assert result[2]["chunk_id"] == "c3"
        assert result[2]["rerank_score"] == 0.75

    @patch("sentence_transformers.CrossEncoder")
    def test_integration_preserves_all_fields(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [
            {
                "chunk_id": "c1", "text": "text1",
                "metadata": {"ticker": "AAPL", "page": 1, "extra": "preserve"},
                "source_file": "file1.pdf", "section": "Item 7",
            },
            {
                "chunk_id": "c2", "text": "text2",
                "metadata": {"ticker": "MSFT", "page": 2},
                "source_file": "file2.pdf", "section": "Item 8",
            },
        ]
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", chunks)

        for orig, res in zip(chunks, result):
            assert res["chunk_id"] == orig["chunk_id"]
            assert res["text"] == orig["text"]
            assert res["metadata"] == orig["metadata"]
            assert res["source_file"] == orig["source_file"]
            assert res["section"] == orig["section"]


# ============================================================================
# Edge cases
# ============================================================================

class TestRerankerEdgeCases:
    """Validate edge cases and error conditions."""

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_preserves_chunk_order_on_failure(self, MockCE):
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model unavailable")
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [
            {"chunk_id": "z", "text": "z"},
            {"chunk_id": "a", "text": "a"},
            {"chunk_id": "m", "text": "m"},
        ]
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", chunks)
        assert [c["chunk_id"] for c in result] == ["z", "a", "m"]
        assert all(c.get("rerank_score") == 0.0 for c in result)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_with_missing_text_fields(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.7, 0.6]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [
            {"chunk_id": "c1", "text": "has text"},
            {"chunk_id": "c2"},  # missing 'text' field
        ]
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", chunks)
        assert len(result) == 2
        assert "rerank_score" in result[0]
        assert "rerank_score" in result[1]