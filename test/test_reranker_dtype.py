"""
Focused unit tests for the configurable reranker dtype feature.

Covers:
  - dtype configuration parsing (default + explicit override)
  - FP32 vs FP16 resolution on CUDA
  - CPU fallback to FP32 when float16 requested without CUDA
  - invalid dtype handling (ValueError)
  - torch_dtype forwarded to CrossEncoder model_kwargs
  - score output shape/type
  - ranking consistency (top-n preserved)
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

import torch  # noqa: E402


class TestRerankerDTypeConfig:
    """dtype configuration parsing."""

    def test_default_from_settings(self):
        import config.settings as settings
        import reranker as mod

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(settings, "RERANKER_DTYPE", "float16")
        # reload so the default arg picks up the patched value
        monkeypatch.setitem(sys.modules, "reranker", importlib.reload(mod))

        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda")
        assert rr._requested_dtype == "float16"
        monkeypatch.undo()

    def test_explicit_override(self):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="float32")
        assert rr._requested_dtype == "float32"

    def test_case_insensitive_and_whitespace(self):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="  Float16 ")
        assert rr._requested_dtype == "float16"

    def test_invalid_dtype_raises(self):
        from reranker import CrossEncoderReranker
        with pytest.raises(ValueError):
            CrossEncoderReranker(device="cuda", dtype="bfloat16")


class TestRerankerDTypeResolution:
    """Effective dtype resolution at load time."""

    @patch("sentence_transformers.CrossEncoder")
    def test_fp32_on_cuda(self, MockCE):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="float32")
        rr._lazy_init()
        assert rr._effective_dtype == "float32"
        assert rr._dtype_fallback_reason is None
        # torch_dtype forwarded to the underlying model load
        _, kwargs = MockCE.call_args
        assert kwargs["model_kwargs"]["torch_dtype"] == torch.float32

    @patch("sentence_transformers.CrossEncoder")
    def test_fp16_on_cuda(self, MockCE):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="float16")
        rr._lazy_init()
        assert rr._effective_dtype == "float16"
        assert rr._dtype_fallback_reason is None
        _, kwargs = MockCE.call_args
        assert kwargs["model_kwargs"]["torch_dtype"] == torch.float16

    @patch("sentence_transformers.CrossEncoder")
    def test_fp16_on_cpu_falls_back_to_fp32(self, MockCE):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cpu", dtype="float16")
        rr._lazy_init()
        assert rr._effective_dtype == "float32"
        assert rr._dtype_fallback_reason == "CPU fallback"
        _, kwargs = MockCE.call_args
        assert kwargs["model_kwargs"]["torch_dtype"] == torch.float32

    @patch("sentence_transformers.CrossEncoder")
    def test_fp32_on_cpu(self, MockCE):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cpu", dtype="float32")
        rr._lazy_init()
        assert rr._effective_dtype == "float32"
        assert rr._dtype_fallback_reason is None
        _, kwargs = MockCE.call_args
        assert kwargs["model_kwargs"]["torch_dtype"] == torch.float32


class TestRerankerDTypeInference:
    """Score output shape/type and ranking consistency under dtype config."""

    @patch("sentence_transformers.CrossEncoder")
    def test_score_output_shape_and_type(self, MockCE):
        import numpy as np
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(5)]
        rr = CrossEncoderReranker(device="cuda", dtype="float16")
        result = rr.rerank("query", chunks)
        assert len(result) == 5
        assert all(isinstance(c["rerank_score"], float) for c in result)
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    @patch("sentence_transformers.CrossEncoder")
    def test_ranking_top_n_equal_across_dtypes(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        MockCE.return_value = mock_model

        from reranker import CrossEncoderReranker
        chunks = [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(8)]
        rr16 = CrossEncoderReranker(device="cuda", dtype="float16")
        rr32 = CrossEncoderReranker(device="cuda", dtype="float32")
        r16 = rr16.rerank("query", chunks, top_n=8)
        r32 = rr32.rerank("query", chunks, top_n=8)
        assert [c["chunk_id"] for c in r16] == [c["chunk_id"] for c in r32]


class TestRerankerDTypeEdgeCases:
    """Empty / single candidate handling with dtype configured."""

    @patch("sentence_transformers.CrossEncoder")
    def test_empty_candidates(self, MockCE):
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="float16")
        assert rr.rerank("query", []) == []

    @patch("sentence_transformers.CrossEncoder")
    def test_single_candidate(self, MockCE):
        mock_model = Mock()
        mock_model.predict.return_value = [0.5]
        MockCE.return_value = mock_model
        from reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(device="cuda", dtype="float16")
        result = rr.rerank("query", [{"chunk_id": "c1", "text": "t1"}])
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.5