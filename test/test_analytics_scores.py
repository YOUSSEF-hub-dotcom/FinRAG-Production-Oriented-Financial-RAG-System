"""Unit tests for the analytics evaluation-scores loader.

Covers ``_read_evaluation_scores``:
  - a valid top-level ``sample_count`` is preferred when present;
  - it falls back to the length of the ``samples`` list otherwise;
  - a missing file still yields the zeros/unknown fallback.
"""

import json

import pytest

from app.api import analytics


@pytest.fixture(autouse=True)
def _reset_eval_cache():
    analytics._eval_scores_cache = None
    yield
    analytics._eval_scores_cache = None


def _write_scores(tmp_path, payload):
    p = tmp_path / "evaluation_scores.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_uses_top_level_sample_count_when_present(tmp_path, monkeypatch):
    scores = {
        "samples": [{}] * 20,
        "aggregate_scores": {
            "faithfulness": 0.9,
            "answer_relevance": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.75,
        },
        "sample_count": 7,
        "judge_model": "qwen/qwen3.6-27b",
    }
    monkeypatch.setattr(
        analytics, "EVALUATION_SCORES_PATH", _write_scores(tmp_path, scores)
    )
    result = analytics._read_evaluation_scores()
    assert result["sample_count"] == 7
    assert result["faithfulness"] == 0.9


def test_sample_count_falls_back_to_samples_length(tmp_path, monkeypatch):
    scores = {
        "samples": [{}] * 25,
        "aggregate_scores": {
            "faithfulness": 0.99,
            "answer_relevance": 0.92,
            "context_precision": 0.845,
            "context_recall": 0.847,
        },
        "judge_model": "qwen/qwen3.6-27b",
    }
    monkeypatch.setattr(
        analytics, "EVALUATION_SCORES_PATH", _write_scores(tmp_path, scores)
    )
    result = analytics._read_evaluation_scores()
    assert result["sample_count"] == 25
    assert result["faithfulness"] == 0.99
    assert result["answer_relevance"] == 0.92
    assert result["context_precision"] == 0.845
    assert result["context_recall"] == 0.847
    assert result["judge_model"] == "qwen/qwen3.6-27b"


def test_missing_file_returns_zero_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        analytics, "EVALUATION_SCORES_PATH", tmp_path / "does_not_exist.json"
    )
    result = analytics._read_evaluation_scores()
    assert result == {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "sample_count": 0,
        "judge_model": "unknown",
    }