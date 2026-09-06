"""Stage 1d: log the evaluation to MLflow and apply the Production/Staging gate.

Usage: Financial_env/bin/python run_stage_1d.py [tracking_uri]
Runs from ~/Financial_RAG (project root = cwd).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import mlflow

from config.settings import (
    EVALUATION_RESULTS_PATH,
    EVALUATION_SCORES_PATH,
    TEST_DATASET_PATH,
)
from evaluation.mlflow_tracker import MlflowTracker

URI = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///mlflow.db"

if not Path(EVALUATION_SCORES_PATH).exists():
    raise SystemExit(f"Scores not found at {EVALUATION_SCORES_PATH}")

scores = json.loads(EVALUATION_SCORES_PATH.read_text(encoding="utf-8"))
aggregate = {
    m: float(scores.get(m, 0.0))
    for m in ("faithfulness", "answer_relevance", "context_precision", "context_recall")
}

params = {
    "dataset_size": len(scores.get("samples", [])),
    "test_dataset": str(TEST_DATASET_PATH),
    "evaluation_results": str(EVALUATION_RESULTS_PATH),
    "evaluation_scores": str(EVALUATION_SCORES_PATH),
    "judge_model": scores.get("judge_model") or "not-recorded",
}

mlflow.set_tracking_uri(URI)
tracker = MlflowTracker(registry_uri=URI)
gate, target = tracker.log_and_transition(aggregate_scores=aggregate, params=params)

print("\n=== STAGE 1D ===")
print(f"GATE: {'PASS' if gate.passed else 'FAIL'} -> {target}")
print(json.dumps(gate.to_dict(), indent=2))
