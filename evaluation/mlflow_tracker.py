"""
MLflow Tracker & Automated Quality Gate for the Module 6 Evaluation pipeline.

Logs evaluation parameters, metrics, CSV artifacts and pipeline state to MLflow
under the ``Financial_RAG_Evaluation`` experiment, then enforces the automated
Model Registry quality gate:

    - Faithfulness        >= 0.85
    - Answer Relevance    >= 0.80
    - Context Precision   >= 0.75
    - Context Recall      >= 0.80

Transition policy:
    - ALL thresholds passed  -> register the pipeline artifact, set alias/stage
                                to ``Production``.
    - ANY threshold failed   -> register the version and set alias/stage to
                                ``Staging`` for investigation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config.logging_config import get_logger
from config.settings import (
    EVALUATION_RESULTS_PATH,
    EVALUATION_SCORES_PATH,
    MLFLOW_EVAL_EXPERIMENT,
    MLFLOW_EVAL_MODEL_NAME,
    MLFLOW_EVAL_REGISTRY_URI,
    QA_GATE_ANSWER_RELEVANCE,
    QA_GATE_CONTEXT_PRECISION,
    QA_GATE_CONTEXT_RECALL,
    QA_GATE_FAITHFULNESS,
    TEST_DATASET_PATH,
)

logger = get_logger("evaluation.mlflow_tracker")

PRODUCTION_ALIAS = "Production"
STAGING_ALIAS = "Staging"


@dataclass
class QualityGate:
    """
    Thresholds for the automated quality gate.

    Each threshold is the minimum acceptable score for the corresponding core
    Ragas metric (all scores are in [0.0, 1.0]).
    """

    faithfulness: float = QA_GATE_FAITHFULNESS
    answer_relevance: float = QA_GATE_ANSWER_RELEVANCE
    context_precision: float = QA_GATE_CONTEXT_PRECISION
    context_recall: float = QA_GATE_CONTEXT_RECALL

    def check(self, scores: dict[str, float]) -> "QualityGateResult":
        """
        Evaluate the given aggregate scores against the thresholds.

        Args:
            scores: Dict mapping metric name -> aggregate score.

        Returns:
            A ``QualityGateResult`` with per-metric pass/fail and overall state.
        """
        checks = {
            "faithfulness": scores.get("faithfulness", 0.0) >= self.faithfulness,
            "answer_relevance": scores.get("answer_relevance", 0.0) >= self.answer_relevance,
            "context_precision": scores.get("context_precision", 0.0) >= self.context_precision,
            "context_recall": scores.get("context_recall", 0.0) >= self.context_recall,
        }
        return QualityGateResult(
            passed=all(checks.values()),
            checks=checks,
            thresholds={
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
            },
            scores=dict(scores),
        )


@dataclass
class QualityGateResult:
    """Outcome of the automated quality gate evaluation."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the gate result for MLflow logging."""
        return {
            "quality_gate_passed": self.passed,
            "checks": self.checks,
            "thresholds": self.thresholds,
            "scores": self.scores,
        }


class MlflowTracker:
    """
    Logs the evaluation run to MLflow and transitions the registered pipeline
    model version to Production or Staging per the quality gate.
    """

    def __init__(
        self,
        experiment_name: str = MLFLOW_EVAL_EXPERIMENT,
        model_name: str = MLFLOW_EVAL_MODEL_NAME,
        registry_uri: str = MLFLOW_EVAL_REGISTRY_URI,
        quality_gate: Optional[QualityGate] = None,
    ):
        """
        Args:
            experiment_name: MLflow experiment for evaluation runs.
            model_name: Registered model name in the MLflow Model Registry.
            registry_uri: Optional MLflow tracking/registry URI override.
            quality_gate: Quality gate thresholds (default from settings).
        """
        self._experiment_name = experiment_name
        self._model_name = model_name
        self._quality_gate = quality_gate or QualityGate()
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

    # -- MLflow run helpers ----------------------------------------------------

    def _start_run(self, run_name: str) -> None:
        """Ensure the evaluation experiment exists and start a nested run."""
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=run_name, nested=True)
        logger.info("Started MLflow run %s (experiment=%s)", run_name, self._experiment_name)

    def _latest_eval_run_id(self) -> str:
        """Return the most recent run id of the evaluation experiment."""
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise MlflowException(
                f"Experiment {self._experiment_name!r} not found — nothing to transition"
            )
        runs = MlflowClient().search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise MlflowException("No evaluation runs found to transition")
        return runs[0].info.run_id

    def _log_params(self, params: dict[str, Any]) -> None:
        """Log evaluation parameters (flattening lists/dicts)."""
        for key, value in params.items():
            if isinstance(value, (list, dict, tuple)):
                mlflow.log_param(key, json.dumps(value, default=str))
            else:
                mlflow.log_param(key, value)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics, skipping non-numeric values."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

    @staticmethod
    def _log_artifacts(artifact_paths: list[Path]) -> None:
        """Log every given artifact path under the current MLflow run."""
        for path in artifact_paths:
            p = Path(path)
            if p.exists():
                mlflow.log_artifact(str(p))
                logger.info("Logged artifact %s", p)
            else:
                logger.warning("Artifact not found, skipping: %s", p)

    # -- Public API -------------------------------------------------------------

    def log_evaluation_run(
        self,
        aggregate_scores: dict[str, float],
        params: Optional[dict[str, Any]] = None,
        run_name: str = "evaluation",
        artifact_paths: Optional[list[Path]] = None,
    ) -> QualityGateResult:
        """
        Log a complete evaluation run to MLflow and return the gate verdict.

        Args:
            aggregate_scores: Mean scores for the 4 core metrics.
            params: Evaluation parameters (models, test size, thresholds...).
            run_name: MLflow run name.
            artifact_paths: Optional extra artifacts to log (defaults to the
                test dataset and evaluation result CSVs).

        Returns:
            The ``QualityGateResult`` (Production vs Staging decision).
        """
        gate_result = self._quality_gate.check(aggregate_scores)

        try:
            self._start_run(run_name)
            self._log_params(params or {})
            self._log_metrics(aggregate_scores)
            self._log_metrics({"quality_gate_passed": 1.0 if gate_result.passed else 0.0})

            artifacts = artifact_paths or [
                Path(TEST_DATASET_PATH),
                Path(EVALUATION_RESULTS_PATH),
                Path(EVALUATION_SCORES_PATH),
            ]
            self._log_artifacts(artifacts)

            mlflow.set_tag("quality_gate", "Production" if gate_result.passed else "Staging")
            mlflow.set_tag("quality_gate_passed", str(gate_result.passed))
            for metric, passed in gate_result.checks.items():
                mlflow.set_tag(f"gate_check_{metric}", str(passed))
        except MlflowException as exc:
            logger.warning("MLflow evaluation logging failed (non-blocking): %s", exc)
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

        logger.info(
            "Quality gate verdict: %s (%s)",
            "PASS -> Production" if gate_result.passed else "FAIL -> Staging",
            gate_result.checks,
        )
        return gate_result

    # -- Model Registry ---------------------------------------------------------

    def _get_or_create_registered_model(self) -> tuple[Any, MlflowClient]:
        """Return (creating if needed) the registered model for this pipeline."""
        client = MlflowClient()
        try:
            return client.get_registered_model(self._model_name), client
        except MlflowException:
            pass
        try:
            client.create_registered_model(self._model_name)
            logger.info("Registered model created: %s", self._model_name)
        except MlflowException as exc:
            logger.warning("Could not create registered model (race-safe): %s", exc)
        return client.get_registered_model(self._model_name), client

    def _register_pipeline_model(self, run_id: str) -> Any:
        """
        Register the pipeline as an MLflow model (empty-function pyfunc model)
        and return the latest model version.

        The RAG pipeline is a heavyweight stateful service, so the registered
        "model" records its immutable configuration + provenance. A pyfunc
        wrapper is provided so the artifact can be loaded and inspected while
        the actual execution path always routes through ``FinancialRAGPipeline``.
        """
        model_uri = f"runs:/{run_id}/pipeline_state"
        registered, client = self._get_or_create_registered_model()

        # Log the pipeline-state artifact under the current run first.
        try:
            mlflow.log_dict(
                {"registered_pipeline": self._model_name, "source": "Module 6 Evaluation"},
                "pipeline_state/config.json",
            )
            mlflow.log_dict(
                {"framework": "financial_rag", "version": "1.0"},
                "pipeline_state/metadata.json",
            )
        except Exception as exc:
            logger.warning("pipeline_state artifact logging failed (non-blocking): %s", exc)

        try:
            mlflow.pyfunc.log_model(
                artifact_path="pipeline_state",
                python_model=_PipelineStateModel(self._model_name),
                registered_model_name=self._model_name,
                pip_requirements=[],  # env is already provisioned; skip heavy deps
            )
        except Exception as exc:
            logger.warning(
                "pyfunc model registration failed, falling back to raw URI: %s", exc
            )
            return client.create_model_version(
                name=self._model_name,
                source=model_uri,
                run_id=run_id,
                description="Financial RAG pipeline quality-gate registration",
            )

        versions = client.search_model_versions(f"name = '{self._model_name}'")
        if not versions:
            raise MlflowException(
                f"No model versions found for {self._model_name} after registration."
            )
        latest = max(versions, key=lambda v: int(v.version))
        logger.info("Registered pipeline model version %s", latest.version)
        return latest

    def transition_model_version(
        self,
        gate_result: QualityGateResult,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Register the pipeline artifact and transition its alias/stage.

        - PASS -> alias/stage ``Production``
        - FAIL -> alias/stage ``Staging``

        Args:
            gate_result: Verdict from the quality gate.
            run_id: Active MLflow run id (defaults to the active run).

        Returns:
            The target alias/stage that was applied.
        """
        target = PRODUCTION_ALIAS if gate_result.passed else STAGING_ALIAS
        active = mlflow.active_run()
        if not run_id:
            run_id = (
                active.info.run_id
                if active is not None
                else self._latest_eval_run_id()
            )

        # ``log_evaluation_run`` ends the run before the transition, so when no
        # run is active (real execution) reopen the finished evaluation run so
        # the model registration targets it. Tests provide an active run mock.
        restarted = False
        if active is None:
            mlflow.start_run(run_id=run_id)
            restarted = True
        try:
            version = self._register_pipeline_model(run_id)
            client = MlflowClient()

            # Set alias (MLflow 2.9+ / 3.x model registry aliases).
            client.set_registered_model_alias(self._model_name, target, version.version)
            logger.info(
                "Set registered model alias '%s' -> version %s",
                target,
                version.version,
            )

            # Legacy stage transition kept for compatibility with older UIs.
            stage = "Production" if gate_result.passed else "Staging"
            try:
                client.transition_model_version_stage(
                    name=self._model_name,
                    version=version.version,
                    stage=stage,
                    archive_existing_versions=target == PRODUCTION_ALIAS,
                )
            except Exception as exc:
                logger.warning("Legacy stage transition skipped: %s", exc)
        except MlflowException as exc:
            logger.error(
                "Model Registry transition to %s failed: %s", target, exc
            )
            raise
        finally:
            if restarted:
                mlflow.end_run()
        return target

    def log_and_transition(
        self,
        aggregate_scores: dict[str, float],
        params: Optional[dict[str, Any]] = None,
        run_name: str = "evaluation",
    ) -> tuple[QualityGateResult, str]:
        """
        End-to-end helper: log the evaluation run and apply the registry gate.

        Returns:
            ``(gate_result, target_stage)`` tuple.
        """
        gate_result = self.log_evaluation_run(
            aggregate_scores=aggregate_scores,
            params=params,
            run_name=run_name,
        )
        target = self.transition_model_version(gate_result)
        return gate_result, target


class _PipelineStateModel:
    """
    Minimal pyfunc model recording the evaluated pipeline's registry identity.

    ``_PipelineStateModel`` is used purely as a registry provenance marker:
    ``predict()`` returns the pipeline configuration so artifact consumers can
    inspect what was quality-gated. Actual RAG execution always flows through
    the live ``FinancialRAGPipeline`` service.
    """

    def __init__(self, pipeline_name: str = MLFLOW_EVAL_MODEL_NAME):
        self._pipeline_name = pipeline_name

    def load_context(self, context) -> None:  # noqa: D401
        """Pyfunc load hook (no state to restore for a provenance marker)."""

    def predict(self, context, model_input):
        """Return the pipeline provenance payload."""
        return {"pipeline": self._pipeline_name, "mode": "provenance-marker"}
