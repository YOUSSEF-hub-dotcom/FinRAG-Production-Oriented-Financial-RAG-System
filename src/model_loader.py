"""
MLflow Production Model Loader for Financial RAG Pipeline.

Dynamically fetches the active Production pipeline configuration from the
MLflow Model Registry using the alias URI:
    models:/<REGISTERED_MODEL_NAME>@<Production alias>

Falls back gracefully to direct FinancialRAGPipeline instantiation when:
    - MLflow model loading is disabled via env var
    - No Production model is registered yet
    - Registry is unreachable (offline/dev mode)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config.logging_config import get_logger
from config.settings import (
    GROQ_FALLBACK_MODEL,
    GROQ_PRIMARY_MODEL,
    LLM_MAX_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_POST_RETRIEVAL,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MLFLOW_EVAL_MODEL_NAME,
    MLFLOW_EVAL_REGISTRY_URI,
    MLFLOW_MODEL_LOADING,
    MLFLOW_PRODUCTION_ALIAS,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

logger = get_logger("model_loader")

# Cache the loaded pipeline to avoid re-instantiation on every request.
_cached_pipeline: Optional[Any] = None


def _build_alias_uri(
    model_name: str = MLFLOW_EVAL_MODEL_NAME,
    alias: str = MLFLOW_PRODUCTION_ALIAS,
) -> str:
    """Build the MLflow alias URI: models:/<name>@<alias>."""
    return f"models:/{model_name}@{alias}"


def _load_production_config(
    client: MlflowClient,
    model_name: str = MLFLOW_EVAL_MODEL_NAME,
    alias: str = MLFLOW_PRODUCTION_ALIAS,
) -> Optional[dict[str, Any]]:
    """
    Load the pipeline_state artifact from the Production model version.

    Returns:
        The parsed config.json dict, or None if unavailable.
    """
    try:
        # Resolve the model version by alias
        version = client.get_model_version_by_alias(model_name, alias)
        if version is None:
            logger.warning(
                "No model version found for alias '%s' on '%s'",
                alias, model_name,
            )
            return None

        logger.info(
            "Found Production model: %s version=%s (run=%s)",
            model_name, version.version, version.run_id,
        )

        # Download the pipeline_state/config.json artifact
        artifact_uri = f"runs:/{version.run_id}/pipeline_state/config.json"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        config = json.loads(Path(local_path).read_text(encoding="utf-8"))
        logger.info("Loaded Production pipeline config: %s", config)
        return config

    except MlflowException as exc:
        logger.warning("Could not load Production config from registry: %s", exc)
        return None


def load_production_pipeline(
    model_name: str = MLFLOW_EVAL_MODEL_NAME,
    alias: str = MLFLOW_PRODUCTION_ALIAS,
    registry_uri: str = MLFLOW_EVAL_REGISTRY_URI,
) -> Optional[Any]:
    """
    Attempt to load a FinancialRAGPipeline from the MLflow Production alias.

    The registered "model" is a provenance marker, so this function:
    1. Verifies a Production model version exists in the registry
    2. Loads its pipeline_state artifact to confirm provenance
    3. Instantiates a fresh FinancialRAGPipeline (the pipeline is a stateful
       service that cannot be fully deserialized)

    Returns:
        A configured FinancialRAGPipeline instance, or None if the
        Production model is not available.
    """
    if not MLFLOW_MODEL_LOADING:
        logger.info("MLflow model loading is disabled (MLFLOW_MODEL_LOADING=false)")
        return None

    try:
        if registry_uri:
            mlflow.set_tracking_uri(registry_uri)
            mlflow.set_registry_uri(registry_uri)

        client = MlflowClient(tracking_uri=registry_uri, registry_uri=registry_uri)
        alias_uri = _build_alias_uri(model_name, alias)

        logger.info("Attempting MLflow Production model fetch: %s", alias_uri)

        config = _load_production_config(client, model_name, alias)
        if config is None:
            logger.info("No Production model found — falling back to direct init")
            return None

        # Import here to avoid circular dependency at module level
        from pipeline import FinancialRAGPipeline

        pipeline = FinancialRAGPipeline(
            enable_hybrid_retrieval=ENABLE_HYBRID_RETRIEVAL,
            enable_post_retrieval=ENABLE_POST_RETRIEVAL,
        )
        logger.info(
            "FinancialRAGPipeline loaded via MLflow Production alias "
            "(model=%s, config=%s)",
            model_name,
            json.dumps(config, default=str)[:200],
        )
        return pipeline

    except Exception as exc:
        logger.warning(
            "MLflow Production model loading failed (non-blocking): %s", exc
        )
        return None


def get_or_create_pipeline(
    model_name: str = MLFLOW_EVAL_MODEL_NAME,
    alias: str = MLFLOW_PRODUCTION_ALIAS,
    registry_uri: str = MLFLOW_EVAL_REGISTRY_URI,
) -> Any:
    """
    Get the cached Production pipeline, or create one via MLflow, or fall back.

    This is the main entry point for the backend lifespan. It implements the
    three-tier loading strategy:
        1. Return cached pipeline (warm restart)
        2. Try MLflow Production alias fetch
        3. Fall back to direct FinancialRAGPipeline() instantiation

    Returns:
        A FinancialRAGPipeline instance (guaranteed non-None).
    """
    global _cached_pipeline

    if _cached_pipeline is not None:
        return _cached_pipeline

    # Tier 2: MLflow Production alias
    pipeline = load_production_pipeline(model_name, alias, registry_uri)
    if pipeline is not None:
        _cached_pipeline = pipeline
        return pipeline

    # Tier 3: Direct instantiation (offline/dev/test mode)
    logger.info("Falling back to direct FinancialRAGPipeline instantiation")
    from pipeline import FinancialRAGPipeline

    pipeline = FinancialRAGPipeline(
        enable_hybrid_retrieval=ENABLE_HYBRID_RETRIEVAL,
        enable_post_retrieval=ENABLE_POST_RETRIEVAL,
    )
    _cached_pipeline = pipeline
    return pipeline


def invalidate_cache() -> None:
    """Clear the cached pipeline (e.g. after a model transition or hot-reload)."""
    global _cached_pipeline
    if _cached_pipeline is not None:
        try:
            _cached_pipeline.close()
        except Exception:
            pass
        _cached_pipeline = None
        logger.info("Pipeline cache invalidated")
