"""
Module 6: Evaluation & Quality Gate Engine.

Offline evaluation pipeline for the Financial RAG system:
    1. Synthetic Dataset Generation   -> evaluation/synthetic_generator.py
    2. Tracing & Execution Loop       -> evaluation/batch_runner.py
    3. LLM-as-a-Judge Evaluation      -> evaluation/judge_evaluator.py
    4. MLflow Tracking & Quality Gate -> evaluation/mlflow_tracker.py

The pipeline reads raw financial documents from ``data/`` (AAPL, MSFT, NVDA),
generates a STRICT 25-question synthetic QA dataset via the Ragas
``TestsetGenerator``, runs each question through the online
``FinancialRAGPipeline``, scores the traced responses with a Groq
LLM-as-a-Judge tier, and finally logs everything to MLflow while enforcing an
automated Model Registry quality gate (Production vs Staging).
"""

from evaluation.judge_evaluator import JudgeEvaluator, JudgeScoreSchema
from evaluation.batch_runner import BatchRunner
from evaluation.synthetic_generator import SyntheticDataGenerator
from evaluation.mlflow_tracker import (
    MlflowTracker,
    QualityGate,
    QualityGateResult,
)

__all__ = [
    "JudgeEvaluator",
    "JudgeScoreSchema",
    "BatchRunner",
    "SyntheticDataGenerator",
    "MlflowTracker",
    "QualityGate",
    "QualityGateResult",
]
