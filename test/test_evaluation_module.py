"""
Unit tests for Module 6: Evaluation & Quality Gate Engine.

Covers (offline, no external API calls):
  1. Synthetic dataset generation -- schema validation (EXACTLY 25 questions),
     document loading, ragas testset -> CSV conversion.
  2. Batch execution loop -- question/answer/contexts/ground_truth tracing.
  3. LLM-as-a-Judge -- Pydantic scoring schemas, metric math, judge JSON
     parsing, primary -> fallback failover, async scoring.
  4. MLflow Tracking & Quality Gate -- threshold checks and automated Model
     Registry Production/Staging transitions (MlflowClient mocked).

All external services (Groq, ragas, MLflow, MLflow Client) are mocked so the
suite runs fully offline.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from evaluation.batch_runner import BatchRunner
from evaluation.judge_evaluator import (
    METRIC_ANSWER_RELEVANCE,
    METRIC_CONTEXT_PRECISION,
    METRIC_CONTEXT_RECALL,
    METRIC_FAITHFULNESS,
    AnswerRelevanceVerdict,
    ContextPrecisionVerdict,
    ContextRecallVerdict,
    EvaluationResult,
    FaithfulnessVerdict,
    JudgeEvaluator,
    JudgeScoreSchema,
    SampleScore,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
)
from evaluation.mlflow_tracker import (
    PRODUCTION_ALIAS,
    STAGING_ALIAS,
    MlflowTracker,
    QualityGate,
    QualityGateResult,
)
from evaluation.synthetic_generator import (
    SYNTHETIC_TEST_SIZE,
    SyntheticDataGenerator,
    load_documents,
)


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------------


class _FakeTestset:
    """Minimal stand-in for a ragas Testset exposing to_pandas()."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


def _make_testset_rows(count: int) -> list[dict]:
    return [
        {
            "user_input": f"What was AAPL revenue in 2025? Sample {i}",
            "reference": f"AAPL revenue was reported as $395B in 2025. Sample {i}",
            "synthesizer_name": "single_hop_specific",
        }
        for i in range(count)
    ]


class TestSyntheticDataGenerator:
    def test_test_size_strictly_limited_to_25(self):
        with pytest.raises(ValueError):
            SyntheticDataGenerator(test_size=26)
        with pytest.raises(ValueError):
            SyntheticDataGenerator(test_size=10)

    def test_default_test_size_is_25(self):
        gen = SyntheticDataGenerator()
        assert gen._test_size == SYNTHETIC_TEST_SIZE == 25

    def test_load_documents_reads_ticker_filings(self, tmp_path):
        for ticker in ("AAPL", "MSFT", "NVDA"):
            d = tmp_path / ticker / "10-K" / "0001"
            d.mkdir(parents=True)
            (d / "full-submission.txt").write_text("Apple revenue $395B. " * 20)
        docs = load_documents(
            tickers=["AAPL", "MSFT", "NVDA"],
            data_dir=tmp_path,
            max_doc_chars=5000,
        )
        assert len(docs) == 3
        assert {d.metadata["ticker"] for d in docs} == {"AAPL", "MSFT", "NVDA"}
        assert all(d.page_content for d in docs)

    def test_load_documents_raises_when_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_documents(data_dir=tmp_path)

    def test_convert_testset_produces_question_ground_truth(self, tmp_path):
        gen = SyntheticDataGenerator(output_path=tmp_path / "out.csv")
        fake = _FakeTestset(_make_testset_rows(5))
        df = gen._convert_testset(fake)
        assert list(df.columns) == ["question", "ground_truth"]
        assert len(df) == 5
        assert df["question"].iloc[0].startswith("What was AAPL")
        assert df["ground_truth"].iloc[0].startswith("AAPL revenue")

    def test_convert_testset_rejects_missing_columns(self):
        gen = SyntheticDataGenerator()
        bad = _FakeTestset([{"question": "x", "ground_truth": "y"}])
        with pytest.raises(ValueError):
            gen._convert_testset(bad)

    @patch.object(SyntheticDataGenerator, "_build_testset_generator")
    def test_generate_saves_exactly_25_rows(self, mock_builder, tmp_path):
        out_path = tmp_path / "test_dataset.csv"
        gen = SyntheticDataGenerator(output_path=out_path)

        generator_mock = MagicMock()
        generator_mock.llm = MagicMock()
        generator_mock.generate_with_langchain_docs.return_value = _FakeTestset(
            _make_testset_rows(25)
        )
        mock_builder.return_value = generator_mock

        with patch.object(
            SyntheticDataGenerator, "_build_query_distribution", return_value=[]
        ):
            df = gen.generate(documents=[MagicMock()])

        assert len(df) == 25
        assert list(df.columns) == ["question", "ground_truth"]
        saved = pd.read_csv(out_path)
        assert len(saved) == 25
        assert list(saved.columns) == ["question", "ground_truth"]

    @patch.object(SyntheticDataGenerator, "_build_testset_generator")
    def test_generate_enforces_size_limit(self, mock_builder, tmp_path):
        gen = SyntheticDataGenerator(output_path=tmp_path / "test_dataset.csv")
        mock_builder.return_value = MagicMock()
        with patch.object(
            SyntheticDataGenerator, "_build_query_distribution", return_value=[]
        ):
            with pytest.raises(ValueError):
                gen.generate(documents=[], test_size=30)


# ---------------------------------------------------------------------------
# Batch Runner (Tracing & Execution Loop)
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Minimal pipeline double with query() and _last_contexts tracking."""

    def __init__(self, answer="Apple reported $395B revenue.", contexts=None):
        self._answer = answer
        self._last_contexts = contexts or []
        self.query_calls = []

    def query(self, user_query, top_k=3, **kwargs):
        self.query_calls.append(user_query)
        parsed = MagicMock()
        parsed.answer = self._answer
        return {
            "parsed": parsed,
            "raw_output": json.dumps({"answer": self._answer}),
            "model_used": "openai/gpt-oss-120b",
            "pipeline_run_id": "run_abc",
        }


class TestBatchRunner:
    def _write_dataset(self, path: Path, count: int = 25) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "question": f"What was AAPL revenue in 2025? Sample {i}",
                "ground_truth": f"AAPL revenue was reported as $395B in 2025. Sample {i}",
            }
            for i in range(count)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_load_test_dataset_validates_columns(self, tmp_path):
        path = tmp_path / "test_dataset.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"question": ["q"], "answer": ["a"]}).to_csv(path, index=False)
        runner = BatchRunner(pipeline=_FakePipeline(), input_path=path)
        with pytest.raises(ValueError):
            runner.load_test_dataset()

    def test_load_test_dataset_missing_file(self, tmp_path):
        runner = BatchRunner(
            pipeline=_FakePipeline(), input_path=tmp_path / "nope.csv"
        )
        with pytest.raises(FileNotFoundError):
            runner.load_test_dataset()

    def test_run_batch_traces_all_25(self, tmp_path):
        dataset_path = tmp_path / "test_dataset.csv"
        self._write_dataset(dataset_path, 25)
        out_path = tmp_path / "evaluation_results.csv"
        runner = BatchRunner(
            pipeline=_FakePipeline(contexts=[{"text": "chunk-one", "metadata": {"ticker": "AAPL"}}]),
            input_path=dataset_path,
            output_path=out_path,
            concurrency=4,
        )
        df = runner.run()
        assert len(df) == 25
        for col in ("question", "ground_truth", "contexts", "answer", "model_used", "pipeline_run_id"):
            assert col in df.columns
        first = df.iloc[0]
        assert first["answer"] == "Apple reported $395B revenue."
        ctx = json.loads(first["contexts"])
        assert ctx[0]["text"] == "chunk-one"
        assert ctx[0]["metadata"]["ticker"] == "AAPL"
        assert Path(out_path).exists()

    def test_run_one_handles_pipeline_failure(self, tmp_path):
        class _FailingPipeline:
            _last_contexts = []

            def query(self, **kwargs):
                raise RuntimeError("boom")

        runner = BatchRunner(
            pipeline=_FailingPipeline(),
            input_path=tmp_path,
            output_path=tmp_path / "out.csv",
        )
        trace = runner._run_one("Q?", "A.")
        assert trace["question"] == "Q?"
        assert trace["ground_truth"] == "A."
        assert trace["answer"] == ""


# ---------------------------------------------------------------------------
# LLM-as-a-Judge Evaluator
# ---------------------------------------------------------------------------


class TestJudgeSchemasAndMath:
    def test_judge_score_schema_valid(self):
        score = JudgeScoreSchema(metric="faithfulness", score=0.9, reasoning="ok", model="qwen")
        assert score.score == 0.9

    def test_judge_score_schema_clamps_out_of_range(self):
        score = JudgeScoreSchema(metric="answer_relevance", score=1.5, reasoning="x")
        assert score.score == 1.0
        score = JudgeScoreSchema(metric="answer_relevance", score=-0.2, reasoning="x")
        assert score.score == 0.0

    def test_compute_faithfulness(self):
        v = FaithfulnessVerdict(supported_claims=["a", "b"], unsupported_claims=["c"])
        assert compute_faithfulness(v) == pytest.approx(2 / 3)

    def test_compute_faithfulness_empty_is_zero(self):
        assert compute_faithfulness(FaithfulnessVerdict()) == 0.0

    def test_compute_context_precision_rank_aware(self):
        # 3 contexts, ranks 1 and 2 relevant -> correct retrieval
        v = ContextPrecisionVerdict(relevant_context_indices=[1, 2])
        assert compute_context_precision(v, 3) == pytest.approx(
            (1 / 1 + 2 / 2) / 2
        )
        # late hit only: rank 3 relevant
        v2 = ContextPrecisionVerdict(relevant_context_indices=[3])
        assert compute_context_precision(v2, 3) == pytest.approx((1 / 3) / 1)

    def test_compute_context_precision_no_relevant(self):
        assert compute_context_precision(ContextPrecisionVerdict(), 3) == 0.0

    def test_compute_context_recall(self):
        v = ContextRecallVerdict(
            attributed_claims=["a"], unattributed_claims=["b", "c"]
        )
        assert compute_context_recall(v) == pytest.approx(1 / 3)

    def test_compute_context_recall_empty(self):
        assert compute_context_recall(ContextRecallVerdict()) == 0.0

    def test_parse_json_output_valid(self):
        parsed = JudgeEvaluator._parse_json_output(
            '{"score": 0.75, "reasoning": "good"}', AnswerRelevanceVerdict
        )
        assert isinstance(parsed, AnswerRelevanceVerdict)
        assert parsed.score == 0.75

    def test_parse_json_output_invalid_raises(self):
        with pytest.raises(ValueError):
            JudgeEvaluator._parse_json_output("not json", AnswerRelevanceVerdict)

    def test_extract_context_texts(self):
        payload = json.dumps([{"text": "t1", "metadata": {}}, {"text": "t2"}])
        texts = JudgeEvaluator._extract_context_texts(payload)
        assert texts == ["t1", "t2"]
        assert JudgeEvaluator._extract_context_texts("") == []


class TestJudgeEvaluator:
    def _make_evaluator(self, client=None):
        return JudgeEvaluator(client=client or MagicMock())

    @pytest.mark.asyncio
    async def test_score_sample_returns_valid_sample(self):
        evaluator = self._make_evaluator()
        with (
            patch.object(evaluator, "_judge_faithfulness") as m_faith,
            patch.object(evaluator, "_judge_answer_relevance") as m_rel,
            patch.object(evaluator, "_judge_context_precision") as m_prec,
            patch.object(evaluator, "_judge_context_recall") as m_rec,
        ):
            m_faith.return_value = (JudgeScoreSchema(metric=METRIC_FAITHFULNESS, score=0.9, reasoning="f", model="qwen"), "qwen")
            m_rel.return_value = (JudgeScoreSchema(metric=METRIC_ANSWER_RELEVANCE, score=0.8, reasoning="r", model="qwen"), "qwen")
            m_prec.return_value = (JudgeScoreSchema(metric=METRIC_CONTEXT_PRECISION, score=0.85, reasoning="p", model="qwen"), "qwen")
            m_rec.return_value = (JudgeScoreSchema(metric=METRIC_CONTEXT_RECALL, score=0.95, reasoning="c", model="qwen"), "qwen")

            sample = await evaluator.score_sample(
                question="q", ground_truth="gt", answer="a", contexts=["c1", "c2"]
            )
        assert isinstance(sample, SampleScore)
        assert sample.faithfulness == 0.9
        assert sample.answer_relevance == 0.8
        assert sample.context_precision == 0.85
        assert sample.context_recall == 0.95
        assert sample.judge_model == "qwen"

    @pytest.mark.asyncio
    async def test_call_judge_fails_over_to_fallback(self):
        evaluator = self._make_evaluator()
        evaluator._complete = AsyncMock(
            side_effect=[
                RuntimeError("primary failed"),
                '{"score": 0.6, "reasoning": "fallback used"}',
            ]
        )
        verdict, model = await evaluator._call_judge(
            "prompt", AnswerRelevanceVerdict
        )
        assert model == evaluator._fallback_model
        assert verdict.score == 0.6

    @pytest.mark.asyncio
    async def test_call_judge_raises_when_both_fail(self):
        evaluator = self._make_evaluator()
        evaluator._complete = AsyncMock(side_effect=RuntimeError("always"))
        with pytest.raises(RuntimeError):
            await evaluator._call_judge("prompt", AnswerRelevanceVerdict)

    @pytest.mark.asyncio
    async def test_arun_aggregates_scores(self):
        evaluator = self._make_evaluator()
        evaluator.score_sample = AsyncMock(
            return_value=SampleScore(
                question="q",
                ground_truth="gt",
                answer="a",
                faithfulness=0.9,
                answer_relevance=0.8,
                context_precision=0.75,
                context_recall=0.85,
                judge_model="qwen",
            )
        )
        rows = [{"question": "q", "ground_truth": "gt", "answer": "a", "contexts": "[]"}] * 3
        with patch("evaluation.judge_evaluator.EVALUATION_SCORES_PATH", "/tmp/none.json"):
            result = await evaluator.arun(results=rows)
        assert isinstance(result, EvaluationResult)
        assert result.faithfulness == pytest.approx(0.9)
        assert result.answer_relevance == pytest.approx(0.8)
        assert len(result.samples) == 3

    def test_load_evaluation_results_missing_file(self, tmp_path):
        evaluator = self._make_evaluator()
        with patch(
            "evaluation.judge_evaluator.EVALUATION_RESULTS_PATH",
            tmp_path / "missing.csv",
        ):
            with pytest.raises(FileNotFoundError):
                evaluator.load_evaluation_results()

    def test_load_evaluation_results_valid(self, tmp_path):
        path = tmp_path / "results.csv"
        pd.DataFrame(
            [{"question": "q", "answer": "a", "contexts": "[]", "ground_truth": "gt"}]
        ).to_csv(path, index=False)
        evaluator = self._make_evaluator()
        rows = evaluator.load_evaluation_results(path)
        assert rows and rows[0]["question"] == "q"


# ---------------------------------------------------------------------------
# MLflow Tracker & Quality Gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_gate_passes_all_thresholds(self):
        gate = QualityGate()
        result = gate.check(
            {
                "faithfulness": 0.90,
                "answer_relevance": 0.85,
                "context_precision": 0.80,
                "context_recall": 0.85,
            }
        )
        assert isinstance(result, QualityGateResult)
        assert result.passed is True
        assert all(result.checks.values())

    def test_gate_fails_on_any_threshold(self):
        gate = QualityGate()
        result = gate.check(
            {
                "faithfulness": 0.90,
                "answer_relevance": 0.70,  # below 0.80
                "context_precision": 0.80,
                "context_recall": 0.85,
            }
        )
        assert result.passed is False
        assert result.checks["answer_relevance"] is False

    def test_gate_boundary_equality_passes(self):
        gate = QualityGate(
            faithfulness=0.85, answer_relevance=0.80,
            context_precision=0.75, context_recall=0.80,
        )
        result = gate.check(
            {
                "faithfulness": 0.85,
                "answer_relevance": 0.80,
                "context_precision": 0.75,
                "context_recall": 0.80,
            }
        )
        assert result.passed is True


class TestMlflowTracker:
    def _tracker(self):
        return MlflowTracker()

    @patch("evaluation.mlflow_tracker.mlflow.end_run")
    @patch("evaluation.mlflow_tracker.mlflow.set_tag")
    @patch("evaluation.mlflow_tracker.mlflow.log_artifact")
    @patch("evaluation.mlflow_tracker.mlflow.log_metric")
    @patch("evaluation.mlflow_tracker.mlflow.log_param")
    @patch("evaluation.mlflow_tracker.mlflow.start_run")
    @patch("evaluation.mlflow_tracker.mlflow.set_experiment")
    def test_log_evaluation_run_passing(
        self, m_exp, m_start, m_param, m_metric, m_art, m_tag, m_end
    ):
        tracker = self._tracker()
        result = tracker.log_evaluation_run(
            aggregate_scores={
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.85,
            },
            params={"test_size": 25, "judge_model": "qwen/qwen3.6-27b"},
        )
        assert result.passed is True
        m_exp.assert_called_once_with("Financial_RAG_Evaluation")
        m_start.assert_called_once()
        m_tag.assert_any_call("quality_gate", "Production")
        m_metric.assert_any_call("faithfulness", 0.9)

    @patch("evaluation.mlflow_tracker.mlflow.end_run")
    @patch("evaluation.mlflow_tracker.mlflow.set_tag")
    @patch("evaluation.mlflow_tracker.mlflow.log_artifact")
    @patch("evaluation.mlflow_tracker.mlflow.log_metric")
    @patch("evaluation.mlflow_tracker.mlflow.log_param")
    @patch("evaluation.mlflow_tracker.mlflow.start_run")
    @patch("evaluation.mlflow_tracker.mlflow.set_experiment")
    def test_log_evaluation_run_failing(
        self, m_exp, m_start, m_param, m_metric, m_art, m_tag, m_end
    ):
        tracker = self._tracker()
        result = tracker.log_evaluation_run(
            aggregate_scores={
                "faithfulness": 0.4,
                "answer_relevance": 0.2,
                "context_precision": 0.1,
                "context_recall": 0.3,
            }
        )
        assert result.passed is False
        m_tag.assert_any_call("quality_gate", "Staging")

    @patch("evaluation.mlflow_tracker.MlflowTracker._register_pipeline_model")
    @patch("evaluation.mlflow_tracker.MlflowClient")
    def test_transition_to_production_on_pass(self, m_client, m_register):
        version = MagicMock()
        version.version = "3"
        m_register.return_value = version
        tracker = self._tracker()
        gate = QualityGate().check(
            {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.85,
            }
        )
        with patch("evaluation.mlflow_tracker.mlflow.active_run") as m_active:
            run = MagicMock()
            run.info.run_id = "run_id_1"
            m_active.return_value = run
            target = tracker.transition_model_version(gate)

        assert target == PRODUCTION_ALIAS == "Production"
        m_client.return_value.set_registered_model_alias.assert_called_once_with(
            "financial_rag_pipeline", "Production", "3"
        )
        m_client.return_value.transition_model_version_stage.assert_called_once()

    @patch("evaluation.mlflow_tracker.MlflowTracker._register_pipeline_model")
    @patch("evaluation.mlflow_tracker.MlflowClient")
    def test_transition_to_staging_on_fail(self, m_client, m_register):
        version = MagicMock()
        version.version = "4"
        m_register.return_value = version
        tracker = self._tracker()
        gate = QualityGate().check(
            {
                "faithfulness": 0.4,
                "answer_relevance": 0.2,
                "context_precision": 0.1,
                "context_recall": 0.3,
            }
        )
        with patch("evaluation.mlflow_tracker.mlflow.active_run") as m_active:
            run = MagicMock()
            run.info.run_id = "run_id_2"
            m_active.return_value = run
            target = tracker.transition_model_version(gate)

        assert target == STAGING_ALIAS == "Staging"
        m_client.return_value.set_registered_model_alias.assert_called_once_with(
            "financial_rag_pipeline", "Staging", "4"
        )

    def test_pipeline_state_model_predict(self):
        from evaluation.mlflow_tracker import _PipelineStateModel

        model = _PipelineStateModel()
        out = model.predict(None, None)
        assert out["pipeline"] == "financial_rag_pipeline"


# ---------------------------------------------------------------------------
# End-to-end wiring sanity (pure computation, no APIs)
# ---------------------------------------------------------------------------


class TestModule6Wiring:
    def test_full_offline_flow(self, tmp_path):
        """
        Exercise the evaluation pipeline end-to-end with fakes only:
        generate 25 rows -> trace through fake pipeline -> judge (mocked)
        -> quality gate verdict.
        """
        # 1. Synthetic dataset
        gen = SyntheticDataGenerator(output_path=tmp_path / "test_dataset.csv")
        df = gen._convert_testset(_FakeTestset(_make_testset_rows(25)))
        assert len(df) == 25

        # 2. Batch trace via fake pipeline
        out_path = tmp_path / "evaluation_results.csv"
        runner = BatchRunner(
            pipeline=_FakePipeline(),
            input_path=tmp_path / "test_dataset.csv",
            output_path=out_path,
        )
        # write the dataset for the runner
        df.to_csv(tmp_path / "test_dataset.csv", index=False)
        traced = runner.run()
        assert len(traced) == 25

        # 3. Judge scoring (all judges mocked at schema level)
        judge = JudgeEvaluator(client=MagicMock())
        with (
            patch.object(judge, "_judge_faithfulness") as f,
            patch.object(judge, "_judge_answer_relevance") as r,
            patch.object(judge, "_judge_context_precision") as p,
            patch.object(judge, "_judge_context_recall") as c,
            patch("evaluation.judge_evaluator.EVALUATION_SCORES_PATH", str(tmp_path / "scores.json")),
        ):
            f.return_value = (JudgeScoreSchema(metric=METRIC_FAITHFULNESS, score=0.92, reasoning="x", model="qwen"), "qwen")
            r.return_value = (JudgeScoreSchema(metric=METRIC_ANSWER_RELEVANCE, score=0.88, reasoning="x", model="qwen"), "qwen")
            p.return_value = (JudgeScoreSchema(metric=METRIC_CONTEXT_PRECISION, score=0.82, reasoning="x", model="qwen"), "qwen")
            c.return_value = (JudgeScoreSchema(metric=METRIC_CONTEXT_RECALL, score=0.86, reasoning="x", model="qwen"), "qwen")
            result = asyncio.run(judge.arun(results=traced.to_dict(orient="records")))

        # 4. Quality gate -> Production
        gate = QualityGate()
        verdict = gate.check(
            {
                "faithfulness": result.faithfulness,
                "answer_relevance": result.answer_relevance,
                "context_precision": result.context_precision,
                "context_recall": result.context_recall,
            }
        )
        assert verdict.passed is True
        assert result.faithfulness == pytest.approx(0.92)
