"""
LLM-as-a-Judge Evaluator for the Module 6 Evaluation pipeline.

Reads ``artifacts/evaluation_results.csv`` offline and computes the 4 core
Ragas metrics over all 25 samples by prompting a Groq judge tier:

    1. Faithfulness       -- fraction of answer claims supported by contexts
    2. Answer Relevance   -- how well the answer addresses the question
    3. Context Precision  -- rank-aware precision of retrieved contexts
    4. Context Recall     -- fraction of ground-truth claims attributable to contexts

The judge tier runs on the Groq Async API with tenacity exponential-backoff
retries (TPM-safe), primary ``qwen-2.5-coder-32b`` falling back to
``mixtral-8x7b-32768``, temperature 0.0 and max_tokens 1024. Every judge call
is bound to an explicit Pydantic schema (``JudgeScoreSchema`` and metric
verdicts) so responses are structurally validated before scoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

import groq
import tenacity
from pydantic import BaseModel, Field, field_validator, model_validator

from config.logging_config import get_logger
from config.settings import (
    EVALUATION_RESULTS_PATH,
    EVALUATION_SCORES_PATH,
    GROQ_API_KEY,
    JUDGE_FALLBACK_MODEL,
    JUDGE_MAX_CONTEXT_CHARS,
    JUDGE_MAX_RETRIES,
    JUDGE_MAX_TOKENS,
    JUDGE_PRIMARY_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_TIMEOUT_SECONDS,
)

logger = get_logger("evaluation.judge")

# Supported core Ragas metric names.
METRIC_FAITHFULNESS = "faithfulness"
METRIC_ANSWER_RELEVANCE = "answer_relevance"
METRIC_CONTEXT_PRECISION = "context_precision"
METRIC_CONTEXT_RECALL = "context_recall"

SUPPORTED_METRICS: tuple[str, ...] = (
    METRIC_FAITHFULNESS,
    METRIC_ANSWER_RELEVANCE,
    METRIC_CONTEXT_PRECISION,
    METRIC_CONTEXT_RECALL,
)

# ---------------------------------------------------------------------------
# Pydantic structured-output schemas
# ---------------------------------------------------------------------------


class JudgeScoreSchema(BaseModel):
    """
    Canonical normalized score for a single metric on a single sample.

    ``score`` is always in [0.0, 1.0] where higher is better, ``reasoning``
    holds the judge's rationale and ``model`` records which judge model
    produced the verdict (for auditability).
    """

    metric: Literal[
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    model: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> float:
        """
        Clamp out-of-range scores into [0.0, 1.0] before constraint checks.

        ``mode="before"`` is required because Pydantic v2 enforces the
        ``ge``/``le`` field constraints during the validation step, before any
        after-validators have a chance to normalise the value.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        return max(0.0, min(1.0, numeric))


class FaithfulnessVerdict(BaseModel):
    """LLM raw verdict for the faithfulness metric (answer claims vs contexts)."""

    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @field_validator("supported_claims", "unsupported_claims", mode="before")
    @classmethod
    def _coerce_claim_items(cls, value: Any) -> Any:
        """Flatten dict-wrapped claim items (some judge models emit objects)."""
        if not isinstance(value, list):
            return value
        return [_claim_text(item) for item in value]


class AnswerRelevanceVerdict(BaseModel):
    """LLM raw verdict for the answer relevance metric."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> Any:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


class ContextPrecisionVerdict(BaseModel):
    """LLM raw verdict for context precision (per-rank relevance flags)."""

    relevant_context_indices: list[int] = Field(default_factory=list)

    @field_validator("relevant_context_indices", mode="before")
    @classmethod
    def _coerce_indices(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out


class ContextRecallVerdict(BaseModel):
    """LLM raw verdict for context recall (ground-truth claims attribution)."""

    attributed_claims: list[str] = Field(default_factory=list)
    unattributed_claims: list[str] = Field(default_factory=list)

    @field_validator("attributed_claims", "unattributed_claims", mode="before")
    @classmethod
    def _coerce_claim_items(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [_claim_text(item) for item in value]


def _claim_text(item: Any) -> str:
    """Extract a plain string from a claim item (str or dict wrapper)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("claim", "text", "statement", "claim_text", "value"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return json.dumps(item, ensure_ascii=False)
    return str(item)


class SampleScore(BaseModel):
    """Per-sample evaluation result across the 4 core Ragas metrics."""

    question: str
    ground_truth: str
    answer: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    judge_model: str = ""


class EvaluationResult(BaseModel):
    """Aggregated evaluation result across all samples."""

    samples: list[SampleScore] = Field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    judge_model: str = ""

    @model_validator(mode="after")
    def _compute_aggregates(self) -> "EvaluationResult":
        """Recompute the mean aggregates from the samples."""
        for metric in SUPPORTED_METRICS:
            values = [getattr(s, metric) for s in self.samples if s is not None]
            setattr(self, metric, _mean(values) if values else 0.0)
        return self


# ---------------------------------------------------------------------------
# Metric computation helpers (faithful to Ragas methodology)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Safe arithmetic mean that returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def compute_faithfulness(verdict: FaithfulnessVerdict) -> float:
    """
    Faithfulness = supported_claims / (supported_claims + unsupported_claims).

    Ragas defines faithfulness as the fraction of claims in the answer that
    can be verified against the retrieved contexts. Empty claim lists yield
    a conservative 0.0.
    """
    supported = len(verdict.supported_claims)
    unsupported = len(verdict.unsupported_claims)
    total = supported + unsupported
    return supported / total if total else 0.0


def compute_context_precision(
    verdict: ContextPrecisionVerdict,
    context_count: int,
) -> float:
    """
    Rank-aware context precision.

    Ragas formula: sum over every relevant rank ``k`` of
    ``precision@k / num_relevant`` where ``precision@k`` is the fraction of
    relevant contexts among the top-k retrieved contexts. Verdict indices are
    1-based ranks so the score rewards retrieval that ranks relevant chunks
    higher (precision@k uses only the top-k, hence punishes late hits).
    """
    if context_count <= 0:
        return 0.0

    relevant_ranks = sorted({i for i in verdict.relevant_context_indices if 1 <= i <= context_count})
    if not relevant_ranks:
        return 0.0

    precision_at_k_sum = 0.0
    for k in relevant_ranks:
        # How many relevant chunks are in the top-k retrieved contexts?
        relevant_in_top_k = sum(1 for r in relevant_ranks if r <= k)
        precision_at_k_sum += relevant_in_top_k / k
    return precision_at_k_sum / len(relevant_ranks)


def compute_context_recall(verdict: ContextRecallVerdict) -> float:
    """
    Context recall = attributed_claims / (attributed + unattributed claims).

    Ragas defines context recall as the fraction of ground-truth claims that
    are attributable to the retrieved contexts. Empty ground-truth claim sets
    yield a conservative 0.0.
    """
    attributed = len(verdict.attributed_claims)
    unattributed = len(verdict.unattributed_claims)
    total = attributed + unattributed
    return attributed / total if total else 0.0


# ---------------------------------------------------------------------------
# Prompt templates (structured JSON outputs)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a strict, deterministic LLM-as-a-Judge for a financial RAG system that
answers questions about SEC 10-K filings (AAPL, MSFT, NVDA). You evaluate
retrieval-and-generation quality using the Ragas metric methodology.

Rules:
- Score strictly on the evidence provided; never assume facts outside the
  given question, answer, ground truth and contexts.
- Respond with ONLY valid JSON. No markdown, no code fences, no preamble.
- Scores must be floats between 0.0 and 1.0 inclusive.
"""

_FAITHFULNESS_PROMPT = """\
TASK: Faithfulness -- measure how well the ANSWER is supported by the
CONTEXTS (no hallucination).

Steps:
1. Decompose the answer into a list of atomic factual claims.
2. For each claim, check whether it is entailed by the CONTEXTS.
3. Return JSON: {{"supported_claims": [..], "unsupported_claims": [..]}}

QUESTION:
{question}

ANSWER:
{answer}

CONTEXTS:
{contexts}
"""

_ANSWER_RELEVANCE_PROMPT = """\
TASK: Answer Relevance -- measure how well the ANSWER addresses the QUESTION.

Score the answer 0.0-1.0 (1.0 = fully relevant, directly answers the question;
0.0 = completely off-topic). Return JSON:
{{"score": <float 0.0-1.0>, "reasoning": "<short rationale>"}}

QUESTION:
{question}

ANSWER:
{answer}
"""

_CONTEXT_PRECISION_PROMPT = """\
TASK: Context Precision -- for each ranked CONTEXT chunk, decide whether it
contains information needed to answer the QUESTION according to the GROUND
TRUTH (the reference answer).

Steps:
1. Read each chunk below; chunks are numbered 1..N in retrieval order.
2. Mark a chunk as relevant ONLY if it helps answer the QUESTION per the
   GROUND TRUTH.
3. Return JSON: {{"relevant_context_indices": [1, 3, ..]}} (1-based ranks).

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

CONTEXTS (ranked):
{numbered_contexts}
"""

_CONTEXT_RECALL_PROMPT = """\
TASK: Context Recall -- measure how much of the GROUND TRUTH can be attributed
to the CONTEXTS.

Steps:
1. Decompose the GROUND TRUTH into a list of atomic claims.
2. For each claim, check whether it is entailed by the CONTEXTS.
3. Return JSON: {{"attributed_claims": [..], "unattributed_claims": [..]}}

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

CONTEXTS:
{contexts}
"""


# ---------------------------------------------------------------------------
# Judge Evaluator
# ---------------------------------------------------------------------------


class JudgeEvaluator:
    """
    Async LLM-as-a-Judge that computes the 4 core Ragas metrics per sample.

    The judge tier calls the Groq async chat completions API with tenacity
    exponential-backoff retries (handles TPM rate limits gracefully). Primary
    model ``qwen/qwen3.6-27b`` is preferred; on repeated failure the call is
    retried once with the fallback ``openai/gpt-oss-120b``.
    """

    def __init__(
        self,
        primary_model: str = JUDGE_PRIMARY_MODEL,
        fallback_model: str = JUDGE_FALLBACK_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        max_retries: int = JUDGE_MAX_RETRIES,
        timeout_seconds: float = JUDGE_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """
        Args:
            primary_model: Groq judge model (strict structured output).
            fallback_model: Groq fallback judge model (MoE, 32k context).
            temperature: Judge temperature (0.0 = deterministic).
            max_tokens: Max judge output tokens.
            max_retries: Tenacity retry attempts (exponential backoff).
            timeout_seconds: Per-request timeout.
            api_key: Groq API key (default: settings.GROQ_API_KEY).
            client: Optional pre-built groq.AsyncGroq client (tests inject a mock).
        """
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client = client or groq.AsyncGroq(
            api_key=api_key or GROQ_API_KEY,
            timeout=timeout_seconds,
        )

    # -- Low-level async calls with retry -----------------------------------

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Build the chat messages list from a single user prompt."""
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    async def _complete(self, model: str, messages: list[dict[str, str]]) -> str:
        """
        Single chat completion call wrapped in tenacity exponential retry.

        Retries on Groq API errors, rate limits, timeouts and connection
        errors, backing off with an exponential wait (TPM-safe). Returns the
        raw assistant message content.
        """
        attempt = tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_retries),
            wait=tenacity.wait_exponential(multiplier=1.0, min=2, max=60),
            retry=tenacity.retry_if_exception_type(
                (
                    groq.RateLimitError,
                    groq.APITimeoutError,
                    groq.APIConnectionError,
                    groq.APIStatusError,
                    groq.APIError,
                )
            ),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        @attempt
        async def _call() -> str:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "response_format": {"type": "json_object"},
            }
            if model.startswith("qwen/"):
                kwargs["reasoning_effort"] = "none"
            response = await self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise groq.APIError("Judge returned empty content")
            return content

        return await _call()

    async def _call_judge(
        self,
        prompt: str,
        schema_type: type[BaseModel],
    ) -> tuple[Any, str]:
        """
        Execute a judge call with primary -> fallback model failover.

        Returns a ``(parsed_schema, model_used)`` tuple. Falls back to the
        fallback model once if the primary exhausts retries or its JSON output
        fails schema validation; raises ``RuntimeError`` if both fail.
        """
        messages = self._build_messages(prompt)
        for model in (self._primary_model, self._fallback_model):
            try:
                content = await self._complete(model, messages)
                parsed = self._parse_json_output(content, schema_type)
                logger.debug("Judge %s (%s) parsed successfully", schema_type.__name__, model)
                return parsed, model
            except Exception as exc:
                logger.warning(
                    "Judge call failed with %s (metric schema %s): %s",
                    model,
                    schema_type.__name__,
                    exc,
                )
        raise RuntimeError(
            f"Judge evaluation failed for schema {schema_type.__name__} "
            f"with both models ({self._primary_model}, {self._fallback_model})."
        )

    @staticmethod
    def _parse_json_output(content: str, schema_type: type[BaseModel]) -> BaseModel:
        """Strictly parse and validate judge JSON output via Pydantic."""
        text = content.strip()
        # Strip any accidental markdown fences / thinking tags.
        if text.startswith("```"):
            text = text.strip("`")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Judge returned invalid JSON: {exc}") from exc
        return schema_type.model_validate(data)

    # -- Per-metric judging ---------------------------------------------------

    def _format_contexts(self, contexts: list[str], numbered: bool = False) -> str:
        """Render the contexts list into a readable prompt block."""
        if not contexts:
            return "(no contexts provided)"
        truncated = [c[:JUDGE_MAX_CONTEXT_CHARS] for c in contexts]
        if numbered:
            return "\n\n".join(
                f"[{i + 1}] {ctx}" for i, ctx in enumerate(truncated)
            )
        return "\n\n".join(f"- {ctx}" for ctx in truncated)

    async def _judge_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> tuple[JudgeScoreSchema, str]:
        """Score faithfulness via LLM claim verification against contexts."""
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            contexts=self._format_contexts(contexts),
        )
        verdict, model = await self._call_judge(prompt, FaithfulnessVerdict)
        score = compute_faithfulness(verdict)
        reasoning = (
            f"{len(verdict.supported_claims)} supported / "
            f"{len(verdict.supported_claims) + len(verdict.unsupported_claims)} claims"
        )
        return (
            JudgeScoreSchema(
                metric=METRIC_FAITHFULNESS,
                score=score,
                reasoning=reasoning,
                model=model,
            ),
            model,
        )

    async def _judge_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> tuple[JudgeScoreSchema, str]:
        """Score answer relevance directly via the judge."""
        prompt = _ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
        verdict, model = await self._call_judge(prompt, AnswerRelevanceVerdict)
        return (
            JudgeScoreSchema(
                metric=METRIC_ANSWER_RELEVANCE,
                score=verdict.score,
                reasoning=verdict.reasoning,
                model=model,
            ),
            model,
        )

    async def _judge_context_precision(
        self,
        question: str,
        ground_truth: str,
        contexts: list[str],
    ) -> tuple[JudgeScoreSchema, str]:
        """Score rank-aware context precision via per-rank relevance flags."""
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            numbered_contexts=self._format_contexts(contexts, numbered=True),
        )
        verdict, model = await self._call_judge(prompt, ContextPrecisionVerdict)
        score = compute_context_precision(verdict, len(contexts))
        reasoning = (
            f"relevant ranks: {sorted(verdict.relevant_context_indices)} "
            f"of {len(contexts)} contexts"
        )
        return (
            JudgeScoreSchema(
                metric=METRIC_CONTEXT_PRECISION,
                score=score,
                reasoning=reasoning,
                model=model,
            ),
            model,
        )

    async def _judge_context_recall(
        self,
        question: str,
        ground_truth: str,
        contexts: list[str],
    ) -> tuple[JudgeScoreSchema, str]:
        """Score context recall via ground-truth claim attribution."""
        prompt = _CONTEXT_RECALL_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            contexts=self._format_contexts(contexts),
        )
        verdict, model = await self._call_judge(prompt, ContextRecallVerdict)
        score = compute_context_recall(verdict)
        reasoning = (
            f"{len(verdict.attributed_claims)} attributed / "
            f"{len(verdict.attributed_claims) + len(verdict.unattributed_claims)} claims"
        )
        return (
            JudgeScoreSchema(
                metric=METRIC_CONTEXT_RECALL,
                score=score,
                reasoning=reasoning,
                model=model,
            ),
            model,
        )

    # -- Public API -----------------------------------------------------------

    async def score_sample(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        contexts: list[str],
    ) -> SampleScore:
        """
        Compute all 4 core Ragas metrics for a single sample (async).

        Args:
            question: The original user question.
            ground_truth: Reference answer from the synthetic dataset.
            answer: The pipeline-generated answer.
            contexts: Retrieved context chunks (raw text).

        Returns:
            A validated ``SampleScore`` with the 4 metric scores.
        """
        faith, _ = await self._judge_faithfulness(question, answer, contexts)
        rel, model = await self._judge_answer_relevance(question, answer)
        prec, _ = await self._judge_context_precision(question, ground_truth, contexts)
        rec, _ = await self._judge_context_recall(question, ground_truth, contexts)

        return SampleScore(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            faithfulness=faith.score,
            answer_relevance=rel.score,
            context_precision=prec.score,
            context_recall=rec.score,
            judge_model=model,
        )

    def load_evaluation_results(self, path: Optional[Path] = None) -> list[dict]:
        """Read traced pipeline results from the evaluation CSV."""
        p = path or Path(EVALUATION_RESULTS_PATH)
        import pandas as pd

        if not Path(p).exists():
            raise FileNotFoundError(
                f"Evaluation results not found at {p}. Run BatchRunner first."
            )
        df = pd.read_csv(p)
        for col in ("question", "answer", "contexts", "ground_truth"):
            if col not in df.columns:
                raise ValueError(f"Evaluation results missing column '{col}'")
        return df.to_dict(orient="records")

    @staticmethod
    def _extract_context_texts(contexts_field: str) -> list[str]:
        """Parse the JSON-serialized contexts column into raw text chunks."""
        try:
            data = json.loads(contexts_field or "[]")
        except (json.JSONDecodeError, TypeError):
            return []
        texts = []
        for item in data:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return [t for t in texts if t]

    async def arun(
        self,
        results: Optional[list[dict]] = None,
        concurrency: int = 4,
    ) -> EvaluationResult:
        """
        Evaluate every traced sample and aggregate the 4 core metrics.

        Args:
            results: List of trace dicts (default: load from evaluation_results.csv).
            concurrency: Max parallel judge requests per sample batch.

        Returns:
            Aggregated ``EvaluationResult`` with per-sample and mean scores.
        """
        rows = results if results is not None else self.load_evaluation_results()
        if not rows:
            raise ValueError("No evaluation rows to score.")

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(row: dict) -> SampleScore:
            async with semaphore:
                contexts = self._extract_context_texts(row.get("contexts", ""))
                return await self.score_sample(
                    question=str(row.get("question", "")),
                    ground_truth=str(row.get("ground_truth", "")),
                    answer=str(row.get("answer", "")),
                    contexts=contexts,
                )

        samples = await asyncio.gather(*[_bounded(r) for r in rows])
        result = EvaluationResult(samples=list(samples))
        self.save_scores(result)
        return result

    def run(self, results: Optional[list[dict]] = None) -> EvaluationResult:
        """Synchronous wrapper around the async evaluation loop."""
        return asyncio.run(self.arun(results=results))

    def save_scores(self, result: EvaluationResult) -> None:
        """Persist the evaluation result to artifacts/evaluation_scores.json."""
        path = Path(EVALUATION_SCORES_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Saved evaluation scores to %s", path)
