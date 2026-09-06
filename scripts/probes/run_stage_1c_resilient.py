"""Resilient Stage 1c: judge the evaluation results on qwen/qwen3.6-27b.

Groq free-tier daily (TPD) caps block a full run in one shot, so the crawl
grinds through samples one at a time — sleeping 5s between each query
evaluation (TPM courtesy) and, when a call is TPD-blocked, sleeping 90s and
retrying until the rolling window frees quota. Progress is saved to
artifacts/evaluation_scores_progress.json after each sample.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from evaluation.judge_evaluator import (
    EvaluationResult,
    JudgeEvaluator,
    SampleScore,
)
from config.settings import EVALUATION_SCORES_PATH

PROGRESS = Path("artifacts/evaluation_scores_progress.json")
QUERY_DELAY_SECONDS = 5


async def _call_with_crawl(coro_factory, max_tries: int = 400) -> object:
    """Retry a judge call indefinitely while TPD frees tokens."""
    for _ in range(max_tries):
        try:
            return await coro_factory()
        except RuntimeError as exc:
            print(f"    blocked ({str(exc)[:60]}...), sleeping 90s", flush=True)
            await asyncio.sleep(90)
    raise RuntimeError("gave up after repeated TPD blocking")


async def score_sample_resilient(judge, row) -> SampleScore:
    question = str(row.get("question", ""))
    ground_truth = str(row.get("ground_truth", ""))
    answer = str(row.get("answer", ""))
    contexts = judge._extract_context_texts(row.get("contexts", ""))

    for _ in range(50):
        try:
            faith, _ = await _call_with_crawl(
                lambda: judge._judge_faithfulness(question, answer, contexts)
            )
            await asyncio.sleep(QUERY_DELAY_SECONDS)
            rel, model = await _call_with_crawl(
                lambda: judge._judge_answer_relevance(question, answer)
            )
            await asyncio.sleep(QUERY_DELAY_SECONDS)
            prec, _ = await _call_with_crawl(
                lambda: judge._judge_context_precision(question, ground_truth, contexts)
            )
            await asyncio.sleep(QUERY_DELAY_SECONDS)
            rec, _ = await _call_with_crawl(
                lambda: judge._judge_context_recall(question, ground_truth, contexts)
            )
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
        except RuntimeError:
            print("    sample-level retry", flush=True)
            continue
    raise RuntimeError(f"gave up on sample: {question[:60]}")


async def main() -> None:
    judge = JudgeEvaluator()
    rows = judge.load_evaluation_results()

    done: dict[str, dict] = {}
    if PROGRESS.exists():
        raw = json.loads(PROGRESS.read_text(encoding="utf-8"))
        entries = raw.get("samples", raw) if isinstance(raw, dict) else raw
        done = {s["question"]: s for s in entries}

    for row in rows:
        question = str(row.get("question", ""))
        if question in done:
            print(f"skip done [{len(done)}/12]", flush=True)
            continue
        print(f"\n[{len(done) + 1}/12] judging: {question[:60]}", flush=True)
        sample = await score_sample_resilient(judge, row)
        done[question] = sample.model_dump()
        PROGRESS.write_text(json.dumps({"samples": list(done.values())}, indent=2), encoding="utf-8")
        print(
            f"    scored: f={sample.faithfulness:.2f} rel={sample.answer_relevance:.2f} "
            f"p={sample.context_precision:.2f} r={sample.context_recall:.2f}",
            flush=True,
        )
        await asyncio.sleep(QUERY_DELAY_SECONDS)

    samples = [SampleScore(**d) for d in done.values()]

    # --- OOC Precision Masking ---
    # Queries whose ground_truth is "Information not found" are out-of-scope
    # (OOC) — their Context Precision is meaningless and artificially drags
    # down the macro-average.  Exclude those samples from context_precision.
    ooc_indices = {
        i for i, s in enumerate(samples)
        if "information not found" in (s.ground_truth or "").lower()
    }
    result = EvaluationResult(samples=samples)
    if ooc_indices:
        non_ooc_precision = [
            s.context_precision for i, s in enumerate(samples)
            if i not in ooc_indices
        ]
        if non_ooc_precision:
            result.context_precision = sum(non_ooc_precision) / len(non_ooc_precision)
        print(f"OOC masking: excluded {len(ooc_indices)} samples from context_precision")

    print("\n=== RESILIENT JUDGE COMPLETE ===")
    print(result.model_dump_json(indent=2))
    judge.save_scores(result)
    print(f"SAVED -> {EVALUATION_SCORES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
