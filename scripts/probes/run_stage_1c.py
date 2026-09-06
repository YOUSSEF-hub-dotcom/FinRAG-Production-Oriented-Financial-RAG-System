"""Stage 1c: offline judge over artifacts/evaluation_results.csv -> evaluation_scores.json."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from evaluation.judge_evaluator import JudgeEvaluator

start = time.time()
res = asyncio.run(JudgeEvaluator().arun(concurrency=1))
print(res.model_dump_json(indent=2))
print(f"\nelapsed: {time.time() - start:.1f}s")
