"""Stage 1b: regenerate answers over the first N queries -> evaluation_results.csv.

Usage: Financial_env/bin/python run_stage_1b.py [N]
Runs from ~/Financial_RAG (project root = cwd).
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from src.pipeline import FinancialRAGPipeline
from evaluation.batch_runner import BatchRunner

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 800

pipeline = FinancialRAGPipeline(
    enable_pre_retrieval=True,
    enable_cache=False,
    enable_guardrail=False,
    enable_hybrid_retrieval=True,
    enable_post_retrieval=True,
    top_k=3,
    max_tokens=MAX_TOKENS,
)

df = pd.read_csv("artifacts/test_dataset.csv")
df_n = df.head(N).reset_index(drop=True)
tmp_in = Path("/tmp/opencode/test_dataset_topN.csv")
tmp_in.parent.mkdir(parents=True, exist_ok=True)
df_n.to_csv(tmp_in, index=False)

runner = BatchRunner(pipeline=pipeline, input_path=tmp_in, concurrency=CONCURRENCY, top_k=3)
start = time.time()
result_df = runner.run()
elapsed = time.time() - start

print("\n=== MODEL_USED ===")
print(result_df["model_used"].value_counts().to_dict())
refusals = result_df[
    result_df["answer"].str.contains("not available", case=False, na=False)
]
print("refusal answers:", len(refusals))
print(f"samples: {len(result_df)}")
print(f"elapsed: {elapsed:.1f}s")
for _, row in result_df.iterrows():
    print("\nQ:", row["question"][:70])
    print("A:", str(row["answer"])[:200])
