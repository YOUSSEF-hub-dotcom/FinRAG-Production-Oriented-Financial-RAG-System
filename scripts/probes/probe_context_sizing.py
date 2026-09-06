"""Verify every ground-truth figure lands in the final bounded context (no API)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from src.pipeline import FinancialRAGPipeline

MONEY_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize(s: str) -> str:
    return re.sub(r"[, ]", "", s)


def extract_numbers(text: str):
    out = set()
    for m in MONEY_RE.finditer(text):
        tok = m.group(0)
        norm = normalize(tok)
        if not norm or len(norm) < 2:
            continue
        try:
            val = float(norm)
        except ValueError:
            continue
        if "." in norm and 0.5 <= val <= 100000:
            out.add(norm)
        elif val >= 10000 and "." not in norm:
            out.add(norm)
    return sorted(out)


def find_offset(haystack: str, needle: str):
    i = haystack.find(needle)
    if i >= 0:
        return i
    return haystack.replace(",", "").find(needle)


class _StubGenerator:
    def generate(self, query, retrieved_docs, stream=False):
        return {
            "raw_output": '{"answer": "stub"}',
            "parsed": None,
            "model_used": "stub",
            "fallback_triggered": False,
            "ttft_ms": 0,
        }


pipeline = FinancialRAGPipeline(
    enable_cache=False,
    enable_guardrail=False,
    enable_hybrid_retrieval=True,
    enable_post_retrieval=True,
    top_k=3,
)
pipeline._generator = _StubGenerator()

df = pd.read_csv("artifacts/test_dataset.csv").head(12)

ok = 0
for i, row in df.iterrows():
    q = str(row["question"])
    gt = str(row["ground_truth"])
    targets = extract_numbers(gt)
    try:
        result = pipeline.query(user_query=q, top_k=3)
    except Exception as exc:
        print(f"[{i:02d}] ERROR {exc}")
        continue
    ctxs = list(getattr(pipeline, "_last_contexts", []) or [])
    blob = "\n".join(c.get("text", "") for c in ctxs)
    total_chars = sum(len(c.get("text", "")) for c in ctxs)
    missing = [t for t in targets if find_offset(blob, t) < 0]
    status = "OK " if not missing else "MISS"
    if not missing:
        ok += 1
    print(f"[{i:02d}] {status} targets={targets} missing={missing} total_chars={total_chars}")
    for rank, c in enumerate(ctxs, 1):
        text = c.get("text", "")
        cid = c.get("chunk_id", "") or ""
        print(
            f"    r{rank}[{len(text)}ch] {cid[:40]} "
            f"type={c.get('metadata', {}).get('chunk_type', '?')}"
        )

print(f"\nCOVERAGE: {ok}/{len(df)} queries fully covered")
