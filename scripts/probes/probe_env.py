import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from config.settings import (
    GROQ_PRIMARY_MODEL,
    GROQ_FALLBACK_MODEL,
    JUDGE_PRIMARY_MODEL,
    JUDGE_FALLBACK_MODEL,
    JUDGE_MAX_TOKENS,
    JUDGE_MAX_CONTEXT_CHARS,
)

print("env GROQ_FALLBACK_MODEL =", os.getenv("GROQ_FALLBACK_MODEL"))
print("settings GROQ_FALLBACK_MODEL =", GROQ_FALLBACK_MODEL)
print("settings GROQ_PRIMARY_MODEL =", GROQ_PRIMARY_MODEL)
print("settings JUDGE_PRIMARY_MODEL =", JUDGE_PRIMARY_MODEL)
print("settings JUDGE_FALLBACK_MODEL =", JUDGE_FALLBACK_MODEL)
print("settings JUDGE_MAX_TOKENS =", JUDGE_MAX_TOKENS)
print("settings JUDGE_MAX_CONTEXT_CHARS =", JUDGE_MAX_CONTEXT_CHARS)

from src.pipeline import FinancialRAGPipeline

p = FinancialRAGPipeline(enable_cache=False, enable_guardrail=False)
print("pipeline fallback =", p._generator._fallback_model)
print("pipeline primary =", p._generator._primary_model)
