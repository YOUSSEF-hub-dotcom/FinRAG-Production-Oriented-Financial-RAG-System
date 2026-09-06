"""
Asynchronous Numerical Hallucination Guardrail & Semantic Cache Handler.

Runs a lightweight background verification pass that cross-checks every
numerical claim in the generated answer against the extracted_raw_data.

  - PASS → writes query, context hash, and verified JSON to Redis cache.
  - FAIL → bypasses cache, aborts output, returns safe fallback response.
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import mlflow

from config.logging_config import get_logger

# Numbered module dirs are not valid packages — put them on sys.path (same
# convention as app/api/main.py) so `semantic_cache` imports cleanly.
_src_root = Path(__file__).resolve().parent.parent
for _subdir in ("1_ingestion", "2_caching"):
    _p = str(_src_root / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantic_cache import SemanticCache

logger = get_logger("generation.guardrail")

# ---------------------------------------------------------------------------
# Hashing utility (shared with generator.py)
# ---------------------------------------------------------------------------


def _hash_query(query: str) -> str:
    """Deterministic short hash for cache keys."""
    import hashlib
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Safe fallback response returned when guardrail fails
# ---------------------------------------------------------------------------
SAFE_FALLBACK = (
    "Direct arithmetic verification failed for the numbers present "
    "in the provided reports."
)

# ---------------------------------------------------------------------------
# Numerical extraction helpers
# ---------------------------------------------------------------------------

# Matches: $1,234.56  $1234  (1,234)  -1234  12.5%  1234M  1234B  1234K
_NUM_PATTERN = re.compile(
    r"(?:[\$]?\s*[\(]?\s*-?\s*)"          # optional currency, parens, minus
    r"[\d,]+"                              # integer part with commas
    r"(?:\.\d+)?"                          # optional decimal
    r"(?:\s*[%])"                          # optional percent
    r"|"
    r"(?:[\$]?\s*[\(]?\s*-?\s*)"          # currency prefix
    r"[\d,]+"                              # digits
    r"(?:\.\d+)?"                          # optional decimal
    r"(?:\s*[MBKmbk])?"                   # optional scale suffix
    r"|"
    r"\([\d,]+(?:\.\d+)?\)"               # parenthesized negative
)

# Simplified: extract all number-like tokens
_NUM_EXTRACT = re.compile(
    r"\(?\s*-?\s*\d[\d,]*\.?\d*\s*[%MBKmbk]?\s*\)?"
)


def _extract_numbers(text: str) -> list[str]:
    """Extract all number-like tokens from text, normalized to stripped form."""
    matches = _NUM_EXTRACT.findall(text)
    normalized: list[str] = []
    for m in matches:
        stripped = m.strip()
        # Collapse internal whitespace
        stripped = re.sub(r"\s+", "", stripped)
        if stripped:
            normalized.append(stripped)
    return normalized


def _normalize_number(num_str: str) -> str:
    """
    Normalize a number string for comparison.

    Removes currency symbols, spaces, and standardizes formatting.
    """
    s = num_str.strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("M", "").replace("B", "").replace("K", "")
    s = s.replace("m", "").replace("b", "").replace("k", "")
    s = s.replace("%", "")
    # Keep parens for negatives
    return s


# ---------------------------------------------------------------------------
# Async Verification Loop
# ---------------------------------------------------------------------------

async def verify_numerical_claims(
    answer: str,
    extracted_raw_data: str,
) -> tuple[bool, list[str], list[str]]:
    """
    Asynchronously verify that every number in the answer exists in the raw data.

    This is a lightweight regex-based check (no LLM call) that runs in a
    background thread to avoid blocking the main generation path.

    Args:
        answer: The generated answer text.
        extracted_raw_data: The verbatim extracted facts from context.

    Returns:
        (passed, verified_claims, failed_claims)
    """
    # Run CPU-bound regex work in a thread to stay async
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _verify_sync, answer, extracted_raw_data
    )


def _verify_sync(
    answer: str,
    extracted_raw_data: str,
) -> tuple[bool, list[str], list[str]]:
    """Synchronous verification implementation."""
    answer_numbers = _extract_numbers(answer)
    raw_numbers = _extract_numbers(extracted_raw_data)

    if not answer_numbers:
        # No numbers to verify — pass by default
        return True, [], []

    raw_normalized = {_normalize_number(n) for n in raw_numbers}

    verified: list[str] = []
    failed: list[str] = []

    for num in answer_numbers:
        norm = _normalize_number(num)
        if norm in raw_normalized or norm.lstrip("0") in {n.lstrip("0") for n in raw_normalized}:
            verified.append(num)
        else:
            failed.append(num)

    passed = len(failed) == 0
    return passed, verified, failed


# ---------------------------------------------------------------------------
# Guardrail Orchestrator
# ---------------------------------------------------------------------------

class AsyncGuardrail:
    """
    Orchestrates the full verification + cache flow:

    1. Verify numerical claims in the answer against extracted_raw_data
    2. If PASS → write to Redis semantic cache, log MLflow, return answer
    3. If FAIL → bypass cache, log MLflow, return safe fallback response
    """

    def __init__(self, cache: SemanticCache | None = None):
        self._cache = cache or SemanticCache()

    async def check(
        self,
        query: str,
        answer: str,
        extracted_raw_data: str,
        parsed_output: Any = None,
        skip_cache: bool = False,
        ticker: str | None = None,
        fiscal_year: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full guardrail verification pipeline.

        Args:
            query: Original user query.
            answer: Generated answer text.
            extracted_raw_data: Verbatim extracted facts from context.
            parsed_output: Optional parsed ConsolidatedFinancialAnswer for caching.
            skip_cache: If True, skip writing to Redis cache even on pass.
            ticker: Optional ticker for cache key namespacing.
            fiscal_year: Optional fiscal year for cache key namespacing.

        Returns:
            Dict with: passed, verified_claims, failed_claims, final_output,
            cache_written, detail.
        """
        t0 = time.time()

        # Step 1: Verify numerical claims
        passed, verified, failed = await verify_numerical_claims(
            answer, extracted_raw_data
        )

        elapsed_ms = (time.time() - t0) * 1000
        detail = (
            f"Verified {len(verified)} claims, {len(failed)} failed. "
            f"Elapsed: {elapsed_ms:.1f}ms"
        )

        # Log to MLflow
        try:
            mlflow.log_metric("guardrail_passed", int(passed))
            mlflow.log_metric("guardrail_verified_count", len(verified))
            mlflow.log_metric("guardrail_failed_count", len(failed))
            mlflow.log_metric("guardrail_elapsed_ms", round(elapsed_ms, 2))
        except Exception:
            pass

        cache_written = False

        # Auto-detect invalid / fallback answers that should never be cached
        is_not_available = "not available" in answer.lower()
        auto_skip = is_not_available
        effective_skip = skip_cache or auto_skip

        if passed:
            if not effective_skip:
                # Step 2a: PASS — write to cache
                if parsed_output is not None:
                    try:
                        answer_json = (
                            parsed_output.model_dump_json()
                            if hasattr(parsed_output, "model_dump_json")
                            else json.dumps(str(parsed_output))
                        )
                    except Exception:
                        answer_json = json.dumps({"answer": answer})
                else:
                    answer_json = json.dumps({"answer": answer})

                cache_written = await self._cache.aput(
                    query,
                    answer_json,
                    guardrail_passed=True,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                )
                logger.info("Guardrail PASSED — %s (cache written)", detail)
            else:
                logger.info("Guardrail PASSED — %s (cache skipped: skip_cache=%s)", detail, effective_skip)
            final_output = answer
        else:
            # Step 2b: FAIL — abort output, return safe fallback
            final_output = SAFE_FALLBACK
            logger.warning("Guardrail FAILED — %s", detail)

        return {
            "passed": passed,
            "verified_claims": verified,
            "failed_claims": failed,
            "final_output": final_output,
            "cache_written": cache_written,
            "detail": detail,
        }

    def close(self) -> None:
        """Close underlying cache connections."""
        self._cache.close()
