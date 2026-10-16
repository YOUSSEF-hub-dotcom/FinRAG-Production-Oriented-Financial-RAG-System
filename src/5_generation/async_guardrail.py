"""
Asynchronous Numerical Hallucination Guardrail & Semantic Cache Handler.

Runs a lightweight background verification pass that cross-checks every
numerical claim in the generated answer against the extracted_raw_data.

  - PASS  → writes query, context hash, and verified JSON to Redis cache.
  - FAIL  → bypasses cache, returns safe fallback response.
  - ERROR → bypasses cache, returns an error verdict (never a PASS).
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
from config.settings import GUARDRAIL_TIMEOUT_SECONDS

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

# Machine-readable verdict states shared by the async and sync entry points.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"

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
    # _NUM_EXTRACT absorbs a trailing sentence/clause '.' or ',' into the
    # token (e.g. the year in "fiscal year 2025." -> "2025."). Strip it so
    # the value compares equal to the same number in the raw data. Only
    # trailing punctuation is removed; internal decimal points are kept.
    s = s.rstrip(".,")
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
    2. PASS  -> write to Redis semantic cache, return the answer
    3. FAIL  -> bypass cache, return the safe fallback response
    4. ERROR -> bypass cache, return an error verdict (never a PASS)

    Two entry points share the same result contract:
      * ``check``      -- async, used by the streaming path.
      * ``check_sync`` -- sync, used by the blocking ``pipeline.query`` path.

    Both return ``passed`` plus a machine-readable ``status`` of
    ``passed`` / ``failed`` / ``error`` so callers can propagate a real
    verdict instead of assuming one.
    """

    def __init__(
        self,
        cache: SemanticCache | None = None,
        timeout_seconds: float | None = None,
    ):
        self._cache = cache or SemanticCache()
        self._timeout_seconds = (
            GUARDRAIL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )

    # -- shared helpers --------------------------------------------------

    @staticmethod
    def _answer_json(answer: str, parsed_output: Any) -> str:
        """Serialise the verified answer for cache storage."""
        if parsed_output is not None:
            try:
                return (
                    parsed_output.model_dump_json()
                    if hasattr(parsed_output, "model_dump_json")
                    else json.dumps(str(parsed_output))
                )
            except Exception:
                return json.dumps({"answer": answer})
        return json.dumps({"answer": answer})

    @staticmethod
    def _log_metrics(
        passed: bool, verified: list[str], failed: list[str], elapsed_ms: float
    ) -> None:
        try:
            mlflow.log_metric("guardrail_passed", int(passed))
            mlflow.log_metric("guardrail_verified_count", len(verified))
            mlflow.log_metric("guardrail_failed_count", len(failed))
            mlflow.log_metric("guardrail_elapsed_ms", round(elapsed_ms, 2))
        except Exception:
            pass

    @staticmethod
    def _should_cache(answer: str, skip_cache: bool) -> bool:
        """A PASS is cacheable only when explicitly permitted and not a fallback."""
        return not (skip_cache or "not available" in answer.lower())

    @staticmethod
    def _build_result(
        *,
        passed: bool,
        verified: list[str],
        failed: list[str],
        final_output: str,
        cache_written: bool,
        detail: str,
        status: str,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "passed": passed,
            "verified_claims": verified,
            "failed_claims": failed,
            "final_output": final_output,
            "cache_written": cache_written,
            "detail": detail,
            "status": status,
            "error": error,
        }

    def _finalize(
        self,
        *,
        passed: bool,
        verified: list[str],
        failed: list[str],
        status: str,
        error: str | None,
        elapsed_ms: float,
        cache_written: bool,
        answer: str,
    ) -> dict[str, Any]:
        """Log metrics, pick the final output, and build the verdict dict."""
        detail = (
            f"Verified {len(verified)} claims, {len(failed)} failed. "
            f"Status: {status}. Elapsed: {elapsed_ms:.1f}ms"
        )
        if error:
            detail += f" Error: {error}"
        self._log_metrics(passed, verified, failed, elapsed_ms)
        if status == STATUS_PASSED:
            final_output = answer
            logger.info(
                "Guardrail PASSED -- %s%s",
                detail,
                "" if cache_written else " (cache skipped)",
            )
        elif status == STATUS_FAILED:
            final_output = SAFE_FALLBACK
            logger.warning("Guardrail FAILED -- %s", detail)
        else:
            final_output = answer
            logger.warning("Guardrail ERROR (not cached) -- %s", detail)
        return self._build_result(
            passed=passed,
            verified=verified,
            failed=failed,
            final_output=final_output,
            cache_written=cache_written,
            detail=detail,
            status=status,
            error=error,
        )

    # -- async entry point (streaming) -----------------------------------

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
        Run the full guardrail verification pipeline (async / streaming path).

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
            cache_written, detail, status, error. A timeout or evaluator
            exception yields ``status='error'`` / ``passed=False`` and writes
            nothing to the cache.
        """
        t0 = time.time()
        status = STATUS_PASSED
        error: str | None = None
        passed = False
        verified: list[str] = []
        failed: list[str] = []

        try:
            coro = verify_numerical_claims(answer, extracted_raw_data)
            if self._timeout_seconds and self._timeout_seconds > 0:
                passed, verified, failed = await asyncio.wait_for(
                    coro, timeout=self._timeout_seconds
                )
            else:
                passed, verified, failed = await coro
            status = STATUS_PASSED if passed else STATUS_FAILED
        except asyncio.TimeoutError:
            passed, status = False, STATUS_ERROR
            error = f"timeout after {self._timeout_seconds}s"
        except Exception as exc:
            passed, status = False, STATUS_ERROR
            error = str(exc)

        elapsed_ms = (time.time() - t0) * 1000
        cache_written = False
        if status == STATUS_PASSED and self._should_cache(answer, skip_cache):
            try:
                cache_written = await self._cache.aput(
                    query,
                    self._answer_json(answer, parsed_output),
                    guardrail_passed=True,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                )
            except Exception as exc:
                logger.warning("Guardrail cache write failed (non-blocking): %s", exc)

        return self._finalize(
            passed=passed,
            verified=verified,
            failed=failed,
            status=status,
            error=error,
            elapsed_ms=elapsed_ms,
            cache_written=cache_written,
            answer=answer,
        )

    # -- sync entry point (pipeline.query) -------------------------------

    def check_sync(
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
        Synchronous guardrail verification for the blocking pipeline path.

        The verification itself is CPU-bound regex work, so no event loop is
        involved: ``pipeline.query`` calls this directly and receives a real
        verdict (no more fire-and-forget discard). Same result contract as
        :meth:`check`.
        """
        t0 = time.time()
        status = STATUS_PASSED
        error: str | None = None
        passed = False
        verified: list[str] = []
        failed: list[str] = []

        try:
            passed, verified, failed = _verify_sync(answer, extracted_raw_data)
            status = STATUS_PASSED if passed else STATUS_FAILED
        except Exception as exc:
            passed, status = False, STATUS_ERROR
            error = str(exc)

        elapsed_ms = (time.time() - t0) * 1000
        cache_written = False
        if status == STATUS_PASSED and self._should_cache(answer, skip_cache):
            try:
                cache_written = self._cache.put(
                    query,
                    self._answer_json(answer, parsed_output),
                    guardrail_passed=True,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                )
            except Exception as exc:
                logger.warning("Guardrail cache write failed (non-blocking): %s", exc)

        return self._finalize(
            passed=passed,
            verified=verified,
            failed=failed,
            status=status,
            error=error,
            elapsed_ms=elapsed_ms,
            cache_written=cache_written,
            answer=answer,
        )

    def close(self) -> None:
        """Close underlying cache connections."""
        self._cache.close()
