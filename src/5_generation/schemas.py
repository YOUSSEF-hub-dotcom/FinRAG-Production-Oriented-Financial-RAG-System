"""
Structured Pydantic Output Schemas for the Financial RAG Generation Engine.

Enforces strict JSON schema compliance on LLM outputs before returning
to the user.  Every field is validated with clear error messages so that
malformed generations are caught early and the fallback path is triggered.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Explicit refusal markers. A model answer containing one of these reads as a
# deliberate refusal (in-scope "not available" / out-of-scope), which is a
# first-class outcome — NOT a malformed generation.
_REFUSAL_MARKERS = (
    "not available in the provided",
    "not available in provided",
    "information is not available",
    "is not available in the",
    "cannot be answered",
    "can't be answered",
    "unable to answer",
    "cannot answer",
    "outside the scope",
    "outside the ingested",
    "no relevant information",
    "not present in the provided",
    "not found in the provided",
)


def is_refusal_text(text: str) -> bool:
    """True when *text* reads as an explicit refusal (not a malformed output)."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def classify_generation_output(raw: str) -> str:
    """Classify raw LLM text as ``"structured"``, ``"refusal"`` or ``"malformed"``.

    Distinct malformed detection (D5): the raw-JSON/default-fill path must be
    distinguishable from a legitimate refusal.

    * ``structured`` — contains a parseable JSON object (the answer contract);
    * ``refusal``    — not structured JSON but reads as an explicit refusal;
    * ``malformed``  — neither parseable JSON nor a refusal.
    """
    text = (raw or "").strip()
    if not text:
        return "malformed"

    candidate = text
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(candidate)
    except Exception:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            try:
                parsed = json.loads(brace.group(0))
            except Exception:
                parsed = None

    if parsed is not None:
        return "structured"
    return "refusal" if is_refusal_text(text) else "malformed"


class ConsolidatedFinancialAnswer(BaseModel):
    """
    The canonical output schema for every RAG generation.

    The LLM is prompted to produce JSON matching this structure exactly.
    Pydantic validation ensures no partial or hallucinated output reaches
    the end user.
    """

    internal_thought: str = Field(
        ...,
        description=(
            "Step-by-step financial reasoning. Must include unit-scaling "
            "verification (M/B/K), currency symbol checks, and cross-document "
            "consistency notes."
        ),
        min_length=1,
    )
    extracted_raw_data: str = Field(
        ...,
        description=(
            "Exact verbatim numerical and financial facts copied directly "
            "from the XML context. No interpretation or rounding allowed."
        ),
        min_length=1,
    )
    answer: str = Field(
        ...,
        description=(
            "Executive-level answer synthesized exclusively from "
            "extracted_raw_data. Must reference specific figures."
        ),
        min_length=1,
    )
    sources: List[str] = Field(
        ...,
        description=(
            "List of exact citations formatted as "
            "'ticker - fiscal_year - section - page_number'."
        ),
    )

    @field_validator("sources")
    @classmethod
    def sources_strip_whitespace(cls, v: list[str]) -> list[str]:
        """Drop empty/whitespace-only citation strings.

        An empty result is allowed through here so the model-level validator can
        decide whether it is a legitimate refusal; a non-refusal with no sources
        is still rejected below.
        """
        cleaned: list[str] = []
        for src in v:
            stripped = str(src).strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def enforce_refusal_contract(self) -> "ConsolidatedFinancialAnswer":
        """Require citations, except for an explicit refusal.

        D5: a refusal is a first-class state. When the answer reads as a refusal
        an empty ``sources`` list is valid; otherwise at least one citation is
        mandatory and ``internal_thought`` must reference a number.
        """
        is_refusal = not self.sources and is_refusal_text(self.answer)
        if not self.sources and not is_refusal:
            raise ValueError(
                "At least one non-empty source citation is required unless the "
                "answer is an explicit refusal"
            )
        if not is_refusal and not any(c.isdigit() for c in self.internal_thought):
            raise ValueError(
                "internal_thought should contain at least one numeric reference "
                "for financial reasoning verification"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "internal_thought": (
                    "Revenue for fiscal year 2025 is reported as $394,328M in "
                    "the consolidated statements. Converting M to full scale: "
                    "394,328,000,000. No conflicting figures found across "
                    "segments."
                ),
                "extracted_raw_data": (
                    "Net sales for 2025: $394,328 million. "
                    "Net income for 2025: $93,736 million."
                ),
                "answer": (
                    "Apple Inc. reported total net sales of $394.3 billion "
                    "(approximately $394,328 million) for fiscal year 2025, "
                    "with net income of $93.7 billion ($93,736 million)."
                ),
                "sources": [
                    "AAPL - 2025 - Consolidated Statements of Operations - 30",
                    "AAPL - 2025 - Revenue by Segment - 35",
                ],
            }
        }


class GuardrailVerdict(BaseModel):
    """Result of an asynchronous hallucination verification pass."""

    passed: bool = Field(
        ...,
        description="True if all numerical claims in the answer match extracted_raw_data.",
    )
    verified_claims: List[str] = Field(
        default_factory=list,
        description="List of numerical claims that were successfully verified.",
    )
    failed_claims: List[str] = Field(
        default_factory=list,
        description="List of numerical claims that could not be verified (hallucination).",
    )
    detail: str = Field(
        default="",
        description="Human-readable summary of the verification outcome.",
    )


class CacheEntry(BaseModel):
    """A single entry in the Redis semantic cache."""

    query_hash: str
    answer_json: str
    guardrail_passed: bool
    timestamp_iso: str


class RefusalAnswer(BaseModel):
    """Explicit, first-class schema for a deliberate refusal.

    Distinct from a malformed generation: ``refused`` is always ``True`` and the
    answer carries a human-readable refusal reason, so a refusal can be recorded
    and monitored separately from a schema/parse failure.
    """

    internal_thought: str = Field(
        default="Refusal: the question is outside the ingested corpus.",
        min_length=1,
        description="Reasoning that the request is out of scope / unanswerable.",
    )
    extracted_raw_data: str = Field(
        default="",
        description="Always empty for a refusal (no grounded figures).",
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="User-facing refusal message.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Empty for a refusal (no citations support the refusal).",
    )
    refused: bool = Field(default=True)
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Machine-readable refusal category (e.g. out_of_scope).",
    )
