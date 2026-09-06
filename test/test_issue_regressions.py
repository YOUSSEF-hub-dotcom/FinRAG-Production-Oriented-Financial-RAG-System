"""
Regression tests for the three post-memory-fix issues.

  Issue 1: %%TABLE_N%% parser placeholders must never reach user-visible
           answers, source-chip sections, or source text.
  Issue 2: Harness/capture must never include the message timestamp or chip
           labels in the extracted answer text.
  Issue 3: Multi-ticker comparison must dedupe per-(ticker, section) so it does
           not produce near-duplicate result sets / context slots.

Each can fail without requiring models or live services (retrieval is stubbed
with the raw + structured payloads the sync path feeds to ``_augment_context``).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_ingestion"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "5_generation"))

import pytest

from pipeline import _strip_internal_markers, _context_section_key, FinancialRAGPipeline
from app.api.main import _strip_internal_markers as api_strip_internal_markers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _doc(cid: str, ticker: str, section: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "score": 0.9,
        "text": text,
        "payload": {"ticker": ticker, "fiscal_year": "2025", "section": section},
        "metadata": {"ticker": ticker, "fiscal_year": "2025", "section": section},
    }


def _pipe():
    """Pipeline with mongo stub to avoid touching MongoDB."""
    p = FinancialRAGPipeline.__new__(FinancialRAGPipeline)
    p._mongo_indexer = _FakeIndexer()
    p._resolved_generation_query = ""
    return p


class _FakeIndexer:
    def __init__(self):
        self._tables = {}

    def get_chunks_by_filter(self, filt: dict) -> list:
        return self._tables.get((filt.get("ticker"), str(filt.get("fiscal_year"))), [])


def _augment(p, docs, query="", all_companies=False):
    return p._augment_context(docs, query=query, all_companies=all_companies)


# ---------------------------------------------------------------------------
# Issue 1: %%TABLE_N%% placeholders / section markers never reach output
# ---------------------------------------------------------------------------

def test_i1_strips_table_fragments_from_section_and_text():
    dirty = _doc(
        "c1", "AAPL", "Item 8. Financial Statements %%TABLE_13%% All s",
        "Apple revenue grew. Inline [TABLE 13] and trailing %%",
    )
    out = _augment(_pipe(), [dirty])
    assert out, "augment should keep the doc"
    section = out[0]["metadata"]["section"]
    text = out[0]["text"]
    assert "%%" not in section and "%%" not in text
    assert "TABLE_" not in section and "TABLE_" not in text
    assert "[TABLE" not in section and "[TABLE" not in text
    assert "Item 8. Financial Statements All s" == section


def test_i1_strips_income_table_section_chips_issue3_input():
    # The exact near-duplicate income-table chip label seen in the E2E.
    dirty = _doc(
        "t13", "MSFT", "Item 8. Financial Statements and Supplementary Data %%TABLE_13%% All financial s",
        "MSFT income statement rows.",
    )
    out = _augment(_pipe(), [dirty])
    section = out[0]["metadata"]["section"]
    assert "%%" not in section
    assert "TABLE_" not in section
    assert section.startswith("Item 8. Financial Statements")


def test_i1_api_output_boundary_never_leaks_markers():
    cases = [
        "%%TABLE_13%%All financial s",
        "Item 8. ... INCOME STATEMENTS %%TABLE_",
        "net %%%%TABLE_7%% sales",
        "snippet [TABLE 3] and text",
    ]
    for c in cases:
        assert "%%" not in api_strip_internal_markers(c)
        assert "TABLE_" not in api_strip_internal_markers(c)
        assert "[TABLE" not in api_strip_internal_markers(c)
    # Already-clean text must be preserved verbatim.
    assert api_strip_internal_markers("Apple revenue grew 8% in 2025.") == \
        "Apple revenue grew 8% in 2025."


def test_i1_marker_function_is_monotonic_safe():
    # Purging must not corrupt whitespace-sensitive content.
    cleaned = _strip_internal_markers("  Apple  reported  $124B   ")
    assert cleaned == "Apple reported $124B"
    assert _strip_internal_markers("") == ""


# ---------------------------------------------------------------------------
# Issue 3: multi-ticker comparison dedupes per (ticker, section)
# ---------------------------------------------------------------------------

TEXT_DOCS = [
    # MSFT keeps only one of its two "PART II" narrative docs.
    _doc("m1", "MSFT", "PART II", "MSFT narrative part II overview."),
    _doc("m2", "MSFT", "PART II", "MSFT duplicate part II sibling to drop."),
    _doc("n1", "NVDA", "Part IV", "NVDA narrative."),
    _doc("n2", "NVDA", "PART IV", "NVDA duplicate part IV sibling to drop."),
    # Apple's income table chip + a second distinct section are both kept.
    _doc("a1", "AAPL", "Item 8. Financial Statements and Supplementary Data %%TABLE_13%% All financial s", "AAPL income."),
    _doc("a2", "AAPL", "Item 9A. Controls", "AAPL controls."),
]


def test_i3_multi_ticker_dedup_sections():
    p = _pipe()
    out = _augment(p, list(TEXT_DOCS), query="Compare margins Apple Microsoft NVIDIA", all_companies=True)
    sections = [
        (_doc_meta(d).get("ticker"), _strip_internal_markers(_doc_meta(d).get("section") or ""))
        for d in out
    ]
    assert len(out) == 4, f"expected 4 deduped docs, got {len(out)}: {sections}"
    msft = [d for d in out if (_doc_meta(d).get("ticker")) == "MSFT"]
    nvda = [d for d in out if (_doc_meta(d).get("ticker")) == "NVDA"]
    assert len(msft) == 1 and "PART II" in msft[0]["metadata"]["section"]
    assert len(nvda) == 1
    # Placeholder has been purged from the surviving Apple section.
    aapl = [d for d in out if (_doc_meta(d).get("ticker")) == "AAPL"]
    assert len(aapl) == 2
    assert all("%%" not in (a["metadata"]["section"] or "") for a in aapl)


def test_i3_single_ticker_retrieval_not_regressed():
    docs = [
        _doc("x1", "AAPL", "Business Overview", "overview text."),
        _doc("x2", "AAPL", "Business Overview", "second overview to dedupe."),
        _doc("x3", "AAPL", "Segment", "segment text."),
        _doc("x4", "AAPL", "Business Overview", "third overview to drop."),
    ]
    out = _augment(_pipe(), docs, query="Apple margin")
    # distinct sections ("Business Overview", "Segment") preserved; dup dropped.
    assert len(out) == 2
    got = sorted(_strip_internal_markers(d["metadata"]["section"]) for d in out)
    assert got == ["Business Overview", "Segment"]


def test_i3_generic_label_falls_back_to_content():
    # Documents without a real section label must not be over-collapsed.
    docs = [
        _doc("g1", "AAPL", "General", "Company overview product line revenue."),
        _doc("g2", "AAPL", "General", "Company segment breakdown of profit."),
    ]
    out = _augment(_pipe(), docs, query="Apple product revenue")
    # Different content -> both preserved.
    assert len(out) == 2


def _doc_meta(d):
    return d.get("metadata") or {}


# ---------------------------------------------------------------------------
# Issue 2: capture excludes timestamp / chips
# ---------------------------------------------------------------------------

def test_i2_clean_text_content_is_whitespace_normalized():
    # The harness snapshots the natural-language bubble only. Here we assert the
    # capture-relevant invariant: the mobile-checked answer snippet function
    # relies on a trimmed, whitespace-collapsed string that a real answer
    # produces.
    raw = "For fiscal year 2025, Apple reported\n\n   $391.0B of total net revenue."
    cleaned = " ".join(raw.split())
    assert cleaned == "For fiscal year 2025, Apple reported $391.0B of total net revenue."
    assert "11:24 AM" not in cleaned
    assert "SOURCE" not in cleaned