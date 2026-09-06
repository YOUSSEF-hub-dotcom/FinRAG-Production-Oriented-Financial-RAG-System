"""
Unit tests -- fiscal-year resolution from the question text.

Targets ``FinancialRAGPipeline._query_fiscal_year`` (the parser used by
`_augment_context` to decide WHICH filing's financial-statement tables are
injected and by the hybrid path to scope the weighted pre-filter).

Regression for: "Compare the operating margins and total net revenue between
Apple, Microsoft, and NVIDIA for FY2025." must resolve to "2025" so the FY2025
income statements are injected (with Operating income) instead of the context
falling back to the top balanced chunk's "UNKNOWN" year and refusing.

Requirements:
  - "FY2025", "FY 2025", "fiscal year 2025", "Fiscal Year 2025",
    "fiscal 2025", "FY-2025" -> "2025"
  - Bare calendar years ("2025", "in 2025") are deliberately NOT parsed
    (preserves existing semantics -- the parser is fiscal-year-specific).
  - A bare percentage like "growth of 15%" must not match.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pipeline import FinancialRAGPipeline


class TestQueryFiscalYearResolution:
    """Exact parser-level contract for _query_fiscal_year."""

    _parser = staticmethod(FinancialRAGPipeline._query_fiscal_year)

    @staticmethod
    def resolve(query: str):
        return FinancialRAGPipeline._query_fiscal_year(query)

    def test_fy2025_compact(self):
        q = ("Compare the operating margins and total net revenue between "
             "Apple, Microsoft, and NVIDIA for FY2025.")
        assert self.resolve(q) == "2025"

    def test_fy_space_2025(self):
        assert self.resolve("Compare Apple and Microsoft for FY 2025") == "2025"

    def test_fiscal_year_2025(self):
        assert self.resolve("Compare Apple and Microsoft for fiscal year 2025") == "2025"

    def test_fiscal_year_caps_2025(self):
        assert self.resolve("Compare Apple and Microsoft for Fiscal Year 2025") == "2025"

    def test_fiscal_2025(self):
        assert self.resolve("Compare Apple and Microsoft for fiscal 2025") == "2025"

    def test_fy_hyphen_2025(self):
        assert self.resolve("Compare Apple and Microsoft for FY-2025") == "2025"

    def test_latest_mention_wins(self):
        # "latest mention wins"
        q = "between fiscal 2024 and fiscal 2026, what changed?"
        assert self.resolve(q) == "2026"

    def test_fiscal_2026_parses(self):
        assert self.resolve("What was Apple revenue in fiscal 2026?") == "2026"

    def test_bare_year_2025_is_not_parsed(self):
        # Preserve existing semantics: bare calendar year without a
        # FY/fiscal prefix is NOT a fiscal-year signal.
        assert self.resolve("What was Apple revenue in 2025?") is None

    def test_percentage_growth_is_not_a_year(self):
        assert self.resolve("Compare revenue growth of 15%") is None

    def test_bare_years_without_prefix_not_parsed(self):
        assert self.resolve("Compare 2023 and 2024 performance without a fiscal prefix") is None
