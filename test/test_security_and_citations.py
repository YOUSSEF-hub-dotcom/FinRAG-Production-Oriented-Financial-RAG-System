"""
Tests for the two confirmed security/citation fixes:

1. Upload authorization:
   - anonymous uploads are rejected with 401 (previously reachable)
   - regular (non-admin) uploads are rejected with 403
   - rejected requests leave NO side effects: no file under DATA_DIR,
     no ingestion task, no chunks/vectors

2. Citation / SSE ``sources`` correctness:
   - sources are derived ONLY from the current request's grounded context
     (request_contexts / retrieved_chunks), never from a shared/global slot
   - unrelated chunks from a prior query cannot leak into a later response
   - empty context emits NO fabricated source
   - missing metadata degrades to empty values (never fabricated)
   - LLM citation strings only REORDER real context chunks, they are never
     emitted as chunk ids themselves
"""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Stub llama_parse so mock.patch("llama_parse.LlamaParse") resolves even
# when the real package is not installed.
if "llama_parse" not in sys.modules:
    import types

    fake = types.ModuleType("llama_parse")
    fake.LlamaParse = MagicMock()
    sys.modules["llama_parse"] = fake

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_root = _PROJECT_ROOT / "src"
for _p in (
    str(_PROJECT_ROOT),
    str(_src_root),
    str(_src_root / "1_ingestion"),
    str(_src_root / "5_generation"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.api.auth.dependencies import UserPrincipal, get_current_user as _auth_get_current_user
from app.api.main import (
    _build_stream_sources,
    _build_sources,
    _context_chunk_citation_score,
    _rank_contexts_by_citations,
    _source_record_from_context,
    app,
)
from config.settings import DATA_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_pipeline_and_deps():
    mock_pipe = MagicMock()
    mock_pipe.close = MagicMock()
    with patch("app.api.main.FinancialRAGPipeline", return_value=mock_pipe):
        yield {"pipeline": mock_pipe, "pipeline_cls": mock_pipe}


@pytest.fixture
def client(patch_pipeline_and_deps):
    with TestClient(app) as c:
        yield c


def _override_role(role: str):
    principal = UserPrincipal(user_id=f"t-{role}-1", email=f"{role}@test.local", role=role)
    app.dependency_overrides[_auth_get_current_user] = lambda: principal


@pytest.fixture
def admin_client(client):
    _override_role("admin")
    try:
        yield client
    finally:
        app.dependency_overrides.pop(_auth_get_current_user, None)


@pytest.fixture
def user_client(client):
    _override_role("user")
    try:
        yield client
    finally:
        app.dependency_overrides.pop(_auth_get_current_user, None)


def _make_test_pdf(text: str = "Test financial data Revenue $100M") -> bytes:
    from io import BytesIO

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(w=0, text=text)
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Citation / source building unit tests
# ---------------------------------------------------------------------------

def _ctx(chunk_id: str, score: float, ticker="AAPL", fy="2025", section="Revenue",
         source_file="aapl.txt", text="Revenue grew substantially to support growth."):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "payload": {
            "ticker": ticker,
            "fiscal_year": fy,
            "section": section,
            "source_file": source_file,
            "page_number": 3,
            "text": text,
            "raw_text": text,
        },
    }


class TestContextChunkCitationScore:
    def test_ticker_year_and_content_overlap_boost(self):
        ctx = _ctx("NVDA_txt_1", 0.5, ticker="NVDA", fy="2026", section="Data Center")
        assert _context_chunk_citation_score(ctx, "NVIDIA - 2026 - Data Center") > 0

    def test_irrelevant_citation_scores_zero(self):
        ctx = _ctx("NVDA_txt_1", 0.5, ticker="NVDA", fy="2026", section="Data Center")
        assert _context_chunk_citation_score(ctx, "Tesla - 2024 - Automotive") == 0

    def test_empty_citation_is_zero(self):
        assert _context_chunk_citation_score(_ctx("NVDA_txt_1", 0.5), "") == 0


class TestBuildStreamSources:
    def test_sources_only_from_request_context(self):
        contexts = [
            _ctx("NVDA_txt_supplement", 0.9, ticker="NVDA", fy="2026",
                 section="Data Center", source_file="NVDA_AI_Infrastructure_Expansion_FY2026.pdf"),
            _ctx("NVDA_txt_other", 0.7, ticker="NVDA", fy="2026", section="Data Center"),
        ]
        sources = _build_stream_sources(
            ["NVIDIA - 2026 - Data Center", "AAPL - 2020 - Revenue"], contexts
        )
        ids = [s["chunk_id"] for s in sources]
        # Only chunks that appear in THIS request's context can be emitted.
        assert set(ids) <= {"NVDA_txt_supplement", "NVDA_txt_other"}

    def test_prior_query_chunk_cannot_leak(self):
        # A chunk that was retrieved for an earlier query but is NOT part of
        # this request's context must never be surfaced, even when the LLM
        # citation contains its ticker.
        prior_retrieval = [_ctx("OLD_abc123_0000", 0.99, ticker="NVDA", fy="2026",
                                section="Data Center")]
        current_contexts = [_ctx("NEW_def456_0000", 0.5, ticker="NVDA", fy="2026",
                                 section="Data Center")]
        for leak in prior_retrieval:
            try:
                current_contexts.remove(leak)
            except ValueError:
                pass
        sources = _build_stream_sources(
            ["NVIDIA - 2026 - Data Center"], current_contexts
        )
        ids = [s["chunk_id"] for s in sources]
        assert "OLD_abc123_0000" not in ids
        assert set(ids) <= {"NEW_def456_0000"}

    def test_empty_context_returns_no_sources(self):
        assert _build_stream_sources(["NVIDIA - 2026 - Revenue"], []) == []
        assert _build_stream_sources([], []) == []

    def test_missing_metadata_never_fabricated(self):
        ctx = {
            "chunk_id": "REAL_0000",
            "score": 0.4,
            "payload": {"text": "Some grounded content here."},
        }
        sources = _build_stream_sources([], [ctx])
        assert len(sources) == 1
        src = sources[0]
        assert src["chunk_id"] == "REAL_0000"
        assert src["ticker"] == ""
        assert src["fiscal_year"] == ""
        assert src["section"] == ""
        assert src["source_file"] == ""

    def test_citation_cited_chunk_ranked_first(self):
        cited = _ctx("NVDA_txt_cited", 0.4, ticker="NVDA", fy="2026", section="Data Center")
        higher_score_uncited = _ctx("NVDA_txt_top", 0.95, ticker="NVDA", fy="2026")
        sources = _build_stream_sources(
            ["NVIDIA - 2026 - Data Center"], [higher_score_uncited, cited]
        )
        assert sources[0]["chunk_id"] == "NVDA_txt_cited"
        assert sources[1]["chunk_id"] == "NVDA_txt_top"

    def test_deduplicates_and_respects_limit(self):
        contexts = [_ctx(f"c{i}", (10 - i) / 10) for i in range(20)]
        sources = _build_stream_sources([], contexts)
        ids = [s["chunk_id"] for s in sources]
        assert len(ids) == len(set(ids)) == 6  # default limit

    def test_citation_string_is_never_a_chunk_id(self):
        contexts = [_ctx("REAL_1", 0.6, ticker="NVDA", fy="2026")]
        sources = _build_stream_sources(["NVIDIA - 2026 - Data Center"], contexts)
        assert all(s["chunk_id"].startswith("REAL_") for s in sources)


class TestBuildSourcesSync:
    def test_maps_citation_to_real_chunk(self):
        retrieved = [
            {
                "chunk_id": "NVDA_txt_4d3d41429f6a_0000",
                "score": 0.0261,
                "payload": {
                    "ticker": "NVDA",
                    "fiscal_year": "2026",
                    "section": "Data Center",
                    "source_file": "NVDA_AI_Infrastructure_Expansion_FY2026.pdf",
                    "text": "Data Center revenue was $108,400M for fiscal 2026.",
                },
            }
        ]
        parsed = MagicMock()
        parsed.sources = ["NVIDIA - Data Center - 2026"]
        result = {"parsed": parsed, "retrieved_chunks": retrieved}
        sources = _build_sources(result)
        assert len(sources) == 1
        assert sources[0].chunk_id == "NVDA_txt_4d3d41429f6a_0000"
        # The sync response model carries ticker/year/section + snippet.
        assert sources[0].ticker == "NVDA"
        assert sources[0].fiscal_year == "2026"
        assert sources[0].section == "Data Center"
        assert "108,400M" in sources[0].text_snippet
        # Score is clamped to the SourceCitation allowed range.
        assert 0.0 <= sources[0].score <= 1.0

    def test_no_retrieved_context_yields_empty_sources(self):
        parsed = MagicMock()
        parsed.sources = ["NVIDIA - 2026 - Data Center"]
        assert _build_sources({"parsed": parsed}) == []

    def test_citation_string_not_used_as_chunk_id(self):
        parsed = MagicMock()
        parsed.sources = ["NVIDIA - 2026 - Data Center"]
        retrieved = [
            {
                "chunk_id": "REAL_chunk_1",
                "score": 0.8,
                "payload": {"ticker": "NVDA", "fiscal_year": "2026", "text": "some content"},
            }
        ]
        sources = _build_sources({"parsed": parsed, "retrieved_chunks": retrieved})
        assert sources[0].chunk_id == "REAL_chunk_1"
        assert "NVIDIA - 2026 - Data Center" not in [s.chunk_id for s in sources]


class TestSourceRecordFromContext:
    def test_missing_source_file_results_in_empty_string(self):
        ctx = {
            "chunk_id": "chunk_x",
            "score": 0.5,
            "payload": {"ticker": "NVDA", "raw_text": ""},
        }
        rec = _source_record_from_context(ctx)
        assert rec["source_file"] == ""
        assert rec["text"] == ""


# ---------------------------------------------------------------------------
# Upload authorization + side-effect tests
# ---------------------------------------------------------------------------

class TestUploadAuthorizationSideEffects:
    def test_anonymous_rejected_and_no_side_effects(self, client):
        import app.api.main as main_module
        before = len(main_module._INGESTION_TASKS)
        resp = client.post(
            "/api/v1/documents/upload?ticker=MKTS&fiscal_year=2023",
            files={"file": ("filing.pdf", _make_test_pdf(), "application/pdf")},
        )
        assert resp.status_code == 401
        assert len(main_module._INGESTION_TASKS) == before
        assert not (DATA_DIR / "MKTS").exists()

    def test_regular_user_rejected_and_no_side_effects(self, user_client):
        import app.api.main as main_module
        before = len(main_module._INGESTION_TASKS)
        resp = user_client.post(
            "/api/v1/documents/upload?ticker=MKTS&fiscal_year=2023",
            files={"file": ("filing.pdf", _make_test_pdf(), "application/pdf")},
        )
        assert resp.status_code == 403
        assert "Insufficient privileges" in resp.json()["detail"]
        assert len(main_module._INGESTION_TASKS) == before
        assert not (DATA_DIR / "MKTS").exists()

    def test_admin_upload_accepted(self, admin_client):
        resp = admin_client.post(
            "/api/v1/documents/upload?ticker=MKTS&fiscal_year=2023",
            files={"file": ("filing.pdf", _make_test_pdf(), "application/pdf")},
        )
        assert resp.status_code == 200
        saved = DATA_DIR / "MKTS" / "10-K" / "2023" / "filing.pdf"
        assert saved.exists()
        saved.unlink()
        saved.parent.rmdir()
        saved.parent.parent.rmdir()
        saved.parent.parent.parent.rmdir()