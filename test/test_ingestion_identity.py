"""
Focused offline tests for D6-A: ingestion identity + metadata.

Scope (no services, no live DB, no Docker):
  - SEC accession extraction (path / header / ambiguity / malformed).
  - Amendment-aware document-type extraction (10-K/A, 10-Q/A, ...).
  - ``source_id`` propagation into ``extract_metadata`` output.
  - Upload identity resolution + safe, non-colliding save-path computation.
  - ``_ingest_document`` forwards identity/doc_type into metadata.

These tests are deterministic and must not touch MongoDB / Qdrant / Redis.
"""

import sys
from pathlib import Path

import pytest

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

# Stub llama_parse (same rationale as test_api_upload_parser.py) so importing
# app.api.main never requires the optional package.
if "llama_parse" not in sys.modules:
    import types

    fake = types.ModuleType("llama_parse")
    fake.LlamaParse = object
    sys.modules["llama_parse"] = fake

from metadata_extractor import (  # noqa: E402
    extract_accession,
    extract_doc_type,
    extract_metadata,
)


ACC_2024 = "0000320193-24-000123"
ACC_2025 = "0000320193-25-000079"


# ===========================================================================
# Accession extraction
# ===========================================================================

class TestExtractAccession:

    def test_from_path_directory(self):
        path = f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt"
        assert extract_accession(path) == ACC_2024

    def test_from_header_content(self):
        content = f"<SEC-DOCUMENT>{ACC_2024}.txt : 20241101\nACCESSION NUMBER:\t\t{ACC_2024}\n"
        assert extract_accession(None, content) == ACC_2024

    def test_path_and_content_agree(self):
        path = f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt"
        content = f"ACCESSION NUMBER:\t{ACC_2024}"
        assert extract_accession(path, content) == ACC_2024

    def test_ambiguous_path_and_content_disagree_returns_none(self):
        path = f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt"
        content = f"ACCESSION NUMBER:\t{ACC_2025}"
        assert extract_accession(path, content) is None

    def test_missing_returns_none(self):
        assert extract_accession("/data/AAPL/10-K/2025/filing.txt") is None
        assert extract_accession(None, "no accession here") is None
        assert extract_accession() is None

    def test_malformed_returns_none(self):
        # Wrong digit groups must not be accepted.
        assert extract_accession("/x/0000320193-4-000123/f.txt") is None
        assert extract_accession(None, "ACCESSION NUMBER: 0000320193-24-00012") is None

    def test_filename_variant_not_matched(self):
        # Only directory segments are valid; a filename suffix must not match.
        assert extract_accession(f"/data/AAPL/10-K/2025/{ACC_2024}.txt") is None

    def test_accepts_pathlib_path(self):
        path = Path(f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt")
        assert extract_accession(path) == ACC_2024


# ===========================================================================
# Amendment-aware document type
# ===========================================================================

class TestAmendmentAwareDocType:

    def test_path_amendment_keeps_suffix(self):
        assert extract_doc_type(f"/x/10-K/A/{ACC_2024}/f.txt") == "10-K/A"
        assert extract_doc_type("/x/10-Q/A/2024/f.txt") == "10-Q/A"
        assert extract_doc_type("/x/8-K/A/2024/f.txt") == "8-K/A"

    def test_path_base_type_unchanged(self):
        assert extract_doc_type("/x/10-K/2024/f.txt") == "10-K"
        assert extract_doc_type("/x/10-Q/2024/f.txt") == "10-Q"

    def test_content_amendment_keeps_suffix(self):
        assert extract_doc_type("filing.txt", "<TYPE>10-Q/A\nbody") == "10-Q/A"
        assert extract_doc_type("filing.txt", "<TYPE>10-K/A\nbody") == "10-K/A"

    def test_content_base_type(self):
        assert extract_doc_type("filing.txt", "<TYPE>10-K\nbody") == "10-K"

    def test_unknown_default(self):
        assert extract_doc_type("filing.txt", "no type tag") == "UNKNOWN"


# ===========================================================================
# extract_metadata source_id propagation
# ===========================================================================

class TestExtractMetadataSourceId:

    def test_explicit_source_id_wins(self):
        meta = extract_metadata(
            file_path="/data/AAPL/10-K/2025/f.txt",
            source_id="upload:abc123",
        )
        assert meta["source_id"] == "upload:abc123"

    def test_auto_extracted_from_path(self):
        meta = extract_metadata(
            file_path=f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt",
        )
        assert meta["source_id"] == ACC_2024

    def test_none_when_unresolvable(self):
        meta = extract_metadata(file_path="/data/AAPL/10-K/2025/f.txt")
        assert meta["source_id"] is None

    def test_amendment_doc_type_in_metadata(self):
        meta = extract_metadata(file_path=f"/data/AAPL/10-K/A/{ACC_2024}/f.txt")
        assert meta["doc_type"] == "10-K/A"
        assert meta["source_id"] == ACC_2024

    def test_chunk_id_formula_unchanged(self):
        # D6-A must not alter chunk id generation (would re-mint every id).
        meta = extract_metadata(
            file_path=f"/data/AAPL/10-K/{ACC_2024}/full-submission.txt",
            chunk_index=3,
        )
        assert meta["chunk_id"].endswith("_0003")
        assert meta["chunk_id"].startswith("AAPL_")


# ===========================================================================
# Upload identity + safe save path (app.api.main helpers)
# ===========================================================================

import app.api.main as main  # noqa: E402


class TestResolveUploadIdentity:

    def test_explicit_source_id_wins(self):
        assert main._resolve_upload_identity(b"anything", "upload:xyz") == "upload:xyz"

    def test_extracts_accession_from_header_bytes(self):
        raw = f"<SEC-DOCUMENT>{ACC_2024}.txt\nACCESSION NUMBER:\t{ACC_2024}\n".encode()
        assert main._resolve_upload_identity(raw, None) == ACC_2024

    def test_binary_without_header_returns_none(self):
        assert main._resolve_upload_identity(b"%PDF-1.4 binary junk", None) is None


class TestSafePathComponent:

    def test_strips_traversal(self):
        out = main._safe_path_component("../../etc/passwd")
        assert "/" not in out and ".." not in out and "\\" not in out

    def test_empty_becomes_default(self):
        assert main._safe_path_component("") == "unknown"
        assert main._safe_path_component("...") == "unknown"

    def test_preserves_legit_values(self):
        assert main._safe_path_component("AAPL") == "AAPL"
        assert main._safe_path_component("2024") == "2024"
        assert main._safe_path_component("10-K") == "10-K"
        assert main._safe_path_component(ACC_2024) == ACC_2024


class TestResolveUploadSavePath:

    def test_legacy_layout_preserved_without_source_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "DATA_DIR", tmp_path)
        target = main._resolve_upload_save_path("TEST", "2024", "save_test.pdf", None, None)
        assert target == tmp_path / "TEST" / "10-K" / "2024" / "save_test.pdf"

    def test_source_id_isolates_under_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "DATA_DIR", tmp_path)
        target = main._resolve_upload_save_path(
            "AAPL", "2024", "full-submission.txt", ACC_2024, None
        )
        assert target == tmp_path / "AAPL" / "10-K" / "2024" / ACC_2024 / "full-submission.txt"

    def test_explicit_doc_type_used_in_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "DATA_DIR", tmp_path)
        target = main._resolve_upload_save_path("AAPL", "2024", "f.txt", None, "10-K/A")
        # A path segment cannot contain "/", so the amendment marker is
        # sanitized on disk; the authoritative doc_type is carried in metadata.
        assert target == tmp_path / "AAPL" / "10-K_A" / "2024" / "f.txt"

    def test_collision_does_not_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "DATA_DIR", tmp_path)
        first = main._resolve_upload_save_path("TEST", "2024", "dup.pdf", None, None)
        first.write_bytes(b"ORIGINAL")
        second = main._resolve_upload_save_path("TEST", "2024", "dup.pdf", None, None)
        assert second != first
        assert second.name.endswith("_dup.pdf")
        # Second call must not have touched the original file.
        assert first.read_bytes() == b"ORIGINAL"

    def test_sanitizes_malicious_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "DATA_DIR", tmp_path)
        target = main._resolve_upload_save_path("../../etc", "2024", "f.txt", None, None)
        resolved = target.resolve()
        assert str(tmp_path.resolve()) in str(resolved)
        assert ".." not in target.parts


# ===========================================================================
# _ingest_document forwards identity + doc_type
# ===========================================================================

class TestIngestDocumentPropagation:

    def test_forwards_source_id_and_doc_type(self, tmp_path, monkeypatch):
        import cleaning
        import metadata_extractor as me
        import hybrid_chunker

        filing = tmp_path / "full-submission.txt"
        filing.write_bytes(b"<DOCUMENT><TYPE>10-K\nbody text")

        captured = {}

        class StubParser:
            def parse_file(self, content, filename):
                return {"text_content": "body text", "tables": [], "raw_html": "body text"}

        monkeypatch.setattr(main, "APIFileParser", StubParser)
        monkeypatch.setattr(cleaning, "clean_financial_text", lambda t: t)

        def fake_extract_metadata(**kwargs):
            captured["extract_kwargs"] = kwargs
            return {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "doc_type": "10-K",
                "source_file": filing.name,
                "source_id": kwargs.get("source_id"),
                "section": "General",
                "contains_table": False,
                "chunk_id": "X_0000",
                "page_number": None,
            }

        def fake_chunk_document(**kwargs):
            captured["metadata_base"] = kwargs["metadata_base"]
            return []  # short-circuit before any DB/indexer work

        monkeypatch.setattr(me, "extract_metadata", fake_extract_metadata)
        monkeypatch.setattr(hybrid_chunker, "chunk_document", fake_chunk_document)

        result = main._ingest_document(
            filing, "AAPL", "2024", "task-1", ACC_2024, "10-K/A"
        )

        assert result["chunks_created"] == 0
        assert captured["extract_kwargs"]["source_id"] == ACC_2024
        assert captured["metadata_base"]["doc_type"] == "10-K/A"
        assert captured["metadata_base"]["source_id"] == ACC_2024
