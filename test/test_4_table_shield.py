"""
Unit tests for Module 4 Part 2: Post-Retrieval Engine - Table Shield.

Validates:
  - TableShield initialisation (model, API key, concurrency).
  - Table passthrough logic (100% untouched).
  - Non-table async cleaning (LLM call, financial preservation).
  - Metadata preservation (100% pass-through).
  - Sync wrapper (shield_sync) API contract.
  - Functional shield_chunks entry point.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

from table_shield import TableShield, _CLEANING_PROMPT


# ============================================================================
# Test doubles & Fixtures
# ============================================================================

_SAMPLE_TABLE_CHUNK = {
    "chunk_id": "tbl-001",
    "text": "| Metric | Value |\n|--------|-------|\n| Revenue | 724.5 |\n| Profit  | 97.0  |",
    "metadata": {
        "ticker": "AAPL",
        "fiscal_year": "2024",
        "section": "Item 8",
        "source_file": "aapl_10k_2024.pdf",
        "page_number": "30",
        "contains_table": True,
        "chunk_id": "tbl-001",
    },
}

_SAMPLE_TEXT_CHUNK = {
    "chunk_id": "txt-001",
    "text": "Apple reported Q3 revenue of 724.5 million dollars. "
            "The company also announced EPS of $3.15, up from $2.80 YoY. "
            "More details: https://example.com/news/123 and see chart.png",
    "metadata": {
        "ticker": "AAPL",
        "fiscal_year": "2024",
        "section": "Item 7",
        "source_file": "aapl_10k_2024.pdf",
        "page_number": "20",
        "contains_table": False,
        "chunk_id": "txt-001",
    },
}

_SAMPLE_CLEANED_EXPECTED = (
    "Apple reported Q3 revenue of 724.5 million dollars. "
    "The company also announced EPS of $3.15, up from $2.80 YoY."
)


def _make_shield(**kwargs):
    """Create a TableShield with GROQ_API_KEY forced for test isolation."""
    with patch("table_shield.GROQ_API_KEY", "test-dummy-key"):
        return TableShield(api_key="test-dummy-key", **kwargs)


def _make_text_shield(mock_client, **kwargs):
    """Create a TableShield pre-loaded with a mock client for non-table cleaning tests."""
    shield = _make_shield(**kwargs)
    shield._client = mock_client
    return shield


# ============================================================================
# Constructor
# ============================================================================

class TestTableShieldConstructor:
    """Validate TableShield initialisation."""

    def test_init_defaults(self):
        shield = _make_shield()
        assert shield._model == "openai/gpt-oss-120b"
        assert shield._temperature == 0.0
        assert shield._max_tokens == 512
        assert shield._semaphore._value == 5

    def test_init_custom_params(self):
        shield = _make_shield(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            max_tokens=1024,
            concurrency=3,
        )
        assert shield._model == "openai/gpt-oss-20b"
        assert shield._temperature == 0.1
        assert shield._max_tokens == 1024
        assert shield._semaphore._value == 3

    def test_api_key_requirement(self):
        with patch("table_shield.GROQ_API_KEY", ""):
            shield = TableShield(api_key="")
            shield._client = None
            with patch.dict("sys.modules", {"groq": Mock()}):
                with pytest.raises(RuntimeError, match="GROQ_API_KEY not set"):
                    shield._get_client()


# ============================================================================
# Table passthrough (strict guard rule)
# ============================================================================

class TestTableShieldTablePassthrough:
    """Validate strict table guard rule: tables pass through 100% untouched."""

    @pytest.mark.asyncio
    async def test_table_passthrough_untouched(self):
        shield = _make_shield()
        result = await shield.shield([_SAMPLE_TABLE_CHUNK])

        assert len(result) == 1
        assert result[0]["chunk_id"] == "tbl-001"
        assert result[0]["text"] == _SAMPLE_TABLE_CHUNK["text"]
        assert result[0]["cleaned_text"] == _SAMPLE_TABLE_CHUNK["text"]
        assert result[0]["metadata"]["contains_table"] is True

    @pytest.mark.asyncio
    async def test_multiple_tables_passthrough(self):
        shield = _make_shield()
        table1 = {**_SAMPLE_TABLE_CHUNK, "chunk_id": "tbl-1"}
        table2 = {**_SAMPLE_TABLE_CHUNK, "chunk_id": "tbl-2"}

        result = await shield.shield([table1, table2])

        assert len(result) == 2
        assert result[0]["chunk_id"] == "tbl-1"
        assert result[1]["chunk_id"] == "tbl-2"
        assert all(c["cleaned_text"] == _SAMPLE_TABLE_CHUNK["text"] for c in result)

    @pytest.mark.asyncio
    async def test_table_metadata_preserved(self):
        shield = _make_shield()
        result = await shield.shield([_SAMPLE_TABLE_CHUNK])

        chunk = result[0]
        for key in _SAMPLE_TABLE_CHUNK["metadata"]:
            assert chunk["metadata"][key] == _SAMPLE_TABLE_CHUNK["metadata"][key]
        assert chunk["metadata"]["contains_table"] is True


# ============================================================================
# Non-table async cleaning
# ============================================================================

class TestTableShieldNonTableCleaning:
    """Validate async LLM cleaning for non-table chunks."""

    @pytest.mark.asyncio
    async def test_non_table_chunk_cleaned(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=_SAMPLE_CLEANED_EXPECTED))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)
        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        assert len(result) == 1
        assert result[0]["chunk_id"] == "txt-001"
        assert result[0]["cleaned_text"] == _SAMPLE_CLEANED_EXPECTED
        assert result[0]["text"] == _SAMPLE_CLEANED_EXPECTED
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_table_chunk_with_urls_links_removed(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Apple reported Q3 revenue of 724.5 million dollars. The company also announced EPS of $3.15, up from $2.80 YoY."))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)
        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        cleaned_text = result[0]["cleaned_text"]
        assert "https://example.com/news/123" not in cleaned_text
        assert "chart.png" not in cleaned_text
        assert "Apple reported Q3 revenue of 724.5 million dollars." in cleaned_text

    @pytest.mark.asyncio
    async def test_non_table_chunk_preserves_financial_symbols(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Apple reported Q3 revenue of 724.5 million dollars. EPS of $3.15, YoY growth 10%, (operating margin) 15.2%."))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)
        # Bypass verification to test LLM cleaning independently
        shield._verify_preservation = Mock(return_value=True)

        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        cleaned = result[0]["cleaned_text"]
        assert "724.5" in cleaned
        assert "$3.15" in cleaned
        assert "10%" in cleaned
        assert "(" in cleaned and ")" in cleaned
        assert "$3.15" in cleaned
        assert "10%" in cleaned

    @pytest.mark.asyncio
    async def test_non_table_chunk_financial_preservation_verification(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Some text"))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)
        shield._verify_preservation = Mock(return_value=False)

        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        assert result[0]["cleaned_text"] == _SAMPLE_TEXT_CHUNK["text"]
        shield._verify_preservation.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_cleaning(self):
        chunks = [
            {**_SAMPLE_TEXT_CHUNK, "chunk_id": f"txt-{i}", "text": f"Text chunk {i} with 724.5 million dollars."}
            for i in range(5)
        ]

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Cleaned text"))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client, concurrency=2)
        result = await shield.shield(chunks)

        assert len(result) == 5
        assert mock_client.chat.completions.create.call_count == 5

    @pytest.mark.asyncio
    async def test_empty_text_chunk(self):
        shield = _make_shield()
        shield._client = AsyncMock()

        empty_chunk = {"chunk_id": "empty", "text": "", "metadata": {"contains_table": False}}
        whitespace_chunk = {"chunk_id": "ws", "text": "   \n  ", "metadata": {"contains_table": False}}

        result = await shield.shield([empty_chunk, whitespace_chunk])

        assert len(result) == 2
        assert result[0]["cleaned_text"] == ""
        assert result[1]["cleaned_text"] == ""
        shield._client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_call_exception_handling(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        shield = _make_text_shield(mock_client)
        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        assert result[0]["cleaned_text"] == _SAMPLE_TEXT_CHUNK["text"]
        assert result[0]["text"] == _SAMPLE_TEXT_CHUNK["text"]


# ============================================================================
# Order and metadata preservation
# ============================================================================

class TestTableShieldOrderAndMetadata:
    """Validate chunk order preservation and metadata pass-through."""

    @pytest.mark.asyncio
    async def test_mixed_tables_and_non_tables_order(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Cleaned text"))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)

        mixed_chunks = [
            _SAMPLE_TABLE_CHUNK,
            _SAMPLE_TEXT_CHUNK,
            {**_SAMPLE_TABLE_CHUNK, "chunk_id": "tbl-002"},
            {**_SAMPLE_TEXT_CHUNK, "chunk_id": "txt-002"},
        ]

        result = await shield.shield(mixed_chunks)

        assert len(result) == 4
        assert {c["chunk_id"] for c in result} == {"tbl-001", "txt-001", "tbl-002", "txt-002"}
        for chunk in result:
            if "tbl" in chunk["chunk_id"]:
                assert chunk["metadata"]["contains_table"] is True
                assert chunk["cleaned_text"] == _SAMPLE_TABLE_CHUNK["text"]

    @pytest.mark.asyncio
    async def test_metadata_preservation_100_percent(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Cleaned text"))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)

        extensive_chunk = {
            "chunk_id": "ext-001",
            "text": "Some text",
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Item 7",
                "source_file": "aapl_10k_2024.pdf",
                "page_number": "20",
                "contains_table": False,
                "chunk_id": "ext-001",
                "extra_field": "preserve_this",
                "another_field": 12345,
            },
        }

        result = await shield.shield([extensive_chunk])

        for key in extensive_chunk["metadata"]:
            assert result[0]["metadata"][key] == extensive_chunk["metadata"][key]
        assert result[0]["metadata"]["contains_table"] is False


# ============================================================================
# Sync wrapper
# ============================================================================

class TestTableShieldSyncWrapper:
    """Validate shield_sync sync wrapper and loop contracts."""

    def test_shield_sync_outside_event_loop(self):
        shield = _make_shield()

        async def fake_shield(chunks):
            return [{"chunk_id": "test"}]

        with patch.object(shield, "shield", side_effect=fake_shield):
            result = shield.shield_sync([_SAMPLE_TEXT_CHUNK])
            assert result == [{"chunk_id": "test"}]

    def test_shield_sync_inside_live_loop(self):
        shield = _make_shield()

        async def inner():
            with pytest.raises(RuntimeError, match="cannot run inside a live event loop"):
                shield.shield_sync([_SAMPLE_TEXT_CHUNK])

        asyncio.run(inner())


# ============================================================================
# Functional entry point
# ============================================================================

class TestShieldChunksFunctional:
    """Validate shield_chunks functional interface."""

    @pytest.mark.asyncio
    async def test_shield_chunks_creates_shield(self):
        from table_shield import shield_chunks

        mock_shield = AsyncMock()
        mock_shield.shield.return_value = [{"cleaned_text": "cleaned", "metadata": {}}]

        with patch("table_shield.TableShield", return_value=mock_shield):
            result = await shield_chunks(
                [_SAMPLE_TEXT_CHUNK],
                model="test/model",
                api_key="test-key",
                temperature=0.1,
                max_tokens=1024,
                concurrency=3,
            )

            mock_shield.shield.assert_called_once_with([_SAMPLE_TEXT_CHUNK])
            assert result == [{"cleaned_text": "cleaned", "metadata": {}}]

    @pytest.mark.asyncio
    async def test_shield_chunks_defaults(self):
        from table_shield import shield_chunks

        mock_shield = AsyncMock()
        mock_shield.shield.return_value = []

        with patch("table_shield.TableShield", return_value=mock_shield):
            result = await shield_chunks([_SAMPLE_TEXT_CHUNK])

            mock_shield.shield.assert_called_once_with([_SAMPLE_TEXT_CHUNK])


# ============================================================================
# Defensive programming
# ============================================================================

class TestTableShieldDefensiveProgramming:
    """Validate defensive checks and error recovery."""

    @pytest.mark.asyncio
    async def test_null_metadata_handling(self):
        shield = _make_shield()
        shield._client = AsyncMock()

        chunk = {"chunk_id": "no_meta", "text": "text", "metadata": None}
        result = await shield.shield([chunk])

        assert result[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_missing_metadata_key(self):
        shield = _make_shield()
        shield._client = AsyncMock()

        chunk = {"chunk_id": "no_meta_key", "text": "text"}
        result = await shield.shield([chunk])

        assert result[0]["metadata"] == {}

    def test_prompt_template_formatting(self):
        test_text = "Hello world 123.45 dollars."
        formatted = _CLEANING_PROMPT.format(text=test_text)
        assert "Hello world 123.45 dollars." in formatted
        assert "You are a financial text cleaner" in formatted
        assert "DO NOT change, round, or remove ANY numbers" in formatted


# ============================================================================
# End-to-end mock integration
# ============================================================================

class TestTableShieldEndToEnd:
    """Integration-style tests with deterministic mocks."""

    @pytest.mark.asyncio
    async def test_end_to_end_table_passthrough(self):
        shield = _make_shield()
        result = await shield.shield([_SAMPLE_TABLE_CHUNK])

        assert result[0]["cleaned_text"] == _SAMPLE_TABLE_CHUNK["text"]
        assert result[0]["metadata"]["contains_table"] is True

    @pytest.mark.asyncio
    async def test_end_to_end_non_table_cleaning(self):
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Cleaned financial text"))]
        mock_client.chat.completions.create.return_value = mock_response

        shield = _make_text_shield(mock_client)
        # Bypass verification to test cleaning pipeline end-to-end
        shield._verify_preservation = Mock(return_value=True)

        result = await shield.shield([_SAMPLE_TEXT_CHUNK])

        mock_client.chat.completions.create.assert_called_once()
        assert result[0]["cleaned_text"] == "Cleaned financial text"
        assert result[0]["metadata"]["ticker"] == "AAPL"