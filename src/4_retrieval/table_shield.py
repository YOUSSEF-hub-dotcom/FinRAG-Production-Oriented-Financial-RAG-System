"""
Module 4 -- Retrieval Stage: Contextual Shredding & Table Shield (Post-Retrieval Part 2).

Protects financial tables from LLM hallucination while cleaning non-table chunks.

Table Guard Rule (Strict):
  - If chunk["metadata"]["contains_table"] == True: Pass the Markdown table and
    surrounding text through 100% UNTOUCHED (bypass LLM editing).
  - If chunk["metadata"]["contains_table"] == False: Clean asynchronously via
    small local LLM with a strict prompt that strips filler words, links, and
    noise WITHOUT altering numbers, dates, or financial symbols ($, %, ()).

Schema & Metadata Preservation:
  - Pydantic schema enforces returning `cleaned_text: str` while preserving
    100% of original metadata (ticker, fiscal_year, section, source_file,
    page_number, chunk_id, etc.).
"""

import asyncio
import json
import re
from typing import Any

from config.logging_config import get_logger
from config.settings import GROQ_PRIMARY_MODEL, GROQ_API_KEY

logger = get_logger("retrieval.table_shield")

# Financial symbol pattern - must be preserved exactly
# Matches: $, %, parentheses, dates (Jan 2024, FY2024, 2024-12-31), and
# decimal numbers with a dot (724.5, 1.2) but NOT bare integers which can
# appear in URLs, page numbers, etc. and legitimately be cleaned.
_FINANCIAL_SYMBOLS = re.compile(r"[\$\%\(\)]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{4}|\b\d+\.\d+\b")

# Strict cleaning prompt for non-table chunks
_CLEANING_PROMPT = """You are a financial text cleaner. Your task is to strip filler words, links, and noise from the input text WITHOUT altering any numbers, dates, financial symbols ($, %, parentheses), or metric names.

RULES (MUST FOLLOW EXACTLY):
1. Remove filler phrases (e.g., "in order to", "it should be noted that", "please note", "as mentioned above", "furthermore", "additionally", "moreover").
2. Remove URLs, email addresses, and markdown links.
3. Remove excessive whitespace and normalize to single spaces.
4. DO NOT change, round, or remove ANY numbers (including decimals like 724.5, 1.2, 0.75).
5. DO NOT change, remove, or alter ANY financial symbols: $, %, (, ).
6. DO NOT change or remove ANY dates (e.g., "January 2024", "FY2024", "2024-12-31").
7. DO NOT change metric names or section headers.
8. Preserve markdown table syntax (| ... |) if present.
9. Output ONLY the cleaned text. No explanations, no formatting, no extra commentary.

INPUT TEXT:
{text}

CLEANED TEXT:"""


class TableShieldOutput:
    """Pydantic-like output schema for cleaned chunks (lightweight, no pydantic dep)."""

    __slots__ = ("cleaned_text", "metadata")

    def __init__(self, cleaned_text: str, metadata: dict):
        self.cleaned_text = cleaned_text
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {"cleaned_text": self.cleaned_text, "metadata": self.metadata}


class TableShield:
    """
    Contextual Shredding & Table Shield processor.

    Processes chunks from the re-ranker:
      - Tables (contains_table=True) pass through 100% untouched.
      - Non-tables are cleaned asynchronously via LLM with strict financial preservation.

    Args:
        model: LLM model name (default from settings).
        api_key: Groq API key (default from settings).
        temperature: LLM temperature (0.0 for deterministic cleaning).
        max_tokens: Max tokens for cleaning response.
        concurrency: Max concurrent LLM calls (default 5).
    """

    def __init__(
        self,
        model: str = GROQ_PRIMARY_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        concurrency: int = 5,
        clean_enabled: bool = True,
    ):
        self._model = model
        self._api_key = api_key or GROQ_API_KEY
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._clean_enabled = clean_enabled
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._client = None

        logger.info(
            "TableShield configured: model=%s temp=%.2f max_tokens=%d concurrency=%d clean_enabled=%s",
            self._model,
            self._temperature,
            self._max_tokens,
            concurrency,
            clean_enabled,
        )

    def _get_client(self):
        """Lazy-initialise the Groq async client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
            except Exception as exc:
                logger.error("groq package not available: %s", exc)
                raise RuntimeError("TableShield requires 'groq' package") from exc
            if not self._api_key:
                raise RuntimeError("GROQ_API_KEY not set; TableShield cannot call LLM")
            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def _clean_chunk(self, chunk: dict) -> dict:
        """
        Clean a single non-table chunk via LLM.

        Returns a new chunk dict with cleaned text and preserved metadata.
        """
        text = chunk.get("text", "")
        metadata = chunk.get("metadata") or {}

        if not text.strip():
            return {**chunk, "text": "", "cleaned_text": ""}

        prompt = _CLEANING_PROMPT.format(text=text)

        async with self._semaphore:
            client = self._get_client()
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    seed=42,
                )
                cleaned = response.choices[0].message.content or ""
                cleaned = cleaned.strip()
            except Exception as exc:
                logger.error("LLM cleaning failed for chunk %s: %s", metadata.get("chunk_id"), exc)
                # Graceful degradation: return original text
                cleaned = text

        # Verify financial symbols/numbers preserved (defensive)
        if not self._verify_preservation(text, cleaned):
            logger.warning(
                "Financial preservation check failed for chunk %s -- reverting to original",
                metadata.get("chunk_id"),
            )
            cleaned = text

        return {
            **chunk,
            "text": cleaned,
            "cleaned_text": cleaned,
            "metadata": dict(metadata),  # Ensure metadata preserved
        }

    def _verify_preservation(self, original: str, cleaned: str) -> bool:
        """
        Verify that all financial symbols, numbers, and dates are preserved.

        This is a best-effort check; the LLM prompt is the primary guarantee.
        """
        # Extract all financial tokens from original
        original_tokens = set(_FINANCIAL_SYMBOLS.findall(original))
        if not original_tokens:
            return True  # Nothing to preserve

        # Check each token exists in cleaned (allowing for whitespace differences)
        cleaned_normalised = re.sub(r"\s+", " ", cleaned)
        for token in original_tokens:
            if token not in cleaned_normalised:
                logger.debug("Missing token in cleaned text: %s", token)
                return False
        return True

    async def shield(self, chunks: list[dict]) -> list[dict]:
        """
        Process chunks through the Table Shield.

        Tables pass through untouched; non-tables are cleaned asynchronously.

        Args:
            chunks: List of chunk dicts with 'text', 'metadata' (must contain
                'contains_table' boolean).

        Returns:
            List of processed chunk dicts with 'cleaned_text' field added,
            metadata 100% preserved.
        """
        if not chunks:
            logger.warning("TableShield called with empty chunk list")
            return []

        # Separate tables (passthrough) from non-tables (async clean)
        table_chunks = []
        non_table_chunks = []

        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if meta.get("contains_table") is True:
                table_chunks.append(chunk)
            else:
                non_table_chunks.append(chunk)

        logger.info(
            "TableShield processing: %d tables (passthrough), %d non-tables (async clean)",
            len(table_chunks),
            len(non_table_chunks),
        )

        # Tables: pass through 100% untouched with cleaned_text = original text
        table_results = [
            {**chunk, "cleaned_text": chunk.get("text", ""), "metadata": dict(chunk.get("metadata", {}))}
            for chunk in table_chunks
        ]

        # Non-tables: clean asynchronously (unless LLM cleaning is disabled,
        # in which case they pass through untouched like tables).
        if non_table_chunks:
            if not self._clean_enabled:
                processed_non_tables = [
                    {
                        **chunk,
                        "cleaned_text": chunk.get("text", ""),
                        "metadata": dict(chunk.get("metadata", {})),
                    }
                    for chunk in non_table_chunks
                ]
            else:
                tasks = [self._clean_chunk(chunk) for chunk in non_table_chunks]
                non_table_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle any exceptions gracefully
                processed_non_tables = []
                for i, result in enumerate(non_table_results):
                    if isinstance(result, Exception):
                        logger.error("Async clean task failed: %s", result)
                        original = non_table_chunks[i]
                        processed_non_tables.append(
                            {
                                **original,
                                "cleaned_text": original.get("text", ""),
                                "metadata": dict(original.get("metadata", {})),
                            }
                        )
                    else:
                        processed_non_tables.append(result)
        else:
            processed_non_tables = []

        # Combine: preserve original order by chunk_id
        all_results = table_results + processed_non_tables
        # Sort by original position if possible, otherwise keep table-first order
        chunk_order = {chunk.get("chunk_id", f"idx_{i}"): i for i, chunk in enumerate(chunks)}
        all_results.sort(key=lambda c: chunk_order.get(c.get("chunk_id"), 999999))

        logger.info("TableShield complete: %d chunks returned", len(all_results))
        return all_results

    # Sync wrapper for convenience
    def shield_sync(self, chunks: list[dict]) -> list[dict]:
        """Synchronous entry point (runs async shield in new event loop)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.shield(chunks))
        # If already in a loop, we can't use run(); caller should use await shield()
        raise RuntimeError(
            "TableShield.shield_sync() cannot run inside a live event loop; "
            "use await shield() instead."
        )


async def shield_chunks(
    chunks: list[dict],
    model: str = GROQ_PRIMARY_MODEL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    concurrency: int = 5,
) -> list[dict]:
    """
    Functional entry point for one-shot table shielding.

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata.contains_table'.
        model: LLM model name.
        api_key: Groq API key.
        temperature: LLM temperature.
        max_tokens: Max tokens for cleaning.
        concurrency: Max concurrent calls.

    Returns:
        Processed chunks with 'cleaned_text' and preserved metadata.
    """
    shield = TableShield(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )
    return await shield.shield(chunks)