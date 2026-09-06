"""
Hybrid Chunker — 3-Tier token-bounded chunking engine for SEC filings.

Tier 1: Section-level splitting on SEC Item boundaries.
Tier 2: Table & footnote isolation as atomic chunks (never split).
Tier 3: Token-bounded recursive text splitting (768-1024 tokens, 20% overlap).

Chunk-size and overlap parameters are read from ``config.settings``
(``CHUNK_MIN_TOKENS`` / ``CHUNK_MAX_TOKENS`` / ``CHUNK_OVERLAP_RATIO``) so a
single config change re-bounds every ingestion run consistently.
"""

import hashlib
import re
from pathlib import Path

import tiktoken

from config.logging_config import get_logger
from config.settings import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_RATIO

logger = get_logger("ingestion.chunker")

# SEC Item boundary patterns for Tier 1 section splitting
_SECTION_BOUNDARIES = re.compile(
    r"(?=Item\s+\d+[A-Z]?[\:\.]\s|Part\s+(?:I{1,3}V?|IV)\b)",
    re.IGNORECASE,
)

# Markdown table pattern — lines starting/ending with pipes
_TABLE_LINE = re.compile(r"^\s*\|.+\|\s*$")

# Footnote pattern — common SEC footnote markers
_FOOTNOTE_PATTERN = re.compile(
    r"(?:^\s*\(\d+\)|^\s*\[\d+\]|^\s*\*\s|^\s*\d+\.\s)",
    re.MULTILINE,
)

# Table placeholder pattern from parser
_TABLE_PLACEHOLDER = re.compile(r"%%TABLE_\d+%%")


def _get_token_encoder() -> tiktoken.Encoding:
    """Load tiktoken encoder for cl100k_base (used by most modern models)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: tiktoken.Encoding | None = None) -> int:
    """Count exact token count using tiktoken encoder."""
    if encoder is None:
        encoder = _get_token_encoder()
    return len(encoder.encode(text))


def _split_on_sections(text: str) -> list[dict]:
    """
    Tier 1: Split text on SEC Item boundaries.

    Returns list of dicts with 'section_name' and 'text' keys.
    """
    parts = _SECTION_BOUNDARIES.split(text)
    sections = []

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        # Extract section name — find the Item/Part header in first 300 chars
        search_text = stripped[:300]
        header_match = re.search(
            r"((?:Item\s+\d+[A-Z]?[\:\.]\s+.+?)(?=\s{2,}|$)|Part\s+(?:I{1,3}V?|IV)\b)",
            search_text,
            re.IGNORECASE,
        )
        section_name = header_match.group(0).strip()[:80] if header_match else "General"

        sections.append({
            "section_name": section_name,
            "text": stripped,
        })

    logger.debug("Tier 1: Split document into %d sections", len(sections))
    return sections

def _extract_table_chunks(text: str) -> tuple[list[tuple[str, str]], str]:
    """
    Tier 2: Extract Markdown tables and their immediately following footnotes
    as atomic chunks, tracking the SEC section each table belongs to.

    Pre-scans the full text for Item/Part section boundaries, then maps each
    table's character position to its nearest preceding section — works even
    when the text has no newlines.

    Returns (table_chunks_with_sections, remaining_text) where each table
    chunk is a (section_name, table_text) tuple.
    """
    # Build a position-to-section map by finding all Item/Part headers
    section_positions: list[tuple[int, str]] = []
    for m in re.finditer(
        r"(Item\s+\d+[A-Z]?[\:\.]\s+.+?)(?=\s{2,}|$|Item\s|Part\s)",
        text,
        re.IGNORECASE,
    ):
        section_positions.append((m.start(), m.group(0).strip()[:80]))
    for m in re.finditer(
        r"(Part\s+(?:I{1,3}V?|IV)\b)",
        text,
        re.IGNORECASE,
    ):
        section_positions.append((m.start(), m.group(0).strip()[:80]))
    section_positions.sort(key=lambda x: x[0])

    def _find_section_at(pos: int) -> str:
        """Find the section that covers character position `pos`."""
        result = "General"
        for sec_pos, sec_name in section_positions:
            if sec_pos <= pos:
                result = sec_name
            else:
                break
        return result

    lines = text.split("\n")
    table_chunks: list[tuple[str, str]] = []
    non_table_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        is_table_start = bool(_TABLE_LINE.match(line))

        if is_table_start:
            # Find the character position of this table in the original text
            table_pos = text.find(line.strip()[:50]) if line.strip() else -1
            current_section = _find_section_at(table_pos) if table_pos >= 0 else "General"

            # Collect contiguous table lines
            table_buffer = [line]
            i += 1
            while i < len(lines) and bool(_TABLE_LINE.match(lines[i])):
                table_buffer.append(lines[i])
                i += 1

            # Collect immediately following footnote lines
            while i < len(lines) and bool(_FOOTNOTE_PATTERN.match(lines[i])):
                table_buffer.append(lines[i])
                i += 1

            table_text = "\n".join(table_buffer)
            table_chunks.append((current_section, table_text))
        else:
            non_table_lines.append(line)
            i += 1

    remaining_text = "\n".join(non_table_lines)
    logger.debug(
        "Tier 2: Extracted %d atomic table chunks, remaining text: %d chars",
        len(table_chunks),
        len(remaining_text),
    )
    return table_chunks, remaining_text

def _recursive_token_split(
    text: str,
    encoder: tiktoken.Encoding,
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Tier 3: Recursively split text into token-bounded chunks.

    Split priority: paragraph break -> sentence boundary -> word boundary -> character.
    Enforces strict token count limits using tiktoken.
    """
    if overlap_tokens is None:
        overlap_tokens = int(max_tokens * CHUNK_OVERLAP_RATIO)

    if not text.strip():
        return []

    total_tokens = count_tokens(text, encoder)
    if total_tokens <= max_tokens:
        return [text.strip()]

    # Split hierarchy: paragraphs -> sentences -> words
    split_patterns = [
        re.compile(r"\n{2,}"),           # Paragraph breaks
        re.compile(r"(?<=[.!?])\s+"),    # Sentence boundaries
        re.compile(r"\s+"),              # Word boundaries
    ]

    chunks = []
    for pattern in split_patterns:
        segments = pattern.split(text)
        # Filter empty segments
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) <= 1:
            continue

        current_chunk_parts = []
        current_token_count = 0

        for segment in segments:
            segment_tokens = count_tokens(segment, encoder)

            if current_token_count + segment_tokens <= max_tokens:
                current_chunk_parts.append(segment)
                current_token_count += segment_tokens
            else:
                # Flush current chunk if it meets minimum size
                if current_chunk_parts and current_token_count >= min_tokens:
                    chunks.append(" ".join(current_chunk_parts))

                # Handle segments larger than max_tokens — hard split
                if segment_tokens > max_tokens:
                    hard_chunks = _hard_split_segment(segment, encoder, max_tokens)
                    chunks.extend(hard_chunks)
                    current_chunk_parts = []
                    current_token_count = 0
                else:
                    # Start new chunk with overlap from previous
                    overlap_parts = []
                    overlap_count = 0
                    for prev_part in reversed(current_chunk_parts):
                        prev_tokens = count_tokens(prev_part, encoder)
                        if overlap_count + prev_tokens <= overlap_tokens:
                            overlap_parts.insert(0, prev_part)
                            overlap_count += prev_tokens
                        else:
                            break

                    current_chunk_parts = overlap_parts + [segment]
                    current_token_count = overlap_count + segment_tokens

        # Flush remaining
        if current_chunk_parts and current_token_count >= min_tokens:
            chunks.append(" ".join(current_chunk_parts))
        elif current_chunk_parts and chunks:
            # Merge small final chunk with previous if within limits
            last = chunks[-1]
            merged = last + " " + " ".join(current_chunk_parts)
            if count_tokens(merged, encoder) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(" ".join(current_chunk_parts))

        # If this split level produced valid chunks, use it
        if len(chunks) > 1:
            break

    # Fallback: if no splits worked, return the text as a single chunk
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


def _hard_split_segment(
    text: str, encoder: tiktoken.Encoding, max_tokens: int
) -> list[str]:
    """Force-split a segment that exceeds max_tokens by encoding and slicing."""
    tokens = encoder.encode(text)
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(encoder.decode(chunk_tokens).strip())
    return chunks


def _make_chunk_id(ticker: str, fiscal_year: str, chunk_type: str, source_file: str, index: int) -> str:
    """
    Generate a deterministic chunk ID so re-ingestion is idempotent.

    Format: {TICKER}_{txt|tbl}_{md5 hash}_{index:04d}
    Same inputs always yield the same ID (stable across runs).
    fiscal_year is part of the hash input so filings from different years
    of the same company never collide.
    """
    hash_input = f"{ticker}:{fiscal_year}:{chunk_type}:{source_file}:{index}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return f"{ticker}_{chunk_type}_{short_hash}_{index:04d}"


def chunk_document(
    text: str,
    tables: list[str] | None = None,
    file_path: str | Path = "",
    metadata_base: dict | None = None,
) -> list[dict]:
    """
    Execute the full 3-Tier chunking pipeline on parsed document content.

    Args:
        text: Cleaned text content from Stage 1 parser.
        tables: List of Markdown table strings extracted by parser.
        file_path: Source file path for metadata.
        metadata_base: Base metadata dict from metadata_extractor.

    Returns:
        List of chunk dicts, each containing:
            'chunk_id', 'text', 'chunk_type' ('table'|'text'),
            'token_count', 'metadata' (with section, contains_table, etc.)
    """
    encoder = _get_token_encoder()
    all_chunks: list[dict] = []

    if metadata_base is None:
        metadata_base = {}

    source_name = metadata_base.get("source_file") or str(file_path)

        # --- Tier 2: Extract table atomic chunks from text ---
    # Parser replaces HTML tables with %%TABLE_N%% placeholders in the text.
    # We map each placeholder's character position to its SEC section so parser-
    # extracted tables get correct section labels (not "General").
    placeholder_positions: dict[int, int] = {}
    for _m in re.finditer(r"%%TABLE_(\d+)%%", text):
        placeholder_positions[int(_m.group(1))] = _m.start()

    # Build section-position map from the cleaned text
    section_positions: list[tuple[int, str]] = []
    for _m in re.finditer(
        r"(Item\s+\d+[A-Z]?[\:\.]\s+.+?)(?=\s{2,}|$|Item\s|Part\s)",
        text,
        re.IGNORECASE,
    ):
        section_positions.append((_m.start(), _m.group(0).strip()[:80]))
    for _m in re.finditer(
        r"(Part\s+(?:I{1,3}V?|IV)\b)",
        text,
        re.IGNORECASE,
    ):
        section_positions.append((_m.start(), _m.group(0).strip()[:80]))
    section_positions.sort(key=lambda x: x[0])

    def _find_section_at(pos: int) -> str:
        result = "General"
        for sec_pos, sec_name in section_positions:
            if sec_pos <= pos:
                result = sec_name
            else:
                break
        return result

    # Map parser-extracted tables to their sections via placeholder positions
    parser_tables_with_sections: list[tuple[str, str]] = []
    for _idx, _tbl in enumerate(tables or []):
        if _idx in placeholder_positions:
            _sec = _find_section_at(placeholder_positions[_idx])
        else:
            _sec = metadata_base.get("section", "General")
        parser_tables_with_sections.append((_sec, _tbl))

    # Text-extracted tables also get sections from position map
    text_tables_raw, remaining_text = _extract_table_chunks(text)
    text_tables = text_tables_raw  # Already (section, text) tuples

    all_table_chunks = parser_tables_with_sections + text_tables

    # Create atomic table chunks (with correct section assignment)
    for idx, (tbl_section, table_text) in enumerate(all_table_chunks):
        if not table_text.strip():
            continue
        chunk_id = _make_chunk_id(
            metadata_base.get("ticker", "UNK"),
            metadata_base.get("fiscal_year", "UNKNOWN"),
            "tbl",
            source_name,
            idx,
        )
        chunk = {
            "chunk_id": chunk_id,
            "text": table_text.strip(),
            "chunk_type": "table",
            "token_count": count_tokens(table_text, encoder),
            "metadata": {
                **metadata_base,
                "section": tbl_section,
                "contains_table": True,
                "chunk_type": "table",
            },
        }
        all_chunks.append(chunk)

# --- Tier 1: Split remaining text on section boundaries ---
    sections = _split_on_sections(remaining_text)

    # --- Tier 3: Token-bounded splitting within each section ---
    text_chunk_idx = 0
    for section in sections:
        section_text = section["text"]
        section_name = section["section_name"]

        # Tier 3: Recursive token splitting
        sub_chunks = _recursive_token_split(section_text, encoder)

        for sub_text in sub_chunks:
            chunk_id = _make_chunk_id(
                metadata_base.get("ticker", "UNK"),
                metadata_base.get("fiscal_year", "UNKNOWN"),
                "txt",
                source_name,
                text_chunk_idx,
            )
            chunk = {
                "chunk_id": chunk_id,
                "text": sub_text,
                "chunk_type": "text",
                "token_count": count_tokens(sub_text, encoder),
                "metadata": {
                    **metadata_base,
                    "section": section_name,
                    "contains_table": False,
                    "chunk_type": "text",
                },
            }
            all_chunks.append(chunk)
            text_chunk_idx += 1

    logger.info(
        "Chunking complete: %d total chunks (%d text, %d table)",
        len(all_chunks),
        len([c for c in all_chunks if c["chunk_type"] == "text"]),
        len([c for c in all_chunks if c["chunk_type"] == "table"]),
    )
    return all_chunks
