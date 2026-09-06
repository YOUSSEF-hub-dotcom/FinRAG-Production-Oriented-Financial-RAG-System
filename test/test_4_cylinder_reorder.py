"""
Unit tests for Module 4 Part 2: Post-Retrieval Engine - Cylinder Re-order.

Validates:
  - cylinder_reorder function with default and custom patterns.
  - Dynamic pattern generation for any chunk count (best at head, 2nd-best
    at tail) -- the 5-chunk case reproduces the legacy (1,3,5,4,2) order.
  - 8-chunk cylinder layout used by the production post-retrieval stage.
  - Input validation (explicit patterns must be a permutation of the input).
  - CylinderReorderer class functionality.
  - Backwards-compatible reorder_cylinder alias.
"""

import sys
from pathlib import Path

import pytest

# --- Path Setup ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "4_retrieval"))

from cylinder_reorder import cylinder_reorder, CylinderReorderer, reorder_cylinder

# ============================================================================
# Test data
# ============================================================================

_SAMPLE_FIVE_CHUNKS = [
    {"chunk_id": "rank1", "text": "First highest relevance", "rrf_score": 0.95},
    {"chunk_id": "rank2", "text": "Second highest relevance", "rrf_score": 0.85},
    {"chunk_id": "rank3", "text": "Third highest relevance", "rrf_score": 0.75},
    {"chunk_id": "rank4", "text": "Fourth highest relevance", "rrf_score": 0.65},
    {"chunk_id": "rank5", "text": "Fifth lowest relevance", "rrf_score": 0.55},
]

_SAMPLE_EIGHT_CHUNKS = [
    {"chunk_id": f"rank{i}", "text": f"Chunk {i}", "rerank_score": 1.0 - i * 0.1}
    for i in range(1, 9)
]


# ============================================================================
# Default pattern tests
# ============================================================================

class TestCylinderReorderDefaultPattern:
    """Test cylinder_reorder with the default cylinder pattern (0,2,4,3,1)."""

    def test_default_pattern_order(self):
        """Default pattern should reorder as [1,3,5,4,2] (1-based ranks)."""
        result = cylinder_reorder(_SAMPLE_FIVE_CHUNKS)
        assert len(result) == 5
        # Pattern (0,2,4,3,1) maps to: rank1, rank3, rank5, rank4, rank2
        assert [c["chunk_id"] for c in result] == ["rank1", "rank3", "rank5", "rank4", "rank2"]

    def test_default_pattern_scores(self):
        """Scores are preserved through reorder."""
        result = cylinder_reorder(_SAMPLE_FIVE_CHUNKS)
        scores = [c["rrf_score"] for c in result]
        assert scores == [0.95, 0.75, 0.55, 0.65, 0.85]

    def test_default_pattern_with_rerank_scores(self):
        """Pattern works with rerank_score field."""
        chunks = [
            {"chunk_id": f"pos{i}", "text": f"text{i}", "rerank_score": 1.0 - i * 0.1}
            for i in range(5)
        ]
        result = cylinder_reorder(chunks)
        assert [c["chunk_id"] for c in result] == ["pos0", "pos2", "pos4", "pos3", "pos1"]

    def test_preserves_all_fields(self):
        """All original chunk fields survive reordering."""
        chunks = [
            {
                "chunk_id": f"c{i}",
                "text": f"text{i}",
                "metadata": {"ticker": f"T{i}", "page": i},
                "source": f"source{i}",
                "extra": "preserve",
                "rerank_score": 1.0 - i * 0.1,
            }
            for i in range(5)
        ]
        result = cylinder_reorder(chunks)

        # Expected order by pattern (0,2,4,3,1): c0, c2, c4, c3, c1
        expected_ids = ["c0", "c2", "c4", "c3", "c1"]
        assert [c["chunk_id"] for c in result] == expected_ids

        # Verify each reordered chunk has all original fields
        reordered_chunks = [chunks[0], chunks[2], chunks[4], chunks[3], chunks[1]]
        for original, reordered in zip(reordered_chunks, result):
            for key in original:
                assert reordered[key] == original[key]


# ============================================================================
# Custom pattern tests
# ============================================================================

class TestCylinderReorderCustomPattern:
    """Test cylinder_reorder with custom patterns."""

    def test_custom_pattern_identity(self):
        result = cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(0, 1, 2, 3, 4))
        assert [c["chunk_id"] for c in result] == ["rank1", "rank2", "rank3", "rank4", "rank5"]

    def test_custom_pattern_reverse(self):
        result = cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(4, 3, 2, 1, 0))
        assert [c["chunk_id"] for c in result] == ["rank5", "rank4", "rank3", "rank2", "rank1"]

    def test_custom_pattern_interleave(self):
        result = cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(2, 3, 0, 1, 4))
        assert [c["chunk_id"] for c in result] == ["rank3", "rank4", "rank1", "rank2", "rank5"]

    def test_invalid_pattern_length(self):
        with pytest.raises(ValueError, match="Pattern must have 5 indices"):
            cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(0, 1, 2, 3))

    def test_invalid_pattern_duplicate_indices(self):
        with pytest.raises(ValueError, match="Pattern must be a permutation"):
            cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(0, 1, 2, 0, 3))

    def test_invalid_pattern_missing_indices(self):
        with pytest.raises(ValueError, match="Pattern must be a permutation"):
            cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=(0, 1, 2, 3, 5))


# ============================================================================
# Input validation
# ============================================================================

class TestCylinderReorderInputValidation:
    """Test cylinder_reorder input validation and dynamic sizing."""

    def test_four_chunks_accepted_dynamically(self):
        chunks = [{"chunk_id": f"c{i}"} for i in range(4)]
        result = cylinder_reorder(chunks)
        # 1-based [1,3,4,2] -> 0-based (0,2,3,1)
        assert [c["chunk_id"] for c in result] == ["c0", "c2", "c3", "c1"]

    def test_empty_input_returns_empty_list(self):
        assert cylinder_reorder([]) == []

    def test_six_chunks_accepted_dynamically(self):
        chunks = [{"chunk_id": f"c{i}"} for i in range(6)]
        result = cylinder_reorder(chunks)
        # 1-based [1,3,5,6,4,2] -> 0-based (0,2,4,5,3,1)
        assert [c["chunk_id"] for c in result] == ["c0", "c2", "c4", "c5", "c3", "c1"]

    def test_none_input(self):
        with pytest.raises(TypeError):
            cylinder_reorder(None)

    def test_single_chunk_passthrough(self):
        single = [{"chunk_id": "only", "text": "solo"}]
        assert cylinder_reorder(single) == single

    def test_duplicate_ids_in_input_preserved(self):
        chunks = [{"chunk_id": "dup"}, {"chunk_id": "dup"}]
        result = cylinder_reorder(chunks)
        assert [c["chunk_id"] for c in result] == ["dup", "dup"]


# ============================================================================
# 8-chunk cylinder layout (production post-retrieval contract)
# ============================================================================

class TestCylinderReorderEightChunks:
    """The dynamic 8-chunk cylinder: best at head, 2nd-best at tail."""

    def test_eight_chunk_head_is_best(self):
        result = cylinder_reorder(_SAMPLE_EIGHT_CHUNKS)
        assert result[0]["chunk_id"] == "rank1"

    def test_eight_chunk_tail_is_second_best(self):
        result = cylinder_reorder(_SAMPLE_EIGHT_CHUNKS)
        assert result[-1]["chunk_id"] == "rank2"

    def test_eight_chunk_full_order(self):
        result = cylinder_reorder(_SAMPLE_EIGHT_CHUNKS)
        # 1-based Rank 1,3,5,7,8,6,4,2 -> 0-based (0,2,4,6,7,5,3,1)
        assert [c["chunk_id"] for c in result] == [
            "rank1", "rank3", "rank5", "rank7", "rank8", "rank6", "rank4", "rank2",
        ]

    def test_eight_chunk_permutation_is_complete(self):
        result = cylinder_reorder(_SAMPLE_EIGHT_CHUNKS)
        assert sorted(c["chunk_id"] for c in result) == sorted(
            c["chunk_id"] for c in _SAMPLE_EIGHT_CHUNKS
        )
        assert len({c["chunk_id"] for c in result}) == 8


# ============================================================================
# CylinderReorderer class
# ============================================================================

class TestCylinderReordererClass:
    """Test CylinderReorderer class."""

    def test_constructor_default_pattern(self):
        reorderer = CylinderReorderer()
        result = reorderer(_SAMPLE_FIVE_CHUNKS)
        assert [c["chunk_id"] for c in result] == ["rank1", "rank3", "rank5", "rank4", "rank2"]

    def test_constructor_custom_pattern(self):
        reorderer = CylinderReorderer(pattern=(0, 1, 2, 3, 4))
        result = reorderer(_SAMPLE_FIVE_CHUNKS)
        assert [c["chunk_id"] for c in result] == ["rank1", "rank2", "rank3", "rank4", "rank5"]

    def test_callable(self):
        reorderer = CylinderReorderer()
        result = reorderer(_SAMPLE_FIVE_CHUNKS)
        assert len(result) == 5

    def test_reorder_method_alias(self):
        reorderer = CylinderReorderer()
        result = reorderer.reorder(_SAMPLE_FIVE_CHUNKS)
        assert [c["chunk_id"] for c in result] == ["rank1", "rank3", "rank5", "rank4", "rank2"]

    def test_reorderer_preserves_all_fields(self):
        chunks = [
            {
                "chunk_id": f"c{i}",
                "text": f"text{i}",
                "metadata": {"id": i},
                "score": float(i),
            }
            for i in range(5)
        ]
        reorderer = CylinderReorderer()
        result = reorderer(chunks)

        # Expected order by pattern (0,2,4,3,1)
        expected_ids = ["c0", "c2", "c4", "c3", "c1"]
        assert [c["chunk_id"] for c in result] == expected_ids

        reordered_chunks = [chunks[0], chunks[2], chunks[4], chunks[3], chunks[1]]
        for original, res in zip(reordered_chunks, result):
            assert res["chunk_id"] == original["chunk_id"]
            assert res["text"] == original["text"]
            assert res["metadata"] == original["metadata"]
            assert res["score"] == original["score"]


# ============================================================================
# Backwards compatibility
# ============================================================================

class TestBackwardsCompatibility:
    """Test backwards compatibility aliases."""

    def test_reorder_cylinder_alias(self):
        result = reorder_cylinder(_SAMPLE_FIVE_CHUNKS)
        expected = cylinder_reorder(_SAMPLE_FIVE_CHUNKS)
        assert [c["chunk_id"] for c in result] == [c["chunk_id"] for c in expected]

    def test_reorder_cylinder_custom_pattern(self):
        custom = (1, 0, 2, 3, 4)
        result = reorder_cylinder(_SAMPLE_FIVE_CHUNKS, pattern=custom)
        expected = cylinder_reorder(_SAMPLE_FIVE_CHUNKS, pattern=custom)
        assert [c["chunk_id"] for c in result] == [c["chunk_id"] for c in expected]

    def test_import_reorder_cylinder(self):
        from cylinder_reorder import reorder_cylinder as rc
        assert callable(rc)


# ============================================================================
# Integration tests
# ============================================================================

class TestCylinderReorderIntegration:
    """Integration tests with various patterns."""

    def test_pattern_application_matrix(self):
        chunks = [{"chunk_id": f"r{i}", "score": float(i)} for i in range(5)]

        cases = [
            ((0, 1, 2, 3, 4), ["r0", "r1", "r2", "r3", "r4"]),
            ((4, 3, 2, 1, 0), ["r4", "r3", "r2", "r1", "r0"]),
            ((0, 2, 4, 3, 1), ["r0", "r2", "r4", "r3", "r1"]),
        ]
        for pattern, expected in cases:
            result = cylinder_reorder(chunks, pattern=pattern)
            assert [c["chunk_id"] for c in result] == expected

    def test_realistic_financial_scores(self):
        chunks = [
            {"chunk_id": "q4_earnings", "text": "Q4 earnings beat", "rerank_score": 0.98},
            {"chunk_id": "dividend_increase", "text": "Dividend increased 10%", "rerank_score": 0.87},
            {"chunk_id": "analyst_upgrade", "text": "Analyst upgrade to buy", "rerank_score": 0.76},
            {"chunk_id": "guidance_raise", "text": "2025 guidance raised", "rerank_score": 0.65},
            {"chunk_id": "price_target", "text": "Price target $200", "rerank_score": 0.54},
        ]
        result = cylinder_reorder(chunks)
        expected = ["q4_earnings", "analyst_upgrade", "price_target", "guidance_raise", "dividend_increase"]
        assert [c["chunk_id"] for c in result] == expected


# ============================================================================
# Edge cases
# ============================================================================

class TestCylinderReorderEdgeCases:
    """Edge cases and error conditions."""

    def test_chunks_without_score_field(self):
        chunks_no_score = [
            {"chunk_id": "a", "text": "a"},
            {"chunk_id": "b", "text": "b"},
            {"chunk_id": "c", "text": "c"},
            {"chunk_id": "d", "text": "d"},
            {"chunk_id": "e", "text": "e"},
        ]
        result = cylinder_reorder(chunks_no_score)
        assert len(result) == 5
        assert [c["chunk_id"] for c in result] == ["a", "c", "e", "d", "b"]

    def test_chunks_with_null_values(self):
        chunks_with_none = [
            {"chunk_id": "a", "text": None, "metadata": None},
            {"chunk_id": "b", "text": None, "metadata": {}},
            {"chunk_id": "c", "text": "valid", "metadata": None},
            {"chunk_id": "d", "text": "valid", "metadata": {}},
            {"chunk_id": "e", "text": None, "metadata": {"key": "value"}},
        ]
        result = cylinder_reorder(chunks_with_none)
        assert [c["chunk_id"] for c in result] == ["a", "c", "e", "d", "b"]

    def test_pattern_indices_correct(self):
        """Verify each position in the cylinder pattern is applied correctly."""
        chunks = [{"chunk_id": f"pos{i}", "value": i} for i in range(5)]
        result = cylinder_reorder(chunks)

        expected_ids = ["pos0", "pos2", "pos4", "pos3", "pos1"]
        assert [c["chunk_id"] for c in result] == expected_ids
        expected_values = [0, 2, 4, 3, 1]
        assert [c["value"] for c in result] == expected_values