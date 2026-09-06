"""
Module 4 -- Retrieval Stage: Cylinder Re-ordering (Post-Retrieval Part 3).

Resolves the "Lost in the Middle" attention issue by reordering cleaned chunks
in a staggered "cylinder" sequence that optimises LLM context utilisation.

Dynamic pattern (built for any N >= 2):
    Position 1   <- highest-ranked chunk (head attention sweet spot)
    Position N   <- second-highest-ranked chunk (tail attention sweet spot)
    Positions 2..N-1 <- the remaining ranks staggered: ascending odd ranks
                        (1-based 3, 5, 7, ...) then descending even ranks
                        (1-based N, N-2, ..., 4), so mid-relevance supporting
                        chunks are sandwiched between the two extremes.

For N=5 the generated 0-based pattern is (0, 2, 4, 3, 1) -- Rank 1, 3, 5, 4, 2
-- identical to the original fixed pattern, so the algorithm is a strict
generalisation of the legacy behaviour.
"""

from config.logging_config import get_logger

logger = get_logger("retrieval.cylinder_reorder")

# Legacy 5-chunk pattern kept for backward compatibility / explicit callers.
# Rank 1, 3, 5, 4, 2 (0-based indices 0, 2, 4, 3, 1).
_CYLINDER_PATTERN = (0, 2, 4, 3, 1)


def _build_cylinder_pattern(n: int) -> tuple[int, ...]:
    """
    Generate the 0-based cylinder reorder pattern for ``n`` chunks.

    Layout: best (index 0) first, second-best (index 1) last, then the odd
    0-based ranks ascending followed by the remaining even 0-based ranks
    descending in between.

    Examples (0-based):
        n=2 -> (0, 1)
        n=4 -> (0, 2, 3, 1)
        n=5 -> (0, 2, 4, 3, 1)
        n=8 -> (0, 2, 4, 6, 7, 5, 3, 1)   (Rank 1, 3, 5, 7, 8, 6, 4, 2)
    """
    if n == 1:
        return (0,)
    pattern = [0]                       # head = highest-ranked
    pattern += list(range(2, n, 2))     # ascending even 0-based (odd 1-based) ranks
    largest_odd = n - 1 if (n - 1) % 2 == 1 else n - 2
    pattern += list(range(largest_odd, 1, -2))  # descending odd 0-based ranks
    pattern.append(1)                   # tail = second-highest-ranked
    return tuple(pattern)


def cylinder_reorder(chunks: list[dict], pattern: tuple[int, ...] | None = None) -> list[dict]:
    """
    Reorder chunks using the dynamic cylinder (staggered) pattern.

    Args:
        chunks: List of chunk dicts sorted by rank (1=best, N=worst).
                Each dict should have rerank_score or similar ranking field.
        pattern: Optional custom reorder pattern as 0-based indices. When
                 omitted the pattern is generated dynamically for the given
                 chunk count (best at head, second-best at tail, remainder
                 staggered in between).

    Returns:
        List of chunks reordered per the cylinder pattern.

    Raises:
        ValueError: If an explicit pattern length does not match the chunk
            count or is not a permutation of ``range(len(chunks))``.
    """
    if chunks is None:
        raise TypeError("cylinder_reorder expects a list of chunks, got None")
    if not chunks:
        logger.warning("cylinder_reorder called with an empty chunk list")
        return []

    n = len(chunks)

    if pattern is None:
        pattern = _build_cylinder_pattern(n)
    else:
        if len(pattern) != n:
            raise ValueError(
                f"Pattern must have {n} indices (got {len(pattern)}) "
                f"for {n} chunks"
            )
        if set(pattern) != set(range(n)):
            raise ValueError(
                f"Pattern must be a permutation of indices 0-{n - 1} "
                f"(got {pattern})"
            )

    reordered = [chunks[i] for i in pattern]

    # Log the reordering for observability
    rank_scores = [c.get("rerank_score", c.get("rrf_score", 0.0)) for c in chunks]
    new_scores = [c.get("rerank_score", c.get("rrf_score", 0.0)) for c in reordered]
    logger.info(
        "Cylinder reorder applied (n=%d): original_scores=%s -> reordered_scores=%s",
        n,
        [round(s, 4) for s in rank_scores],
        [round(s, 4) for s in new_scores],
    )

    return reordered


class CylinderReorderer:
    """
    Callable cylinder re-orderer with configurable pattern.

    Useful when the same pattern is applied repeatedly in a pipeline. Passing
    ``pattern=None`` selects the dynamic per-length cylinder layout.
    """

    def __init__(self, pattern: tuple[int, ...] | None = None):
        self._pattern = pattern
        logger.info("CylinderReorderer initialised with pattern: %s", pattern)

    def __call__(self, chunks: list[dict]) -> list[dict]:
        return cylinder_reorder(chunks, self._pattern)

    def reorder(self, chunks: list[dict]) -> list[dict]:
        """Explicit method alias for __call__."""
        return self(chunks)


# Backwards-compatible functional alias
reorder_cylinder = cylinder_reorder
