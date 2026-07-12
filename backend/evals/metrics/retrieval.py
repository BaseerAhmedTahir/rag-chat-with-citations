"""Deterministic retrieval metrics, implemented from scratch.

A retrieved chunk counts as relevant when its (source_file, page_number)
pair matches a labeled relevant source. Page-level matching is the right
granularity here: it is exactly what a citation exposes to the user.
"""

from __future__ import annotations

from collections.abc import Sequence

SourcePage = tuple[str, int]  # (source_file, page_number)


def hit_at_k(
    ranked: Sequence[SourcePage], relevant: set[SourcePage], k: int
) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def recall_at_k(
    ranked: Sequence[SourcePage], relevant: set[SourcePage], k: int
) -> float:
    """Fraction of relevant sources that appear in the top-k results."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant set must not be empty")
    found = {item for item in ranked[:k] if item in relevant}
    return len(found) / len(relevant)


def reciprocal_rank(ranked: Sequence[SourcePage], relevant: set[SourcePage]) -> float:
    """1/rank of the first relevant result, 0.0 if none is retrieved."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_metric(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)
