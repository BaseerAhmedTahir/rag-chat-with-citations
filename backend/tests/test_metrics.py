"""Tests for the hand-rolled retrieval metrics."""

from __future__ import annotations

import pytest

from evals.metrics.retrieval import (
    hit_at_k,
    mean_metric,
    recall_at_k,
    reciprocal_rank,
)

A1 = ("a.pdf", 1)
A2 = ("a.pdf", 2)
B1 = ("b.pdf", 1)
B2 = ("b.pdf", 2)

RANKED = [A1, B1, A2, B2]


def test_hit_at_k_boundaries():
    assert hit_at_k(RANKED, {A2}, k=3) == 1.0
    assert hit_at_k(RANKED, {A2}, k=2) == 0.0
    assert hit_at_k(RANKED, {("c.pdf", 9)}, k=4) == 0.0


def test_recall_at_k_single_relevant_is_hit_rate():
    assert recall_at_k(RANKED, {B1}, k=1) == 0.0
    assert recall_at_k(RANKED, {B1}, k=2) == 1.0


def test_recall_at_k_multiple_relevant_is_fractional():
    relevant = {A1, B2}
    assert recall_at_k(RANKED, relevant, k=2) == 0.5
    assert recall_at_k(RANKED, relevant, k=4) == 1.0


def test_reciprocal_rank_positions():
    assert reciprocal_rank(RANKED, {A1}) == 1.0
    assert reciprocal_rank(RANKED, {B1}) == 0.5
    assert reciprocal_rank(RANKED, {B2}) == 0.25
    assert reciprocal_rank(RANKED, {("c.pdf", 9)}) == 0.0


def test_reciprocal_rank_uses_first_relevant():
    assert reciprocal_rank(RANKED, {B1, B2}) == 0.5


def test_mean_metric():
    assert mean_metric([1.0, 0.0, 0.5]) == 0.5
    with pytest.raises(ValueError):
        mean_metric([])


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        hit_at_k(RANKED, {A1}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, {A1}, k=-1)


def test_empty_relevant_set_raises():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), k=3)
