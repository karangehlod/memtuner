"""Property-based tests for recall evaluator using Hypothesis.

Tests invariants about recall:
- Recall is always in [0, 1]
- Recall never exceeds 1.0
- Recall is 1.0 when all gold retrieved
- Recall is 0.0 when no gold retrieved
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from benchmark.evaluation.recall import RecallEvaluator


@pytest.mark.unit
class TestRecallEvaluatorProperties:
    """Property-based tests for recall evaluator."""

    @given(
        gold_count=st.integers(min_value=1, max_value=100),
        retrieved_count=st.integers(min_value=0, max_value=100),
        overlap=st.integers(min_value=0),
    )
    def test_recall_always_in_valid_range(
        self, gold_count: int, retrieved_count: int, overlap: int
    ) -> None:
        """Recall must always be in [0, 1]."""
        # Clamp overlap to valid range
        overlap = min(overlap, gold_count, retrieved_count)

        evaluator = RecallEvaluator(top_k=100)
        result = evaluator.evaluate(
            retrieved_ids=[f"M{i}" for i in range(retrieved_count)],
            expected_ids=[f"M{i}" for i in range(gold_count)],
        )

        assert 0.0 <= result.value <= 1.0

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        overlap_count=st.integers(min_value=0),
    )
    def test_recall_never_exceeds_one(
        self, gold_count: int, overlap_count: int
    ) -> None:
        """Recall must never exceed 1.0."""
        overlap_count = min(overlap_count, gold_count)

        evaluator = RecallEvaluator(top_k=100)

        # Simulate overlap
        expected_ids = [f"M{i}" for i in range(gold_count)]
        retrieved_ids = [f"M{i}" for i in range(overlap_count)]

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        assert result.value <= 1.0

    @given(count=st.integers(min_value=1, max_value=50))
    def test_recall_perfect_match(self, count: int) -> None:
        """Recall is 1.0 when all gold memories retrieved."""
        evaluator = RecallEvaluator(top_k=100)
        ids = [f"M{i}" for i in range(count)]

        result = evaluator.evaluate(ids, ids)
        assert result.value == 1.0

    @given(gold_count=st.integers(min_value=1, max_value=50))
    def test_recall_no_overlap(self, gold_count: int) -> None:
        """Recall is 0.0 when no gold memories retrieved."""
        evaluator = RecallEvaluator(top_k=100)
        expected_ids = [f"M{i}" for i in range(gold_count)]
        retrieved_ids = [f"N{i}" for i in range(gold_count)]  # Different IDs

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        assert result.value == 0.0

    def test_recall_empty_gold(self) -> None:
        """Recall with empty gold set raises ValueError (fail-fast, no vacuous truth)."""
        evaluator = RecallEvaluator(top_k=100)
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            evaluator.evaluate(["M1", "M2"], [])

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        k=st.integers(min_value=1, max_value=100),
    )
    def test_recall_respects_k(self, gold_count: int, k: int) -> None:
        """Recall@K should only consider top-K results."""
        evaluator = RecallEvaluator(top_k=k)

        expected_ids = [f"M{i}" for i in range(gold_count)]
        # Only put overlap in first k positions
        retrieved_ids = [f"M{i}" for i in range(min(k, gold_count))]
        # Add more after k positions
        retrieved_ids.extend([f"N{i}" for i in range(max(0, gold_count - k))])

        result = evaluator.evaluate(retrieved_ids, expected_ids)

        # Recall should be min(k, gold_count) / gold_count
        expected_recall = min(k, gold_count) / gold_count
        assert abs(result.value - expected_recall) < 1e-9

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        overlap_count=st.integers(min_value=1),
    )
    def test_recall_partial_match(self, gold_count: int, overlap_count: int) -> None:
        """Recall should be overlap / gold_count."""
        overlap_count = min(overlap_count, gold_count)
        evaluator = RecallEvaluator(top_k=100)

        expected_ids = [f"M{i}" for i in range(gold_count)]
        retrieved_ids = (
            [f"M{i}" for i in range(overlap_count)]  # Matching
            + [f"N{i}" for i in range(gold_count)]  # Non-matching
        )

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        expected_recall = overlap_count / gold_count
        assert abs(result.value - expected_recall) < 1e-9
