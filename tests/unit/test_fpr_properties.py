"""Property-based tests for false positive rate evaluator using Hypothesis.

Tests invariants about false positive rate:
- FPR is always in [0, 1]
- FPR is 0 when no false positives
- FPR is 0 when nothing retrieved
- FPR approaches 1 as only FPs retrieved
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmark.evaluation.false_positive import FalsePositiveEvaluator


@pytest.mark.unit
class TestFalsePositiveRateProperties:
    """Property-based tests for false positive rate evaluator."""

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        retrieved_count=st.integers(min_value=0, max_value=100),
    )
    def test_fpr_always_in_valid_range(
        self, gold_count: int, retrieved_count: int
    ) -> None:
        """FPR must always be in [0, 1]."""
        evaluator = FalsePositiveEvaluator()
        expected_ids = [f"M{i}" for i in range(gold_count)]
        # Worst case: all retrieved are FPs
        retrieved_ids = [f"N{i}" for i in range(retrieved_count)]

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        assert 0.0 <= result.value <= 1.0

    def test_fpr_empty_retrieved(self) -> None:
        """FPR should be 0 when nothing is retrieved."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate([], ["M1", "M2"])
        assert result.value == 0.0

    @given(count=st.integers(min_value=1, max_value=50))
    def test_fpr_no_false_positives(self, count: int) -> None:
        """FPR should be 0 when all retrieved are in gold."""
        evaluator = FalsePositiveEvaluator()
        ids = [f"M{i}" for i in range(count)]
        result = evaluator.evaluate(ids, ids)
        assert result.value == 0.0

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        fp_count=st.integers(min_value=1, max_value=5_000),
    )
    def test_fpr_all_false_positives(self, gold_count: int, fp_count: int) -> None:
        """FPR should be 1.0 when all retrieved are FPs."""
        evaluator = FalsePositiveEvaluator()
        expected_ids = [f"M{i}" for i in range(gold_count)]
        retrieved_ids = [f"N{i}" for i in range(fp_count)]

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        assert result.value == 1.0

    @given(
        gold_count=st.integers(min_value=1, max_value=30),
        tp_count=st.integers(min_value=1, max_value=1_000),
        fp_count=st.integers(min_value=1, max_value=5_000),
    )
    def test_fpr_partial_false_positives(
        self, gold_count: int, tp_count: int, fp_count: int
    ) -> None:
        """FPR should be fp_count / (tp_count + fp_count)."""
        tp_count = min(tp_count, gold_count)
        evaluator = FalsePositiveEvaluator()

        expected_ids = [f"M{i}" for i in range(gold_count)]
        retrieved_ids = (
            [f"M{i}" for i in range(tp_count)]  # True positives
            + [f"N{i}" for i in range(fp_count)]  # False positives
        )

        result = evaluator.evaluate(retrieved_ids, expected_ids)
        expected_fpr = fp_count / (tp_count + fp_count)
        assert abs(result.value - expected_fpr) < 1e-9

    @given(
        gold_count=st.integers(min_value=1, max_value=50),
        retrieved_count=st.integers(min_value=1, max_value=100),
    )
    def test_fpr_monotonic_with_fp(self, gold_count: int, retrieved_count: int) -> None:
        """FPR increases as number of FPs increases."""
        evaluator = FalsePositiveEvaluator()
        expected_ids = [f"M{i}" for i in range(gold_count)]

        # Scenario 1: few FPs
        retrieved_few_fp = [f"N{i}" for i in range(min(1, retrieved_count))]
        result_few = evaluator.evaluate(retrieved_few_fp, expected_ids)

        # Scenario 2: more FPs
        retrieved_many_fp = [f"N{i}" for i in range(min(max(1, retrieved_count), 50))]
        result_many = evaluator.evaluate(retrieved_many_fp, expected_ids)

        # More FPs should have higher FPR
        assert result_many.value >= result_few.value - 1e-9
