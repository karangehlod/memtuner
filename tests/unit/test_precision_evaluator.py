"""Unit tests for StandardPrecisionEvaluator.

Verifies the standard IR Precision@K formula:
    Precision@K = |relevant ∩ retrieved[:K]| / K
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.precision import StandardPrecisionEvaluator


@pytest.mark.unit
class TestStandardPrecisionEvaluator:
    """Tests for the standard IR Precision@K metric."""

    def test_perfect_retrieval(self) -> None:
        """All gold items in top K → precision = gold_size / K."""
        evaluator = StandardPrecisionEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002", "noise1", "noise2", "noise3"],
            expected_ids=["M-001", "M-002"],
        )
        # 2 relevant out of K=5
        assert result.value == 2 / 5
        assert result.details["relevant_in_top_k"] == 2
        assert result.details["k"] == 5

    def test_all_results_relevant(self) -> None:
        """When all K results are relevant, precision = 1.0."""
        evaluator = StandardPrecisionEvaluator(top_k=3)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002", "M-003"],
            expected_ids=["M-001", "M-002", "M-003", "M-004"],
        )
        assert result.value == 1.0

    def test_no_relevant_results(self) -> None:
        """When no retrieved items are relevant, precision = 0.0."""
        evaluator = StandardPrecisionEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["noise1", "noise2", "noise3", "noise4", "noise5"],
            expected_ids=["M-001", "M-002"],
        )
        assert result.value == 0.0

    def test_empty_retrieval_returns_zero(self) -> None:
        """Empty retrieved set returns precision = 0.0."""
        evaluator = StandardPrecisionEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=[],
            expected_ids=["M-001"],
        )
        assert result.value == 0.0

    def test_partial_match(self) -> None:
        """Partial overlap gives fractional precision."""
        evaluator = StandardPrecisionEvaluator(top_k=4)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "noise1", "M-002", "noise2"],
            expected_ids=["M-001", "M-002", "M-003"],
        )
        # 2 relevant out of K=4
        assert result.value == 0.5

    def test_gold_larger_than_k_caps_max_achievable(self) -> None:
        """When gold set > K, max achievable is K/K = 1.0 (or min(gold, K)/K)."""
        evaluator = StandardPrecisionEvaluator(top_k=3)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002", "M-003"],
            expected_ids=["M-001", "M-002", "M-003", "M-004", "M-005"],
        )
        # All 3 retrieved are relevant, max_achievable = min(5, 3)/3 = 1.0
        assert result.value == 1.0
        assert result.details["max_achievable"] == 1.0

    def test_max_achievable_when_gold_smaller_than_k(self) -> None:
        """When gold set < K, max achievable precision is gold_size / K."""
        evaluator = StandardPrecisionEvaluator(top_k=10)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002"] + [f"n{i}" for i in range(8)],
            expected_ids=["M-001", "M-002", "M-003"],
        )
        # max_achievable = min(3, 10) / 10 = 0.3
        assert result.details["max_achievable"] == 0.3

    def test_metric_name(self) -> None:
        """Metric name is the standard precision identifier."""
        evaluator = StandardPrecisionEvaluator(top_k=5)
        assert evaluator.metric_name() == "benchmark.precision_at_k"

    def test_only_top_k_considered(self) -> None:
        """Results beyond position K are ignored."""
        evaluator = StandardPrecisionEvaluator(top_k=2)
        result = evaluator.evaluate(
            retrieved_ids=["noise1", "noise2", "M-001", "M-002"],
            expected_ids=["M-001", "M-002"],
        )
        # M-001 and M-002 are at positions 3-4, beyond K=2
        assert result.value == 0.0

    def test_independent_of_contamination(self) -> None:
        """Precision is computed directly, not derived from contamination."""
        from benchmark.evaluation.false_positive import FalsePositiveEvaluator

        precision_eval = StandardPrecisionEvaluator(top_k=5)
        contamination_eval = FalsePositiveEvaluator()

        retrieved = ["M-001", "noise1", "noise2", "noise3", "noise4"]
        expected = ["M-001"]

        prec_result = precision_eval.evaluate(retrieved, expected)
        cont_result = contamination_eval.evaluate(retrieved, expected)

        # Precision = 1/5 = 0.2
        assert prec_result.value == 0.2
        # Contamination = 4/5 = 0.8
        assert cont_result.value == 0.8
        # They are NOT complements (1 - 0.8 = 0.2 happens here but the formulas differ)
        # The key point: they are computed independently
        assert prec_result.metric_name != cont_result.metric_name
