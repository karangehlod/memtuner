"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest

from benchmark.evaluation.aggregator import MetricAggregator
from benchmark.evaluation.base import EvaluationResult
from benchmark.evaluation.false_positive import FalsePositiveEvaluator
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator


@pytest.mark.unit
class TestRecallEvaluator:
    """Tests for Recall@K metric."""

    def test_perfect_recall(self) -> None:
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002", "M-003"],
            expected_ids=["M-001", "M-002"],
        )
        assert result.value == 1.0
        assert result.metric_name == "benchmark.recall_at_k"

    def test_zero_recall(self) -> None:
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["M-099", "M-098"],
            expected_ids=["M-001", "M-002"],
        )
        assert result.value == 0.0

    def test_partial_recall(self) -> None:
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-099"],
            expected_ids=["M-001", "M-002"],
        )
        assert result.value == 0.5

    def test_empty_gold_set_raises(self) -> None:
        evaluator = RecallEvaluator(top_k=5)
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            evaluator.evaluate(retrieved_ids=["M-001"], expected_ids=[])

    def test_empty_retrieved_returns_zero(self) -> None:
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(retrieved_ids=[], expected_ids=["M-001"])
        assert result.value == 0.0

    def test_top_k_limits_results(self) -> None:
        evaluator = RecallEvaluator(top_k=2)
        result = evaluator.evaluate(
            retrieved_ids=["M-099", "M-098", "M-001"],
            expected_ids=["M-001"],
        )
        assert result.value == 0.0

    def test_metric_name(self) -> None:
        evaluator = RecallEvaluator()
        assert evaluator.metric_name() == "benchmark.recall_at_k"


@pytest.mark.unit
class TestFalsePositiveEvaluator:
    """Tests for False Positive Rate metric."""

    def test_no_false_positives(self) -> None:
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001", "M-002", "M-003"],
        )
        assert result.value == 0.0

    def test_all_false_positives(self) -> None:
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(
            retrieved_ids=["M-099", "M-098"],
            expected_ids=["M-001"],
        )
        assert result.value == 1.0

    def test_partial_false_positives(self) -> None:
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(
            retrieved_ids=["M-001", "M-099"],
            expected_ids=["M-001"],
        )
        assert result.value == 0.5

    def test_empty_retrieved_returns_zero(self) -> None:
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(retrieved_ids=[], expected_ids=["M-001"])
        assert result.value == 0.0

    def test_metric_name(self) -> None:
        evaluator = FalsePositiveEvaluator()
        assert evaluator.metric_name() == "benchmark.contamination_rate"


@pytest.mark.unit
class TestTemporalAccuracyEvaluator:
    """Tests for Temporal Accuracy metric."""

    def test_all_within_window(self) -> None:
        evaluator = TemporalAccuracyEvaluator(tolerance_days=1)
        result = evaluator.evaluate_temporal(
            retrieved_days=[1, 2, 3],
            expected_day_range=(1, 3),
        )
        assert result.value == 1.0

    def test_none_within_window(self) -> None:
        evaluator = TemporalAccuracyEvaluator(tolerance_days=0)
        result = evaluator.evaluate_temporal(
            retrieved_days=[10, 20],
            expected_day_range=(1, 3),
        )
        assert result.value == 0.0

    def test_partial_within_window(self) -> None:
        evaluator = TemporalAccuracyEvaluator(tolerance_days=0)
        result = evaluator.evaluate_temporal(
            retrieved_days=[2, 10],
            expected_day_range=(1, 3),
        )
        assert result.value == 0.5

    def test_empty_retrieved_returns_zero(self) -> None:
        """Empty retrieved: value=0.0, query_count=1 (counted as a miss)."""
        evaluator = TemporalAccuracyEvaluator()
        result = evaluator.evaluate_temporal(
            retrieved_days=[],
            expected_day_range=(1, 5),
        )
        assert result.value == 0.0
        assert result.query_count == 1

    def test_metric_name(self) -> None:
        evaluator = TemporalAccuracyEvaluator()
        assert evaluator.metric_name() == "benchmark.temporal_accuracy"


@pytest.mark.unit
class TestMetricAggregator:
    """Tests for MetricAggregator."""

    def test_aggregate_single_metric(self) -> None:
        aggregator = MetricAggregator()
        results = [
            EvaluationResult(metric_name="benchmark.recall_at_k", value=0.8, query_count=1),
            EvaluationResult(metric_name="benchmark.recall_at_k", value=0.6, query_count=1),
        ]
        aggregated = aggregator.aggregate_results(results)
        assert abs(aggregated["benchmark.recall_at_k"] - 0.7) < 1e-9

    def test_aggregate_multiple_metrics(self) -> None:
        aggregator = MetricAggregator()
        results = [
            EvaluationResult(metric_name="benchmark.recall_at_k", value=1.0, query_count=1),
            EvaluationResult(metric_name="benchmark.contamination_rate", value=0.2, query_count=1),
        ]
        aggregated = aggregator.aggregate_results(results)
        assert aggregated["benchmark.recall_at_k"] == 1.0
        assert aggregated["benchmark.contamination_rate"] == 0.2

    def test_aggregate_empty_returns_empty(self) -> None:
        aggregator = MetricAggregator()
        assert aggregator.aggregate_results([]) == {}

    def test_weighted_average(self) -> None:
        aggregator = MetricAggregator()
        results = [
            EvaluationResult(metric_name="benchmark.recall_at_k", value=1.0, query_count=3),
            EvaluationResult(metric_name="benchmark.recall_at_k", value=0.0, query_count=1),
        ]
        aggregated = aggregator.aggregate_results(results)
        assert abs(aggregated["benchmark.recall_at_k"] - 0.75) < 1e-9
