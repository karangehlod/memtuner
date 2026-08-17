"""Contract tests for conversation-aware evaluators.

Ensures FollowUpAccuracyEvaluator and ContradictionResolutionEvaluator
comply with the MetricEvaluator interface contract.
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.contradiction_resolution import ContradictionResolutionEvaluator
from benchmark.evaluation.followup_accuracy import FollowUpAccuracyEvaluator


@pytest.mark.contract
class TestFollowUpAccuracyEvaluatorContract:
    """Contract tests for FollowUpAccuracyEvaluator."""

    def test_implements_metric_evaluator_interface(self) -> None:
        """Test that FollowUpAccuracyEvaluator implements MetricEvaluator."""
        evaluator = FollowUpAccuracyEvaluator()
        assert isinstance(evaluator, MetricEvaluator)

    def test_has_evaluate_method(self) -> None:
        """Test that evaluate method exists."""
        evaluator = FollowUpAccuracyEvaluator()
        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)

    def test_has_metric_name_method(self) -> None:
        """Test that metric_name method exists."""
        evaluator = FollowUpAccuracyEvaluator()
        assert hasattr(evaluator, "metric_name")
        assert callable(evaluator.metric_name)

    def test_metric_name_returns_string(self) -> None:
        """Test that metric_name returns a non-empty string."""
        evaluator = FollowUpAccuracyEvaluator()
        name = evaluator.metric_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_evaluate_returns_evaluation_result(self) -> None:
        """Test that evaluate returns EvaluationResult."""
        evaluator = FollowUpAccuracyEvaluator()
        result = evaluator.evaluate([], [])
        assert isinstance(result, EvaluationResult)

    def test_evaluation_result_has_required_fields(self) -> None:
        """Test that EvaluationResult has all required fields."""
        evaluator = FollowUpAccuracyEvaluator()
        result = evaluator.evaluate([], [])
        assert hasattr(result, "metric_name")
        assert hasattr(result, "value")
        assert hasattr(result, "query_count")

    def test_evaluation_result_value_is_float(self) -> None:
        """Test that result value is a float."""
        evaluator = FollowUpAccuracyEvaluator()
        result = evaluator.evaluate(["M1"], ["M1"])
        assert isinstance(result.value, float)

    def test_evaluate_with_context_returns_evaluation_result(self) -> None:
        """Test that evaluate_with_context returns EvaluationResult."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(retrieved_ids=[], expected_ids=[])
        result = evaluator.evaluate_with_context(context)
        assert isinstance(result, EvaluationResult)

    def test_metric_name_consistent(self) -> None:
        """Test that metric_name is consistent across calls."""
        evaluator = FollowUpAccuracyEvaluator()
        name1 = evaluator.metric_name()
        name2 = evaluator.metric_name()
        assert name1 == name2


@pytest.mark.contract
class TestContradictionResolutionEvaluatorContract:
    """Contract tests for ContradictionResolutionEvaluator."""

    def test_implements_metric_evaluator_interface(self) -> None:
        """Test that ContradictionResolutionEvaluator implements MetricEvaluator."""
        evaluator = ContradictionResolutionEvaluator()
        assert isinstance(evaluator, MetricEvaluator)

    def test_has_evaluate_method(self) -> None:
        """Test that evaluate method exists."""
        evaluator = ContradictionResolutionEvaluator()
        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)

    def test_has_metric_name_method(self) -> None:
        """Test that metric_name method exists."""
        evaluator = ContradictionResolutionEvaluator()
        assert hasattr(evaluator, "metric_name")
        assert callable(evaluator.metric_name)

    def test_metric_name_returns_string(self) -> None:
        """Test that metric_name returns a non-empty string."""
        evaluator = ContradictionResolutionEvaluator()
        name = evaluator.metric_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_evaluate_returns_evaluation_result(self) -> None:
        """Test that evaluate returns EvaluationResult."""
        evaluator = ContradictionResolutionEvaluator()
        result = evaluator.evaluate([], [])
        assert isinstance(result, EvaluationResult)

    def test_evaluation_result_has_required_fields(self) -> None:
        """Test that EvaluationResult has all required fields."""
        evaluator = ContradictionResolutionEvaluator()
        result = evaluator.evaluate([], [])
        assert hasattr(result, "metric_name")
        assert hasattr(result, "value")
        assert hasattr(result, "query_count")

    def test_evaluation_result_value_is_float(self) -> None:
        """Test that result value is a float."""
        evaluator = ContradictionResolutionEvaluator()
        result = evaluator.evaluate(["M1"], ["M1"])
        assert isinstance(result.value, float)

    def test_evaluate_with_context_returns_evaluation_result(self) -> None:
        """Test that evaluate_with_context returns EvaluationResult."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(retrieved_ids=[], expected_ids=[])
        result = evaluator.evaluate_with_context(context)
        assert isinstance(result, EvaluationResult)

    def test_metric_name_consistent(self) -> None:
        """Test that metric_name is consistent across calls."""
        evaluator = ContradictionResolutionEvaluator()
        name1 = evaluator.metric_name()
        name2 = evaluator.metric_name()
        assert name1 == name2

    def test_different_metric_names(self) -> None:
        """Test that the two evaluators have different metric names."""
        followup = FollowUpAccuracyEvaluator()
        contradiction = ContradictionResolutionEvaluator()
        assert followup.metric_name() != contradiction.metric_name()
