"""Unit tests for conversation-aware evaluators."""

from __future__ import annotations

import pytest

from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.contradiction_resolution import ContradictionResolutionEvaluator
from benchmark.evaluation.followup_accuracy import FollowUpAccuracyEvaluator


@pytest.mark.unit
class TestFollowUpAccuracyEvaluator:
    """Unit tests for FollowUpAccuracyEvaluator."""

    def test_non_followup_query_always_scores_1(self) -> None:
        """Non-follow-up queries should always score 1.0."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1"],
            is_followup=False,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 0

    def test_followup_with_expected_results_present(self) -> None:
        """Follow-up with expected results present should score 1.0."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2", "M3"],
            expected_ids=["M1", "M2"],
            is_followup=True,
            references_turn=1,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 1

    def test_followup_with_no_expected_results(self) -> None:
        """Follow-up with no expected results should score 0.0."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=[],
            is_followup=True,
            references_turn=1,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0
        assert result.query_count == 1

    def test_followup_with_partial_retrieval(self) -> None:
        """Follow-up with partial retrieval should score 1.0 (has relevant)."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1", "M2", "M3"],  # M3 not retrieved
            is_followup=True,
            references_turn=1,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 1
        assert result.details["matched_count"] == 2

    def test_followup_with_no_retrieval(self) -> None:
        """Follow-up with no retrieval should score 0.0."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=[],
            expected_ids=["M1", "M2"],
            is_followup=True,
            references_turn=1,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0
        assert result.query_count == 1

    def test_metric_name_correct(self) -> None:
        """Test the metric name is correct."""
        evaluator = FollowUpAccuracyEvaluator()
        assert evaluator.metric_name() == "benchmark.followup_accuracy"

    def test_deterministic_replay(self) -> None:
        """Test that same context produces same result."""
        evaluator = FollowUpAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1"],
            is_followup=True,
            references_turn=2,
        )
        result1 = evaluator.evaluate_with_context(context)
        result2 = evaluator.evaluate_with_context(context)
        assert result1.value == result2.value
        assert result1.query_count == result2.query_count


@pytest.mark.unit
class TestContradictionResolutionEvaluator:
    """Unit tests for ContradictionResolutionEvaluator."""

    def test_single_expected_result_scores_1(self) -> None:
        """Single expected result (no contradiction) should score 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 0
        assert result.details["type"] == "no_contradiction"

    def test_multiple_expected_full_retrieval(self) -> None:
        """Multiple expected with full retrieval should score 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2", "M3"],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 1
        assert result.details["type"] == "contradiction"
        assert result.details["coverage_type"] == "full_coverage"

    def test_multiple_expected_no_retrieval(self) -> None:
        """Multiple expected with no retrieval should score 0.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=[],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0
        assert result.query_count == 1
        assert result.details["coverage_type"] == "no_retrieval"

    def test_multiple_expected_partial_retrieval(self) -> None:
        """Multiple expected with partial retrieval should score 0.5 (coverage) × weights."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1"],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        # Score = 0.7 * 0.5 (partial coverage) + 0.3 * 1.0 (ordering N/A for 1 item)
        expected_score = 0.7 * 0.5 + 0.3 * 1.0
        assert result.value == expected_score
        assert result.details["coverage_type"] == "partial_coverage"

    def test_ordering_newer_first(self) -> None:
        """When newer memories retrieved first, ordering score should be 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1", "M2"],
            retrieved_creation_days={"M1": 2, "M2": 1},  # M1 is newer (day 2)
        )
        result = evaluator.evaluate_with_context(context)
        assert result.details["ordering_score"] == 1.0

    def test_ordering_older_first(self) -> None:
        """When older memories retrieved first, ordering score should be 0.5."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1", "M2"],
            retrieved_creation_days={"M1": 1, "M2": 2},  # M2 is newer but M1 first
        )
        result = evaluator.evaluate_with_context(context)
        assert result.details["ordering_score"] == 0.5
        assert "warning" in result.details

    def test_metric_name_correct(self) -> None:
        """Test the metric name is correct."""
        evaluator = ContradictionResolutionEvaluator()
        assert evaluator.metric_name() == "benchmark.contradiction_resolution"

    def test_deterministic_replay(self) -> None:
        """Test that same context produces same result."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1", "M2"],
            retrieved_creation_days={"M1": 2, "M2": 1},
        )
        result1 = evaluator.evaluate_with_context(context)
        result2 = evaluator.evaluate_with_context(context)
        assert result1.value == result2.value
        assert result1.query_count == result2.query_count

    def test_empty_expected_and_retrieved(self) -> None:
        """Empty expected and retrieved should score 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=[],
            expected_ids=[],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 0
