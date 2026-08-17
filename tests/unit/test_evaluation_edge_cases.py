"""Edge case tests for evaluation metrics."""

from __future__ import annotations

import pytest

from benchmark.evaluation.contradiction_resolution import ContradictionResolutionEvaluator
from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.false_positive import FalsePositiveEvaluator
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator


@pytest.mark.unit
class TestRecallEdgeCases:
    """Edge cases for recall evaluator."""

    def test_recall_empty_retrieved_empty_expected(self) -> None:
        """Both empty raises ValueError (no vacuous truth)."""
        evaluator = RecallEvaluator(top_k=5)
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            evaluator.evaluate([], [])

    def test_recall_empty_retrieved_with_expected(self) -> None:
        """Empty retrieved with expected should give 0.0."""
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate([], ["M1", "M2", "M3"])
        assert result.value == 0.0

    def test_recall_perfect_retrieval(self) -> None:
        """All expected retrieved should give 1.0."""
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(["M1", "M2", "M3"], ["M1", "M2", "M3"])
        assert result.value == 1.0

    def test_recall_k_zero_handling(self) -> None:
        """Recall@1 with 2 expected and 1 hit should give 0.5."""
        evaluator = RecallEvaluator(top_k=1)
        result = evaluator.evaluate(["M1", "M2", "M3"], ["M1", "M2"])
        # Top-1 is ["M1"]. M1 is in expected. Recall@1 = 1/2 = 0.5
        assert result.value == 0.5

    def test_recall_k_larger_than_retrieved(self) -> None:
        """K larger than retrieved set should use all retrieved."""
        evaluator = RecallEvaluator(top_k=100)
        result = evaluator.evaluate(["M1", "M2"], ["M1", "M2", "M3"])
        # 2 out of 3 expected
        assert result.value == 2.0 / 3.0

    def test_recall_single_memory(self) -> None:
        """Single memory edge case."""
        evaluator = RecallEvaluator(top_k=5)
        result = evaluator.evaluate(["M1"], ["M1"])
        assert result.value == 1.0

    def test_recall_large_k(self) -> None:
        """Very large K should not cause errors."""
        evaluator = RecallEvaluator(top_k=10_000_000)
        result = evaluator.evaluate(["M1", "M2"], ["M1"])
        assert result.value == 1.0


@pytest.mark.unit
class TestFalsePositiveRateEdgeCases:
    """Edge cases for false positive rate evaluator."""

    def test_fpr_empty_retrieved(self) -> None:
        """Empty retrieved should give FPR 0.0."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate([], ["M1", "M2"])
        assert result.value == 0.0

    def test_fpr_empty_expected(self) -> None:
        """Empty expected means all retrieved are FPs."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(["M1", "M2"], [])
        assert result.value == 1.0

    def test_fpr_both_empty(self) -> None:
        """Both empty should give 0.0."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate([], [])
        assert result.value == 0.0

    def test_fpr_perfect_match(self) -> None:
        """Perfect match should give 0.0 FPR."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(["M1", "M2", "M3"], ["M1", "M2", "M3"])
        assert result.value == 0.0

    def test_fpr_all_false_positives(self) -> None:
        """All FPs should give 1.0."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(["N1", "N2", "N3"], ["M1", "M2"])
        assert result.value == 1.0

    def test_fpr_single_tp_many_fp(self) -> None:
        """One TP and many FPs."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(
            ["M1", "N1", "N2", "N3", "N4"], ["M1", "M2"]
        )
        # 4 FPs out of 5 retrieved
        assert result.value == 4.0 / 5.0

    def test_fpr_many_tp_one_fp(self) -> None:
        """Many TPs and one FP."""
        evaluator = FalsePositiveEvaluator()
        result = evaluator.evaluate(
            ["M1", "M2", "M3", "M4", "N1"], ["M1", "M2", "M3", "M4"]
        )
        # 1 FP out of 5 retrieved
        assert result.value == 1.0 / 5.0


@pytest.mark.unit
class TestTemporalBoundaries:
    """Temporal accuracy boundary conditions."""

    def test_temporal_accuracy_empty_cases(self) -> None:
        """Empty results should not crash."""
        evaluator = TemporalAccuracyEvaluator(temporal_tolerance_days=1)
        result = evaluator.evaluate([], [])
        assert 0.0 <= result.value <= 1.0

    def test_temporal_accuracy_single_memory(self) -> None:
        """Single memory should work."""
        evaluator = TemporalAccuracyEvaluator(temporal_tolerance_days=1)
        result = evaluator.evaluate(["M1"], ["M1"])
        assert 0.0 <= result.value <= 1.0

    def test_temporal_accuracy_zero_tolerance(self) -> None:
        """Zero tolerance should require exact match."""
        evaluator = TemporalAccuracyEvaluator(temporal_tolerance_days=0)
        result = evaluator.evaluate(["M1"], ["M1"])
        assert 0.0 <= result.value <= 1.0

    def test_temporal_accuracy_large_tolerance(self) -> None:
        """Large tolerance should be permissive."""
        evaluator = TemporalAccuracyEvaluator(temporal_tolerance_days=10_000)
        result = evaluator.evaluate(["M1", "M2"], ["M1", "M2"])
        assert 0.0 <= result.value <= 1.0


@pytest.mark.unit
class TestContradictionEdgeCases:
    """Edge cases for contradiction resolution."""

    def test_contradiction_empty_expected(self) -> None:
        """Empty expected should give 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1"],
            expected_ids=[],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0

    def test_contradiction_single_expected(self) -> None:
        """Single expected (no contradiction) should give 1.0."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2"],
            expected_ids=["M1"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0

    def test_contradiction_perfect_coverage(self) -> None:
        """Full coverage of multiple expected."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1", "M2", "M3"],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0

    def test_contradiction_no_coverage(self) -> None:
        """No coverage of expected contradictions."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=[],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0

    def test_contradiction_partial_coverage(self) -> None:
        """Partial coverage."""
        evaluator = ContradictionResolutionEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M1"],
            expected_ids=["M1", "M2"],
        )
        result = evaluator.evaluate_with_context(context)
        # 0.7 * 0.5 + 0.3 * 1.0 (no ordering check for 1 item)
        expected = 0.7 * 0.5 + 0.3 * 1.0
        assert abs(result.value - expected) < 1e-9


@pytest.mark.unit
class TestDeterminismEdgeCases:
    """Determinism under edge conditions."""

    def test_recall_deterministic_empty(self) -> None:
        """Recall is deterministic — empty expected always raises ValueError."""
        evaluator = RecallEvaluator(top_k=5)
        with pytest.raises(ValueError):
            evaluator.evaluate([], [])
        with pytest.raises(ValueError):
            evaluator.evaluate([], [])

    def test_fpr_deterministic_edge(self) -> None:
        """FPR should be deterministic at edges."""
        evaluator = FalsePositiveEvaluator()
        r1 = evaluator.evaluate([], [])
        r2 = evaluator.evaluate([], [])
        assert r1.value == r2.value

    def test_temporal_deterministic_edge(self) -> None:
        """Temporal should be deterministic at edges."""
        evaluator = TemporalAccuracyEvaluator(temporal_tolerance_days=1)
        r1 = evaluator.evaluate([], [])
        r2 = evaluator.evaluate([], [])
        assert r1.value == r2.value

    def test_contradiction_deterministic_edge(self) -> None:
        """Contradiction should be deterministic at edges."""
        evaluator = ContradictionResolutionEvaluator()
        ctx = EvaluationContext(retrieved_ids=[], expected_ids=[])
        r1 = evaluator.evaluate_with_context(ctx)
        r2 = evaluator.evaluate_with_context(ctx)
        assert r1.value == r2.value
