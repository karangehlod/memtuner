"""Unit tests for the ReliabilityCurveEvaluator."""

from __future__ import annotations

import pytest

from benchmark.evaluation.reliability import ReliabilityCurveEvaluator


@pytest.mark.unit
class TestReliabilityCurveEvaluator:
    """Tests for the reliability curve evaluator."""

    def test_empty_curve(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        result = evaluator.compute_curve()
        assert result.survival_rates == {}
        assert result.total_injected == 0

    def test_single_day_full_survival(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=10, injected_count=10)
        result = evaluator.compute_curve()
        assert result.survival_rates[0] == 1.0
        assert result.total_injected == 10

    def test_decay_over_time(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=10, injected_count=10)
        evaluator.record_day(day=7, alive_count=8)
        evaluator.record_day(day=14, alive_count=5)
        result = evaluator.compute_curve()
        assert result.survival_rates[0] == 1.0
        assert result.survival_rates[7] == 0.8
        assert result.survival_rates[14] == 0.5

    def test_evaluate_returns_latest_rate(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=10, injected_count=10)
        evaluator.record_day(day=7, alive_count=6)
        eval_result = evaluator.evaluate([], [])
        assert eval_result.metric_name == "benchmark.memory_survival_rate"
        assert eval_result.value == 0.6

    def test_evaluate_empty_returns_zero(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        eval_result = evaluator.evaluate([], [])
        assert eval_result.value == 0.0

    def test_metric_name(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        assert evaluator.metric_name() == "benchmark.memory_survival_rate"

    def test_reset_clears_state(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=10, injected_count=10)
        evaluator.reset()
        result = evaluator.compute_curve()
        assert result.total_injected == 0
        assert result.survival_rates == {}

    def test_multiple_injections(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=5, injected_count=5)
        evaluator.record_day(day=3, alive_count=8, injected_count=3)
        result = evaluator.compute_curve()
        assert result.total_injected == 8
        assert result.survival_rates[0] == 5 / 8
        assert result.survival_rates[3] == 1.0

    def test_evaluate_details_contain_day_rates(self) -> None:
        evaluator = ReliabilityCurveEvaluator()
        evaluator.record_day(day=0, alive_count=10, injected_count=10)
        evaluator.record_day(day=7, alive_count=8)
        eval_result = evaluator.evaluate([], [])
        assert "0" in eval_result.details
        assert "7" in eval_result.details
