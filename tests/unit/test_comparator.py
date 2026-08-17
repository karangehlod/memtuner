"""Unit tests for the RunComparator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.models.run_result import BenchmarkRunResult, CostSummary, ScenarioMetrics
from benchmark.reporting.comparator import RunComparator


def _make_result(
    run_id: str,
    recall: float = 0.8,
    temporal: float = 0.9,
    fpr: float = 0.1,
    cost_per_recall: float = 0.02,
) -> BenchmarkRunResult:
    """Create a test BenchmarkRunResult."""
    return BenchmarkRunResult(
        run_id=run_id,
        config_hash="test_hash",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        seed=42,
        memory_modules_enabled=["episodic_buffer"],
        scenario_results=[
            ScenarioMetrics(
                scenario_name="test",
                recall_at_k=recall,
                contamination_rate=fpr,
                temporal_accuracy=temporal,
                total_queries=4,
                correct_recalls=3,
            )
        ],
        cost_summary=CostSummary(
            total_cost=0.05,
            cost_per_correct_recall=cost_per_recall,
        ),
        aggregate_recall_at_k=recall,
        aggregate_temporal_accuracy=temporal,
        aggregate_contamination_rate=fpr,
    )


@pytest.mark.unit
class TestRunComparator:
    """Tests for the RunComparator."""

    def test_compare_two_runs(self) -> None:
        comparator = RunComparator()
        run_a = _make_result("run_a", recall=0.8)
        run_b = _make_result("run_b", recall=0.9)
        table = comparator.compare([run_a, run_b])
        assert len(table.run_ids) == 2
        assert len(table.rows) == 4  # 4 default metrics

    def test_compare_with_custom_metrics(self) -> None:
        comparator = RunComparator()
        run_a = _make_result("run_a")
        run_b = _make_result("run_b")
        table = comparator.compare([run_a, run_b], metric_names=["recall_at_k"])
        assert len(table.rows) == 1
        assert table.rows[0].metric_name == "recall_at_k"

    def test_delta_calculation(self) -> None:
        comparator = RunComparator()
        run_a = _make_result("run_a", recall=0.8)
        run_b = _make_result("run_b", recall=0.9)
        table = comparator.compare([run_a, run_b], metric_names=["recall_at_k"])
        delta = table.rows[0].delta
        assert delta is not None
        assert delta == pytest.approx(0.1, abs=0.001)

    def test_delta_none_for_more_than_two_runs(self) -> None:
        comparator = RunComparator()
        runs = [_make_result(f"run_{i}") for i in range(3)]
        table = comparator.compare(runs, metric_names=["recall_at_k"])
        assert table.rows[0].delta is None

    def test_format_text_produces_output(self) -> None:
        comparator = RunComparator()
        run_a = _make_result("run_a")
        run_b = _make_result("run_b")
        table = comparator.compare([run_a, run_b])
        text = table.format_text()
        assert "recall_at_k" in text
        assert "run_a" in text
        assert "run_b" in text

    def test_extract_all_default_metrics(self) -> None:
        comparator = RunComparator()
        run_a = _make_result("run_a", recall=0.85, temporal=0.92, fpr=0.05)
        table = comparator.compare([run_a, run_a])
        metric_names = [row.metric_name for row in table.rows]
        assert "recall_at_k" in metric_names
        assert "temporal_accuracy" in metric_names
        assert "contamination_rate" in metric_names
        assert "cost_per_correct_recall" in metric_names
