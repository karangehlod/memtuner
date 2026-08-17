"""Integration test: end-to-end benchmark run with delayed_recall scenario.

Wires all real components and runs a full benchmark pipeline.
Verifies the complete flow from config → gold → memory → evaluation → report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.config.schema import (
    BenchmarkConfig,
    BenchmarkScopeConfig,
    MemoryConfig,
    MemorySelectionConfig,
)
from benchmark.cost.tracker import InMemoryCostTracker
from benchmark.evaluation.false_positive import FalsePositiveEvaluator
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator
from benchmark.factory.registry import MemoryModuleRegistry
from benchmark.factory.resolver import ConfigResolver
from benchmark.gold.oracle import GoldOracle
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.models.run_result import BenchmarkRunResult
from benchmark.orchestrator.benchmark_runner import BenchmarkRunner
from benchmark.reporting.json_report import JsonReportWriter
from benchmark.reporting.summary import SummaryReportGenerator
from benchmark.scenario.loader import GoldDatasetScenario
from benchmark.time.simulated_clock import SimulatedClock

DATASETS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
FIXED_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _build_full_pipeline() -> tuple[BenchmarkRunner, list[GoldDatasetScenario]]:
    """Wire up the complete benchmark pipeline with real components."""
    config = BenchmarkConfig(
        memory=MemoryConfig(
            enabled=MemorySelectionConfig(
                short_term=["episodic_buffer"],
                long_term=["episodic_store", "preference_store"],
            ),
        ),
        benchmark=BenchmarkScopeConfig(
            evaluation_horizon=14,
            seed=42,
            scenarios=["delayed_recall"],
        ),
    )

    registry = MemoryModuleRegistry()
    registry.register("episodic_buffer", EpisodicBuffer)
    registry.register("episodic_store", EpisodicStore)
    registry.register("preference_store", PreferenceStore)

    resolver = ConfigResolver(registry)
    memory_modules = resolver.resolve_memory_modules(config)

    clock = SimulatedClock(epoch=FIXED_EPOCH)
    oracle = GoldOracle()
    dataset = oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
    scenario = GoldDatasetScenario(dataset, evaluation_horizon=config.benchmark.evaluation_horizon)

    evaluators = [
        RecallEvaluator(top_k=5),
        FalsePositiveEvaluator(),
        TemporalAccuracyEvaluator(tolerance_days=1),
    ]

    cost_tracker = InMemoryCostTracker()

    runner = BenchmarkRunner(
        time_provider=clock,
        gold_oracle=oracle,
        memory_modules=memory_modules,
        evaluators=evaluators,
        cost_tracker=cost_tracker,
        config=config,
    )

    return runner, [scenario]


@pytest.mark.integration
class TestEndToEndBenchmarkRun:
    """Integration test: full benchmark run end-to-end."""

    def test_run_completes_without_error(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert isinstance(result, BenchmarkRunResult)

    def test_run_produces_run_id(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert len(result.run_id) > 0

    def test_run_produces_scenario_results(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].scenario_name == "delayed_recall"

    def test_run_metrics_are_in_range(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert 0.0 <= result.aggregate_recall_at_k <= 1.0
        assert 0.0 <= result.aggregate_temporal_accuracy <= 1.0
        assert 0.0 <= result.aggregate_contamination_rate <= 1.0

    def test_scenario_has_queries(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert result.scenario_results[0].total_queries > 0

    def test_cost_summary_populated(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert result.cost_summary.total_cost >= 0.0

    def test_memory_modules_listed(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        assert "episodic_buffer" in result.memory_modules_enabled
        assert "episodic_store" in result.memory_modules_enabled

    def test_summary_report_generates(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        report = SummaryReportGenerator().generate(result)
        assert "BENCHMARK RESULTS" in report
        assert result.run_id in report

    def test_json_report_generates(self) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        json_str = JsonReportWriter().to_string(result)
        assert result.run_id in json_str

    def test_json_report_writes_to_file(self, tmp_path: Path) -> None:
        runner, scenarios = _build_full_pipeline()
        result = runner.run(scenarios)
        output = tmp_path / "report.json"
        JsonReportWriter().write(result, output)
        assert output.exists()


@pytest.mark.integration
class TestDeterministicReplay:
    """Deterministic replay: same config + same seed → identical result."""

    def test_two_runs_produce_same_metrics(self) -> None:
        runner_a, scenarios_a = _build_full_pipeline()
        result_a = runner_a.run(scenarios_a)

        runner_b, scenarios_b = _build_full_pipeline()
        result_b = runner_b.run(scenarios_b)

        assert result_a.aggregate_recall_at_k == result_b.aggregate_recall_at_k
        assert result_a.aggregate_temporal_accuracy == result_b.aggregate_temporal_accuracy
        assert result_a.aggregate_contamination_rate == result_b.aggregate_contamination_rate

    def test_two_runs_produce_same_scenario_metrics(self) -> None:
        runner_a, scenarios_a = _build_full_pipeline()
        result_a = runner_a.run(scenarios_a)

        runner_b, scenarios_b = _build_full_pipeline()
        result_b = runner_b.run(scenarios_b)

        for sr_a, sr_b in zip(
            result_a.scenario_results, result_b.scenario_results, strict=True
        ):
            assert sr_a.recall_at_k == sr_b.recall_at_k
            assert sr_a.contamination_rate == sr_b.contamination_rate
            assert sr_a.total_queries == sr_b.total_queries
            assert sr_a.correct_recalls == sr_b.correct_recalls
