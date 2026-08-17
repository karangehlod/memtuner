"""Unit tests for the explorer data loader and server factory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.exceptions.evaluation_errors import BenchmarkError
from benchmark.explorer.data_loader import ExplorerDataLoader
from benchmark.models.run_result import BenchmarkRunResult, CostSummary, ScenarioMetrics


def _make_run_result(run_id: str, recall: float = 0.85) -> BenchmarkRunResult:
    """Create a sample BenchmarkRunResult for testing."""
    scenario = ScenarioMetrics(
        scenario_name="delayed_recall",
        recall_at_k=recall,
        contamination_rate=0.1,
        temporal_accuracy=0.9,
        memory_survival_rates={0: 1.0, 7: 0.8},
        total_queries=4,
        correct_recalls=3,
    )
    return BenchmarkRunResult(
        run_id=run_id,
        config_hash="abc123",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        seed=42,
        memory_modules_enabled=["episodic_buffer", "episodic_store"],
        scenario_results=[scenario],
        cost_summary=CostSummary(
            total_token_cost=0.05,
            total_storage_cost=0.001,
            total_cost=0.051,
            cost_per_correct_recall=0.017,
        ),
        aggregate_recall_at_k=recall,
        aggregate_temporal_accuracy=0.9,
        aggregate_contamination_rate=0.1,
    )


@pytest.fixture()
def results_directory(tmp_path: Path) -> Path:
    """Create a temp directory with sample run result files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    for run_id, recall in [("run-001", 0.85), ("run-002", 0.92)]:
        result = _make_run_result(run_id, recall)
        file_path = results_dir / f"run_{run_id}.json"
        file_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return results_dir


@pytest.mark.unit
class TestExplorerDataLoader:
    """Tests for the ExplorerDataLoader class."""

    def test_load_all_runs_success(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        runs = loader.load_all_runs()
        assert len(runs) == 2

    def test_load_all_runs_missing_directory_raises(self, tmp_path: Path) -> None:
        loader = ExplorerDataLoader(tmp_path / "nonexistent")
        with pytest.raises(BenchmarkError, match="Results directory not found"):
            loader.load_all_runs()

    def test_get_run_returns_loaded_run(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        run = loader.get_run("run-001")
        assert run is not None
        assert run.run_id == "run-001"

    def test_get_run_returns_none_for_unknown(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        assert loader.get_run("nonexistent") is None

    def test_list_run_ids_sorted(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        ids = loader.list_run_ids()
        assert ids == ["run-001", "run-002"]

    def test_get_metric_series_recall(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("recall_at_k")
        assert len(series) == 2
        values = [s["value"] for s in series]
        assert 0.85 in values
        assert 0.92 in values

    def test_get_metric_series_temporal_accuracy(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("temporal_accuracy")
        assert len(series) == 2

    def test_get_metric_series_contamination_rate(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("contamination_rate")
        assert len(series) == 2

    def test_get_metric_series_total_cost(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("total_cost")
        assert len(series) == 2
        assert all(s["value"] == 0.051 for s in series)

    def test_get_metric_series_cost_per_correct_recall(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("cost_per_correct_recall")
        assert len(series) == 2

    def test_get_metric_series_unknown_metric_returns_empty(
        self, results_directory: Path
    ) -> None:
        loader = ExplorerDataLoader(results_directory)
        loader.load_all_runs()
        series = loader.get_metric_series("nonexistent_metric")
        assert series == []

    def test_load_skips_invalid_json_files(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "run_bad.json").write_text("not valid json", encoding="utf-8")
        loader = ExplorerDataLoader(results_dir)
        runs = loader.load_all_runs()
        assert len(runs) == 0

    def test_load_skips_invalid_schema_files(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "run_bad.json").write_text(
            json.dumps({"not": "a valid run result"}),
            encoding="utf-8",
        )
        loader = ExplorerDataLoader(results_dir)
        runs = loader.load_all_runs()
        assert len(runs) == 0

    def test_reload_clears_previous_data(self, results_directory: Path) -> None:
        loader = ExplorerDataLoader(results_directory)
        runs1 = loader.load_all_runs()
        assert len(runs1) == 2
        # Remove one file and reload
        for f in results_directory.glob("run_run-002*.json"):
            f.unlink()
        runs2 = loader.load_all_runs()
        assert len(runs2) == 1
