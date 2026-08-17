"""Unit tests for the explorer FastAPI server factory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.models.run_result import BenchmarkRunResult, CostSummary, ScenarioMetrics


def _make_run_result(run_id: str) -> BenchmarkRunResult:
    """Create a sample run result."""
    scenario = ScenarioMetrics(
        scenario_name="delayed_recall",
        recall_at_k=0.85,
        contamination_rate=0.1,
        temporal_accuracy=0.9,
        memory_survival_rates={0: 1.0},
        total_queries=4,
        correct_recalls=3,
    )
    return BenchmarkRunResult(
        run_id=run_id,
        config_hash="abc123",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        seed=42,
        memory_modules_enabled=["episodic_buffer"],
        scenario_results=[scenario],
        cost_summary=CostSummary(
            total_token_cost=0.0,
            total_storage_cost=0.0,
            total_cost=0.0,
            cost_per_correct_recall=0.0,
        ),
        aggregate_recall_at_k=0.85,
        aggregate_temporal_accuracy=0.9,
        aggregate_contamination_rate=0.1,
    )


@pytest.fixture()
def results_directory_with_data(tmp_path: Path) -> Path:
    """Create a temp directory with sample run results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = _make_run_result("run-001")
    (results_dir / "run_001.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return results_dir


@pytest.mark.unit
class TestExplorerServerFactory:
    """Tests for create_explorer_app function."""

    def test_create_app_returns_fastapi_instance(
        self, results_directory_with_data: Path
    ) -> None:
        pytest.importorskip("fastapi")
        from benchmark.explorer.server import create_explorer_app
        app = create_explorer_app(results_directory_with_data)
        assert app is not None
        assert app.title == "Agentic Memory Benchmark Explorer"

    def test_api_list_runs(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)

        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "count" in data

    def test_api_get_run_found(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)

        # First load data
        client.post("/api/reload")
        response = client.get("/api/runs/run-001")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "run-001"

    def test_api_get_run_not_found(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)
        response = client.get("/api/runs/nonexistent")
        assert response.status_code == 404

    def test_api_metrics_endpoint(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)

        client.post("/api/reload")
        response = client.get("/api/metrics/recall_at_k")
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "recall_at_k"
        assert "series" in data

    def test_api_reload_endpoint(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)
        response = client.post("/api/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["reloaded"] is True

    def test_index_returns_html(self, results_directory_with_data: Path) -> None:
        try:
            from fastapi.testclient import TestClient

            from benchmark.explorer.server import create_explorer_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = create_explorer_app(results_directory_with_data)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Benchmark Explorer" in response.text
