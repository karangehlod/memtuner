"""Unit tests for reporting: SummaryReportGenerator, JsonReportWriter, and CsvReportWriter."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.models.run_result import BenchmarkRunResult, CostSummary, ScenarioMetrics
from benchmark.reporting.csv_report import CsvReportWriter
from benchmark.reporting.json_report import JsonReportWriter
from benchmark.reporting.summary import SummaryReportGenerator
from benchmark.cli.commands.analyze_command import _artifact_entry, _write_tagged_json


def _make_run_result() -> BenchmarkRunResult:
    """Create a sample BenchmarkRunResult for testing."""
    scenario = ScenarioMetrics(
        scenario_name="delayed_recall",
        recall_at_k=0.85,
        contamination_rate=0.1,
        temporal_accuracy=0.9,
        memory_survival_rates={0: 1.0, 7: 0.8, 14: 0.6},
        total_queries=4,
        correct_recalls=3,
    )
    return BenchmarkRunResult(
        run_id="test-run-001",
        config_hash="abc123def456",
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
        aggregate_recall_at_k=0.85,
        aggregate_temporal_accuracy=0.9,
        aggregate_contamination_rate=0.1,
    )


@pytest.mark.unit
class TestSummaryReportGenerator:
    """Tests for the SummaryReportGenerator."""

    def test_generates_nonempty_report(self) -> None:
        generator = SummaryReportGenerator()
        result = _make_run_result()
        report = generator.generate(result)
        assert len(report) > 0

    def test_report_contains_run_id(self) -> None:
        generator = SummaryReportGenerator()
        result = _make_run_result()
        report = generator.generate(result)
        assert "test-run-001" in report

    def test_report_contains_metrics(self) -> None:
        generator = SummaryReportGenerator()
        result = _make_run_result()
        report = generator.generate(result)
        assert "Recall@K" in report
        assert "Temporal Accuracy" in report
        assert "Contamination Rate" in report

    def test_report_contains_scenario_name(self) -> None:
        generator = SummaryReportGenerator()
        result = _make_run_result()
        report = generator.generate(result)
        assert "delayed_recall" in report

    def test_report_contains_cost_info(self) -> None:
        generator = SummaryReportGenerator()
        result = _make_run_result()
        report = generator.generate(result)
        assert "Cost" in report or "cost" in report.lower()


@pytest.mark.unit
class TestJsonReportWriter:
    """Tests for the JsonReportWriter."""

    def test_to_string_is_valid_json(self) -> None:
        writer = JsonReportWriter()
        result = _make_run_result()
        json_str = writer.to_string(result)
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "test-run-001"

    def test_to_string_contains_all_fields(self) -> None:
        writer = JsonReportWriter()
        result = _make_run_result()
        json_str = writer.to_string(result)
        parsed = json.loads(json_str)
        assert "scenario_results" in parsed
        assert "cost_summary" in parsed
        assert "aggregate_recall_at_k" in parsed

    def test_write_creates_file(self, tmp_path: Path) -> None:
        writer = JsonReportWriter()
        result = _make_run_result()
        output_file = tmp_path / "results" / "output.json"
        writer.write(result, output_file)
        assert output_file.exists()
        parsed = json.loads(output_file.read_text())
        assert parsed["run_id"] == "test-run-001"


@pytest.mark.unit
class TestCsvReportWriter:
    """Tests for the CsvReportWriter."""

    def test_to_summary_string_is_valid_csv(self) -> None:
        writer = CsvReportWriter()
        result = _make_run_result()
        csv_str = writer.to_summary_string(result)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2  # header + data
        assert rows[0][0] == "run_id"

    def test_summary_contains_run_id(self) -> None:
        writer = CsvReportWriter()
        result = _make_run_result()
        csv_str = writer.to_summary_string(result)
        assert "test-run-001" in csv_str

    def test_to_scenarios_string_is_valid_csv(self) -> None:
        writer = CsvReportWriter()
        result = _make_run_result()
        csv_str = writer.to_scenarios_string(result)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 scenario
        assert rows[1][1] == "delayed_recall"

    def test_write_creates_both_files(self, tmp_path: Path) -> None:
        writer = CsvReportWriter()
        result = _make_run_result()
        paths = writer.write(result, tmp_path / "csv_out")
        assert paths["summary"].exists()
        assert paths["scenarios"].exists()

    def test_summary_csv_has_correct_metric_values(self) -> None:
        writer = CsvReportWriter()
        result = _make_run_result()
        csv_str = writer.to_summary_string(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert float(row["aggregate_recall_at_k"]) == pytest.approx(0.85, abs=0.001)
        assert float(row["aggregate_temporal_accuracy"]) == pytest.approx(0.9, abs=0.001)


@pytest.mark.unit
class TestAnalyzeArtifacts:
    def test_artifact_entry_includes_tagged_metadata(self) -> None:
        artifact = _artifact_entry(
            "strategy_recall_precision",
            "image",
            "/tmp/strategy_recall_precision.png",
            "Compares recall and precision for each retrieval backend.",
        )

        assert artifact == {
            "tag": "strategy_recall_precision",
            "type": "image",
            "path": "/tmp/strategy_recall_precision.png",
            "description": "Compares recall and precision for each retrieval backend.",
        }

    def test_write_tagged_json_creates_manifest_file(self, tmp_path: Path) -> None:
        payload = {
            "artifacts": [
                _artifact_entry(
                    "embedding_backend_sweep",
                    "image",
                    "/tmp/embedding_backend_sweep.png",
                    "Shows backend-model tradeoffs.",
                )
            ]
        }

        path = _write_tagged_json(tmp_path, "artifact_manifest", payload)

        manifest_path = Path(path)
        assert manifest_path.exists()
        parsed = json.loads(manifest_path.read_text())
        assert parsed["artifacts"][0]["tag"] == "embedding_backend_sweep"
