"""CSV report output.

Serializes benchmark results to CSV format for spreadsheet analysis.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from benchmark.models.run_result import BenchmarkRunResult


class CsvReportWriter:
    """Writes benchmark results as CSV files.

    Produces two CSV files:
    - summary.csv: aggregate metrics per run
    - scenarios.csv: per-scenario metrics

    No computation — pure serialization.
    """

    def write(self, result: BenchmarkRunResult, output_dir: Path) -> dict[str, Path]:
        """Write benchmark results to CSV files.

        Args:
            result: The benchmark run result.
            output_dir: Directory to write CSV files into.

        Returns:
            Dictionary mapping file type to output path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / f"summary_{result.run_id}.csv"
        scenarios_path = output_dir / f"scenarios_{result.run_id}.csv"

        self._write_summary_csv(result, summary_path)
        self._write_scenarios_csv(result, scenarios_path)

        return {"summary": summary_path, "scenarios": scenarios_path}

    def to_summary_string(self, result: BenchmarkRunResult) -> str:
        """Serialize summary metrics to a CSV string.

        Args:
            result: The benchmark run result.

        Returns:
            CSV-formatted string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(self._summary_headers())
        writer.writerow(self._summary_row(result))
        return output.getvalue()

    def to_scenarios_string(self, result: BenchmarkRunResult) -> str:
        """Serialize per-scenario metrics to a CSV string.

        Args:
            result: The benchmark run result.

        Returns:
            CSV-formatted string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(self._scenario_headers())
        for scenario in result.scenario_results:
            writer.writerow(self._scenario_row(result.run_id, scenario))
        return output.getvalue()

    def _write_summary_csv(self, result: BenchmarkRunResult, output_path: Path) -> None:
        """Write the summary CSV file."""
        with output_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(self._summary_headers())
            writer.writerow(self._summary_row(result))

    def _write_scenarios_csv(self, result: BenchmarkRunResult, output_path: Path) -> None:
        """Write the per-scenario CSV file."""
        with output_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(self._scenario_headers())
            for scenario in result.scenario_results:
                writer.writerow(self._scenario_row(result.run_id, scenario))

    def _summary_headers(self) -> list[str]:
        """Return summary CSV column headers."""
        return [
            "run_id",
            "config_hash",
            "seed",
            "aggregate_recall_at_k",
            "aggregate_temporal_accuracy",
            "aggregate_contamination_rate",
            "total_cost_usd",
            "cost_per_correct_recall_usd",
            "memory_modules",
        ]

    def _summary_row(self, result: BenchmarkRunResult) -> list[str]:
        """Return a summary CSV data row."""
        return [
            result.run_id,
            result.config_hash,
            str(result.seed),
            f"{result.aggregate_recall_at_k:.4f}",
            f"{result.aggregate_temporal_accuracy:.4f}",
            f"{result.aggregate_contamination_rate:.4f}",
            f"{result.cost_summary.total_cost:.6f}",
            f"{result.cost_summary.cost_per_correct_recall:.6f}",
            ";".join(result.memory_modules_enabled),
        ]

    def _scenario_headers(self) -> list[str]:
        """Return scenario CSV column headers."""
        return [
            "run_id",
            "scenario_name",
            "recall_at_k",
            "precision_at_k",
            "contamination_rate",
            "temporal_accuracy",
            "total_queries",
            "correct_recalls",
        ]

    def _scenario_row(self, run_id: str, scenario: object) -> list[str]:
        """Return a scenario CSV data row."""
        from benchmark.models.run_result import ScenarioMetrics

        if not isinstance(scenario, ScenarioMetrics):
            return []
        return [
            run_id,
            scenario.scenario_name,
            f"{scenario.recall_at_k:.4f}",
            f"{scenario.precision_at_k:.4f}",
            f"{scenario.contamination_rate:.4f}",
            f"{scenario.temporal_accuracy:.4f}",
            str(scenario.total_queries),
            str(scenario.correct_recalls),
        ]
