"""Explorer data loader — reads benchmark results for visualization.

Loads JSON result files and provides data for the explorer UI.
Read-only — never modifies result files.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.exceptions.evaluation_errors import BenchmarkError
from benchmark.models.run_result import BenchmarkRunResult


class ExplorerDataLoader:
    """Loads benchmark results for the explorer UI.

    Scans an output directory for result JSON files and provides
    structured access to run data for visualization.
    """

    def __init__(self, results_directory: Path) -> None:
        """Initialize with the results directory.

        Args:
            results_directory: Path to directory containing run_*.json files.
        """
        self._results_directory = results_directory
        self._runs: dict[str, BenchmarkRunResult] = {}

    def load_all_runs(self) -> list[BenchmarkRunResult]:
        """Load all benchmark result files from the directory.

        Returns:
            List of loaded BenchmarkRunResult objects.

        Raises:
            BenchmarkError: If the directory does not exist.
        """
        if not self._results_directory.exists():
            raise BenchmarkError(f"Results directory not found: {self._results_directory}")

        self._runs.clear()
        json_files = sorted(self._results_directory.glob("run_*.json"))

        for json_file in json_files:
            result = self._load_single_file(json_file)
            if result is not None:
                self._runs[result.run_id] = result

        return list(self._runs.values())

    def get_run(self, run_id: str) -> BenchmarkRunResult | None:
        """Get a specific run by ID.

        Args:
            run_id: The run identifier.

        Returns:
            The BenchmarkRunResult or None if not found.
        """
        return self._runs.get(run_id)

    def list_run_ids(self) -> list[str]:
        """List all loaded run IDs.

        Returns:
            Sorted list of run IDs.
        """
        return sorted(self._runs.keys())

    def get_metric_series(self, metric_name: str) -> list[dict[str, object]]:
        """Get a metric across all loaded runs for charting.

        Args:
            metric_name: The metric to extract (e.g., "recall_at_k").

        Returns:
            List of dicts with run_id and metric value.
        """
        series: list[dict[str, object]] = []
        metric_mapping = {
            "recall_at_k": lambda r: r.aggregate_recall_at_k,
            "temporal_accuracy": lambda r: r.aggregate_temporal_accuracy,
            "contamination_rate": lambda r: r.aggregate_contamination_rate,
            "total_cost": lambda r: r.cost_summary.total_cost,
            "cost_per_correct_recall": lambda r: r.cost_summary.cost_per_correct_recall,
        }

        extractor = metric_mapping.get(metric_name)
        if extractor is None:
            return series

        for run_id, result in sorted(self._runs.items()):
            series.append(
                {
                    "run_id": run_id,
                    "value": extractor(result),
                }
            )
        return series

    def _load_single_file(self, file_path: Path) -> BenchmarkRunResult | None:
        """Load and validate a single result file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The parsed BenchmarkRunResult or None on error.
        """
        try:
            raw_data = json.loads(file_path.read_text(encoding="utf-8"))
            return BenchmarkRunResult.model_validate(raw_data)
        except (json.JSONDecodeError, ValueError):
            return None
