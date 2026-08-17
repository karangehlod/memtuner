"""JSON report output.

Serializes benchmark results to JSON format.
"""

from __future__ import annotations

from pathlib import Path

from benchmark.models.run_result import BenchmarkRunResult


class JsonReportWriter:
    """Writes benchmark results as JSON files.

    No computation — pure serialization.
    """

    def write(self, result: BenchmarkRunResult, output_path: Path) -> None:
        """Write benchmark results to a JSON file.

        Args:
            result: The benchmark run result.
            output_path: Path to the output JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(result.model_dump_json(indent=2))

    def to_string(self, result: BenchmarkRunResult) -> str:
        """Serialize benchmark results to a JSON string.

        Args:
            result: The benchmark run result.

        Returns:
            JSON-formatted string.
        """
        return result.model_dump_json(indent=2)
