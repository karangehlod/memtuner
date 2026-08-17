"""Run comparator — compares metrics across two or more benchmark runs.

Pure comparison logic. No evaluation or memory internals.
"""

from __future__ import annotations

from benchmark.models.run_result import BenchmarkRunResult


class RunComparator:
    """Compares metrics across multiple benchmark runs.

    Takes pre-computed BenchmarkRunResult objects and produces
    a comparison table. Contains no evaluation logic — only
    reads and diffs existing metrics.
    """

    def compare(
        self,
        runs: list[BenchmarkRunResult],
        metric_names: list[str] | None = None,
    ) -> ComparisonTable:
        """Compare multiple runs across requested metrics.

        Args:
            runs: List of BenchmarkRunResult objects to compare.
            metric_names: Optional list of metric names to include.
                          Defaults to all standard metrics.

        Returns:
            A ComparisonTable with per-run metric values and deltas.
        """
        if metric_names is None:
            metric_names = [
                "recall_at_k",
                "temporal_accuracy",
                "contamination_rate",
                "cost_per_correct_recall",
            ]

        rows: list[ComparisonRow] = []
        for metric_name in metric_names:
            values = [self._extract_metric(run, metric_name) for run in runs]
            rows.append(
                ComparisonRow(
                    metric_name=metric_name,
                    run_values=dict(zip([r.run_id for r in runs], values, strict=True)),
                )
            )

        return ComparisonTable(
            run_ids=[r.run_id for r in runs],
            rows=rows,
        )

    def _extract_metric(self, run: BenchmarkRunResult, metric_name: str) -> float:
        """Extract a metric value from a run result.

        Args:
            run: The benchmark run result.
            metric_name: The metric to extract.

        Returns:
            The metric value.
        """
        metric_mapping = {
            "recall_at_k": run.aggregate_recall_at_k,
            "temporal_accuracy": run.aggregate_temporal_accuracy,
            "contamination_rate": run.aggregate_contamination_rate,
            "cost_per_correct_recall": run.cost_summary.cost_per_correct_recall,
            "total_cost": run.cost_summary.total_cost,
        }
        return metric_mapping.get(metric_name, 0.0)


class ComparisonRow:
    """A single metric comparison across runs.

    Attributes:
        metric_name: The metric being compared.
        run_values: Mapping of run_id → metric value.
    """

    def __init__(self, metric_name: str, run_values: dict[str, float]) -> None:
        self.metric_name = metric_name
        self.run_values = run_values

    @property
    def delta(self) -> float | None:
        """Return the delta between first and last run, if exactly 2 runs."""
        values = list(self.run_values.values())
        if len(values) == 2:
            return values[1] - values[0]
        return None


class ComparisonTable:
    """Complete comparison of metrics across multiple runs.

    Attributes:
        run_ids: Ordered list of run IDs being compared.
        rows: List of ComparisonRow, one per metric.
    """

    def __init__(self, run_ids: list[str], rows: list[ComparisonRow]) -> None:
        self.run_ids = run_ids
        self.rows = rows

    def format_text(self) -> str:
        """Format the comparison as a human-readable text table.

        Returns:
            Formatted text table string.
        """
        lines: list[str] = []
        col_width = max(len(rid) for rid in self.run_ids) + 2
        metric_width = max(len(row.metric_name) for row in self.rows) + 2

        header = f"{'Metric':<{metric_width}}"
        for run_id in self.run_ids:
            header += f"{run_id:>{col_width}}"
        if len(self.run_ids) == 2:
            header += f"{'Delta':>{col_width}}"

        separator = "─" * len(header)
        lines.append(f"\n{separator}")
        lines.append(header)
        lines.append(separator)

        for row in self.rows:
            line = f"{row.metric_name:<{metric_width}}"
            for run_id in self.run_ids:
                value = row.run_values.get(run_id, 0.0)
                if row.metric_name.endswith("cost") or "cost" in row.metric_name:
                    line += f"${value:>{col_width - 1}.4f}"
                else:
                    line += f"{value:>{col_width}.1%}"
            if row.delta is not None:
                delta = row.delta
                sign = "+" if delta >= 0 else ""
                if "cost" in row.metric_name:
                    line += f" {sign}${delta:.4f}"
                else:
                    line += f" {sign}{delta:.1%}"
            lines.append(line)

        lines.append(separator)
        return "\n".join(lines)
