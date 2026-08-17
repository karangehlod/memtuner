"""Summary report generator.

Produces human-readable text summaries from benchmark run results.
"""

from __future__ import annotations

from benchmark.models.run_result import BenchmarkRunResult


class SummaryReportGenerator:
    """Generates human-readable text summaries from benchmark results.

    This class contains NO evaluation logic — it only formats
    pre-computed results for human consumption.
    """

    def generate(self, result: BenchmarkRunResult) -> str:
        """Generate a complete text summary.

        Args:
            result: The benchmark run result to summarize.

        Returns:
            Formatted text summary string.
        """
        lines: list[str] = []
        lines.append(self._header(result))
        lines.append(self._aggregate_metrics(result))
        lines.append(self._scenario_details(result))
        lines.append(self._cost_section(result))
        lines.append(self._footer())
        return "\n".join(lines)

    def _header(self, result: BenchmarkRunResult) -> str:
        """Generate the report header.

        Args:
            result: The benchmark result.

        Returns:
            Header string.
        """
        separator = "═" * 50
        return (
            f"\n{separator}\n"
            f"  BENCHMARK RESULTS — Run {result.run_id}\n"
            f"{separator}\n"
            f"  Config Hash: {result.config_hash}\n"
            f"  Seed: {result.seed}\n"
            f"  Modules: {', '.join(result.memory_modules_enabled)}\n"
        )

    def _aggregate_metrics(self, result: BenchmarkRunResult) -> str:
        """Generate aggregate metrics section.

        Args:
            result: The benchmark result.

        Returns:
            Aggregate metrics string.
        """
        return (
            f"\n  Aggregate Metrics:\n"
            f"  {'─' * 40}\n"
            f"  Recall@K:              {result.aggregate_recall_at_k:.1%}\n"
            f"  Temporal Accuracy:     {result.aggregate_temporal_accuracy:.1%}\n"
            f"  Contamination Rate:    {result.aggregate_contamination_rate:.1%}\n"
        )

    def _scenario_details(self, result: BenchmarkRunResult) -> str:
        """Generate per-scenario details.

        Args:
            result: The benchmark result.

        Returns:
            Scenario details string.
        """
        lines: list[str] = [f"\n  Per-Scenario Results:\n  {'─' * 40}"]
        for scenario in result.scenario_results:
            lines.append(
                f"  [{scenario.scenario_name}]\n"
                f"    Recall@K:            {scenario.recall_at_k:.1%}\n"
                f"    Precision@K:         {scenario.precision_at_k:.1%}\n"
                f"    Contamination Rate:  {scenario.contamination_rate:.1%}\n"
                f"    Temporal Accuracy:   {scenario.temporal_accuracy:.1%}\n"
                f"    Queries: {scenario.total_queries} | "
                f"Correct: {scenario.correct_recalls}\n"
            )
        return "\n".join(lines)

    def _cost_section(self, result: BenchmarkRunResult) -> str:
        """Generate cost summary section.

        Args:
            result: The benchmark result.

        Returns:
            Cost section string.
        """
        cost = result.cost_summary
        return (
            f"\n  Cost Summary:\n"
            f"  {'─' * 40}\n"
            f"  Total Cost:            ${cost.total_cost:.4f}\n"
            f"  Cost/Correct Recall:   ${cost.cost_per_correct_recall:.4f}\n"
        )

    def _footer(self) -> str:
        """Generate the report footer.

        Returns:
            Footer string.
        """
        return f"\n{'═' * 50}\n"
