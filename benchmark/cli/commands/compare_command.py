"""Compare command — compares two or more benchmark runs.

Reads JSON result files and produces a comparison table.
No business logic — purely reads, compares, and formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@click.command("compare")
@click.option(
    "--runs",
    "-r",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Paths to JSON result files to compare (use multiple -r flags).",
)
@click.option(
    "--metrics",
    "-m",
    default=None,
    help="Comma-separated list of metrics to compare (default: all standard).",
)
def compare_runs(runs: tuple[str, ...], metrics: str | None) -> None:
    """Compare metrics across two or more benchmark runs."""
    from benchmark.models.run_result import BenchmarkRunResult
    from benchmark.reporting.comparator import RunComparator

    if len(runs) < 2:
        click.echo("❌ At least two run result files are required for comparison.", err=True)
        raise SystemExit(1)

    loaded_results: list[BenchmarkRunResult] = []
    for run_path_str in runs:
        run_path = Path(run_path_str)
        try:
            raw_data = json.loads(run_path.read_text(encoding="utf-8"))
            result = BenchmarkRunResult.model_validate(raw_data)
            loaded_results.append(result)
            click.echo(f"📄 Loaded run: {result.run_id} from {run_path}")
        except (json.JSONDecodeError, ValueError) as error:
            click.echo(f"❌ Failed to load {run_path}: {error}", err=True)
            raise SystemExit(1) from error

    metric_names: list[str] | None = None
    if metrics:
        metric_names = [m.strip() for m in metrics.split(",")]

    comparator = RunComparator()
    table = comparator.compare(loaded_results, metric_names)
    click.echo(table.format_text())
