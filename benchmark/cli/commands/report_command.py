"""Report command — generates summary report from a previous run.

Reads a JSON result file and produces human-readable or CSV output.
No business logic — purely reads and formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@click.command("report")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a JSON result file from a benchmark run.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "csv"], case_sensitive=False),
    default="text",
    help="Output format for the report.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    default=None,
    help="Optional output file path. If omitted, prints to stdout.",
)
def generate_report(input_path: str, output_format: str, output_path: str | None) -> None:
    """Generate a report from a previous benchmark run result."""
    from benchmark.models.run_result import BenchmarkRunResult

    result_path = Path(input_path)
    click.echo(f"📄 Loading results from {result_path}")

    try:
        raw_data = json.loads(result_path.read_text(encoding="utf-8"))
        result = BenchmarkRunResult.model_validate(raw_data)
    except (json.JSONDecodeError, ValueError) as error:
        click.echo(f"❌ Failed to parse result file: {error}", err=True)
        raise SystemExit(1) from error

    report_content = _format_report(result, output_format)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_content, encoding="utf-8")
        click.echo(f"💾 Report written to {out}")
    else:
        click.echo(report_content)


def _format_report(result: object, output_format: str) -> str:
    """Format the benchmark result in the requested format.

    Args:
        result: The BenchmarkRunResult.
        output_format: One of "text", "json", "csv".

    Returns:
        Formatted report string.
    """
    from benchmark.models.run_result import BenchmarkRunResult
    from benchmark.reporting.csv_report import CsvReportWriter
    from benchmark.reporting.json_report import JsonReportWriter
    from benchmark.reporting.summary import SummaryReportGenerator

    if not isinstance(result, BenchmarkRunResult):
        return ""

    if output_format == "json":
        return JsonReportWriter().to_string(result)
    elif output_format == "csv":
        return CsvReportWriter().to_summary_string(result)
    else:
        return SummaryReportGenerator().generate(result)
