"""Explore command — launches the local explorer UI.

Starts a local web server for browsing benchmark results.
Requires the `explorer` optional dependency (fastapi + uvicorn).
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command("explore")
@click.option(
    "--results-dir",
    "-d",
    type=click.Path(exists=True),
    default="outputs",
    help="Directory containing benchmark result JSON files.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind the explorer server.",
)
@click.option(
    "--port",
    type=int,
    default=8501,
    help="Port for the explorer server.",
)
def explore_results(results_dir: str, host: str, port: int) -> None:
    """Launch the local explorer UI for browsing benchmark results."""
    try:
        from benchmark.explorer.server import run_explorer_server
    except ImportError:
        click.echo(
            "❌ Explorer requires optional dependencies. "
            "Install with: pip install -e '.[explorer]'",
            err=True,
        )
        raise SystemExit(1) from None

    results_path = Path(results_dir)
    click.echo(f"🔍 Starting explorer server at http://{host}:{port}")
    click.echo(f"📂 Loading results from {results_path}")
    click.echo("   Press Ctrl+C to stop.")
    run_explorer_server(results_path, host=host, port=port)
