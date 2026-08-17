"""Run command — executes benchmark scenarios.

Delegates all composition to BenchmarkComposer.
No business logic — purely parses CLI options and invokes the composer.
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@click.command("run")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to benchmark config YAML.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="data/output",
    help="Output directory for results.",
)
@click.option(
    "--gold-dataset",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom gold dataset JSON (overrides config scenarios).",
)
@click.option(
    "--pack",
    type=click.Choice(["longmemeval", "locomo", "private"]),
    default=None,
    help="Benchmark pack to use (overrides --gold-dataset and config scenarios).",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=None,
    help="Data directory for the benchmark pack.",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Maximum number of queries to use from the pack.",
)
@click.option(
    "--allow-strategy-fallback",
    is_flag=True,
    default=False,
    help="Allow fallback to default scoring if strategy fails (not recommended).",
)
def run_benchmark(
    config: str,
    output_dir: str,
    gold_dataset: str | None,
    pack: str | None,
    data_dir: str | None,
    max_queries: int | None,
    allow_strategy_fallback: bool,
) -> None:
    """Execute benchmark scenarios from a config file."""
    from benchmark.application.composer import BenchmarkComposer
    from benchmark.application.errors import CompositionError
    from benchmark.config.loader import load_config_from_path
    from benchmark.reporting.json_report import JsonReportWriter
    from benchmark.reporting.summary import SummaryReportGenerator

    config_path = Path(config)
    click.echo(f"📄 Loading config from {config_path}")

    benchmark_config = load_config_from_path(config_path)

    # Determine dataset source
    dataset_path: Path | None = None
    dataset_override = None

    if pack:
        dataset_override = _load_pack_dataset(pack, data_dir, max_queries, benchmark_config)
        click.echo(
            f"📦 Pack '{pack}' loaded: "
            f"{len(dataset_override.queries)} queries, "
            f"{len(dataset_override.events)} event days"
        )
    elif gold_dataset:
        dataset_path = Path(gold_dataset)
        click.echo(f"📋 Using custom gold dataset: {dataset_path}")
    else:
        # Use first configured scenario from built-in datasets
        dataset_path = _resolve_scenario_dataset(benchmark_config)
        if dataset_path:
            click.echo(f"📋 Using scenario dataset: {dataset_path}")

    # Compose the benchmark (single composition path)
    composer = BenchmarkComposer()

    try:
        composed = composer.compose(
            config=benchmark_config,
            dataset_path=dataset_path,
            dataset_override=dataset_override,
            allow_strategy_fallback=allow_strategy_fallback,
        )
    except CompositionError as exc:
        click.echo(f"❌ {exc}", err=True)
        raise SystemExit(1) from exc

    # Report composition
    plan = composed.run_plan
    click.echo(f"🧠 Memory modules: {list(plan.memory_modules)}")
    click.echo(f"🔄 Strategy: {plan.effective_strategy}")
    if plan.lifecycle_policies:
        click.echo(f"♻️  Lifecycle policies: {list(plan.lifecycle_policies)}")
    click.echo(f"📊 Recall@K: K={plan.recall_k}")
    click.echo(f"📅 Effective horizon: {plan.effective_horizon} days")

    # Execute
    click.echo("🚀 Starting benchmark run...")
    result = composed.runner.run(composed.scenarios)

    # Report results
    summary_generator = SummaryReportGenerator()
    summary = summary_generator.generate(result)
    click.echo(summary)

    # Write output
    output_path = Path(output_dir)
    json_writer = JsonReportWriter()
    json_path = output_path / f"run_{result.run_id}.json"
    click.echo(f"📝 Writing results to {json_path.resolve()}")
    json_writer.write(result, json_path)
    click.echo(f"💾 Results written to {json_path.resolve()}")


def _load_pack_dataset(
    pack: str,
    data_dir: str | None,
    max_queries: int | None,
    config: object,
) -> object:
    """Load a benchmark pack dataset.

    Args:
        pack: Pack name (longmemeval, locomo, private).
        data_dir: Optional data directory override.
        max_queries: Max queries to load.
        config: The benchmark configuration.

    Returns:
        A GoldDataset instance.
    """
    from benchmark.packs.registry import PackRegistry

    pack_instance = PackRegistry.get(pack)
    default_dirs = {
        "longmemeval": "data/input/longmemeval",
        "locomo": "data/input",
    }
    resolved_data_dir = Path(data_dir) if data_dir else Path(default_dirs.get(pack, f"data/input/{pack}"))
    pack_instance.load(resolved_data_dir)
    return pack_instance.to_gold_dataset(
        max_queries=max_queries,
        seed=42,
        evaluation_horizon=config.benchmark.evaluation_horizon,
    )


def _resolve_scenario_dataset(config: object) -> Path | None:
    """Resolve the first available scenario dataset from config.

    Args:
        config: The benchmark configuration.

    Returns:
        Path to the dataset file, or None if not found.
    """
    datasets_dir = Path("benchmark/gold/datasets")
    for scenario_name in config.benchmark.scenarios:
        dataset_path = datasets_dir / f"{scenario_name}.json"
        if dataset_path.exists():
            return dataset_path
    return None
