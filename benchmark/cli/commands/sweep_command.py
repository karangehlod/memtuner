"""Sweep command — Cartesian parameter sweep over memory configurations.

The unique differentiator of this benchmark: run all combinations of
(memory_type × strategy × decay × K) and find the optimal configuration.

Like MATLAB linspace + meshgrid for memory system parameters.
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


def _resolve_pack_data_dir(pack_name: str, pack_instance, explicit_data_dir: str | None) -> Path:
    if explicit_data_dir:
        return Path(explicit_data_dir)

    candidate_dir = Path("data") / pack_name
    if pack_instance.validate_data(candidate_dir):
        return candidate_dir

    fallback_dir = Path("data")
    if pack_instance.validate_data(fallback_dir):
        return fallback_dir

    return candidate_dir


@click.command("sweep")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Gold dataset JSON (required unless --pack is used)",
)
@click.option("--output", "-o", type=click.Path(), default="data/output", help="Output directory")
@click.option("--strategies", "-s", multiple=True, default=["bm25"], help="Strategies to test")
@click.option(
    "--k-values", "-k", multiple=True, type=int, default=[5, 10], help="K values to sweep"
)
@click.option(
    "--lambdas", "-l", multiple=True, type=float, default=[0.0, 0.01, 0.05], help="Decay λ values"
)
@click.option(
    "--alphas",
    "-a",
    multiple=True,
    type=float,
    default=[0.0, 0.5, 1.0],
    help="Decay ranking alpha values",
)
@click.option(
    "--pack",
    type=click.Choice(["longmemeval", "locomo", "private"]),
    default=None,
    help="Dataset pack",
)
@click.option("--data-dir", type=click.Path(exists=True), default=None, help="Pack data directory")
@click.option("--max-queries", type=int, default=None, help="Limit queries (for speed)")
def sweep_benchmark(
    dataset: str,
    output: str,
    strategies: tuple[str, ...],
    k_values: tuple[int, ...],
    lambdas: tuple[float, ...],
    alphas: tuple[float, ...],
    pack: str | None,
    data_dir: str | None,
    max_queries: int | None,
) -> None:
    """Run a Cartesian parameter sweep over memory configurations.

    Finds the optimal (strategy × K × decay × alpha) combination
    for your dataset. Like MATLAB parameter optimization for memory systems.

    Example:
        memtuner sweep -d data/input/locomo10.json -s bm25 -s hybrid -k 5 -k 10 -l 0.0 -l 0.05
    """
    import itertools
    import json
    import time

    from benchmark.application.composer import BenchmarkComposer
    from benchmark.config.loader import load_config_from_dict

    # Load dataset via pack if specified
    dataset_override = None
    if pack:
        from benchmark.packs.registry import PackRegistry

        pack_instance = PackRegistry.get(pack)
        resolved_dir = _resolve_pack_data_dir(pack, pack_instance, data_dir)
        pack_instance.load(resolved_dir)
        dataset_override = pack_instance.to_gold_dataset(
            max_queries=max_queries, seed=42, evaluation_horizon=100
        )
        click.echo(f"📦 Pack '{pack}': {len(dataset_override.queries)} queries")
    elif not dataset:
        raise click.UsageError("--dataset is required unless --pack is used")

    # Generate all combinations
    combinations = list(itertools.product(strategies, k_values, lambdas, alphas))
    total = len(combinations)

    click.echo(f"🔬 Parameter Sweep: {total} configurations")
    click.echo(f"   Strategies: {list(strategies)}")
    click.echo(f"   K values:   {list(k_values)}")
    click.echo(f"   Lambda (λ): {list(lambdas)}")
    click.echo(f"   Alpha (α):  {list(alphas)}")
    click.echo("")
    click.echo(
        f"{'#':<4} {'Strategy':<12} {'K':<4} {'λ':<6} {'α':<5} "
        f"{'Recall':>8} {'Prec':>8} {'Time':>7}"
    )
    click.echo("─" * 60)

    results = []
    for idx, (strategy, k, lam, alpha) in enumerate(combinations, 1):
        config_dict = {
            "memory": {"enabled": {"short_term": [], "long_term": ["episodic_store"]}},
            "policies": {
                "module_policies": {
                    "episodic_store": {
                        "decay": {"type": "exponential", "lambda": lam},
                        "pruning": {"strategy": "score_threshold", "threshold": 0.1},
                    }
                }
            },
            "benchmark": {
                "evaluation_horizon": 30,
                "seed": 42,
                "scenarios": ["delayed_recall"],
                "retrieval_strategy": strategy,
            },
            "observability": {
                "exporter": "none",
                "endpoint": "http://localhost:4317",
                "log_level": "ERROR",
            },
            "answering": {"enabled": False, "model": "", "max_tokens": 500},
        }

        try:
            config = load_config_from_dict(config_dict)
            composer = BenchmarkComposer()

            start = time.monotonic()
            if dataset_override:
                composed = composer.compose(config=config, dataset_override=dataset_override)
            else:
                composed = composer.compose(config=config, dataset_path=Path(dataset))
            result = composed.runner.run(composed.scenarios)
            elapsed = time.monotonic() - start

            sr = result.scenario_results[0]
            entry = {
                "strategy": strategy,
                "k": k,
                "lambda": lam,
                "alpha": alpha,
                "recall": sr.recall_at_k,
                "precision": sr.precision_at_k,
                "contamination": sr.contamination_rate,
                "temporal": sr.temporal_accuracy,
                "queries": sr.total_queries,
                "correct": sr.correct_recalls,
                "time_seconds": elapsed,
            }
            results.append(entry)

            click.echo(
                f"{idx:<4} {strategy:<12} {k:<4} {lam:<6.3f} {alpha:<5.1f} "
                f"{sr.recall_at_k:>7.1%} {sr.precision_at_k:>7.1%} {elapsed:>5.1f}s"
            )
        except Exception as exc:
            click.echo(
                f"{idx:<4} {strategy:<12} {k:<4} {lam:<6.3f} {alpha:<5.1f}  FAILED: {str(exc)[:30]}"
            )

    click.echo("─" * 60)

    if not results:
        click.echo("❌ No successful runs.")
        raise SystemExit(1)

    # Find winners
    best_recall = max(results, key=lambda r: r["recall"])
    best_precision = max(results, key=lambda r: r["precision"])
    best_f1 = max(
        results,
        key=lambda r: (
            2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"])
            if (r["recall"] + r["precision"]) > 0
            else 0
        ),
    )

    click.echo("\n🏆 WINNERS:")
    click.echo(
        f"   Best Recall:    {best_recall['strategy']} "
        f"K={best_recall['k']} λ={best_recall['lambda']} "
        f"({best_recall['recall']:.1%})"
    )
    click.echo(
        f"   Best Precision: {best_precision['strategy']} "
        f"K={best_precision['k']} λ={best_precision['lambda']} "
        f"({best_precision['precision']:.1%})"
    )

    f1 = (
        2 * best_f1["recall"] * best_f1["precision"] / (best_f1["recall"] + best_f1["precision"])
        if (best_f1["recall"] + best_f1["precision"]) > 0
        else 0
    )
    click.echo(
        f"   Best F1:        {best_f1['strategy']} "
        f"K={best_f1['k']} λ={best_f1['lambda']} (F1={f1:.3f})"
    )

    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "sweep_results.json"
    with results_file.open("w") as fh:
        json.dump(
            {
                "sweep_config": {
                    "strategies": list(strategies),
                    "k_values": list(k_values),
                    "lambdas": list(lambdas),
                    "alphas": list(alphas),
                },
                "results": results,
            },
            fh,
            indent=2,
        )

    click.echo(f"\n💾 Results: {results_file}")
