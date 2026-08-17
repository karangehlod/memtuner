"""Generate gold command — creates parameterized gold datasets deterministically.

Exposes GoldGenerator via CLI for reproducible benchmark dataset creation.
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@click.command("generate-gold")
@click.option(
    "--seed",
    type=int,
    required=True,
    help="Random seed for deterministic generation.",
)
@click.option(
    "--users",
    type=int,
    default=10,
    help="Number of distinct users to simulate.",
)
@click.option(
    "--days",
    type=int,
    default=7,
    help="Number of simulated days.",
)
@click.option(
    "--events-per-day",
    type=int,
    default=10,
    help="Events to inject per day.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path for generated gold dataset JSON.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate dataset without writing to file.",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate output against schema before writing.",
)
def generate_gold(
    seed: int,
    users: int,
    days: int,
    events_per_day: int,
    output: str,
    dry_run: bool,
    validate: bool,
) -> None:
    """Generate a parameterized gold dataset deterministically.

    All parameters are deterministic: same seed + config = identical output file.
    Useful for reproducible benchmark scenarios and sweeps.
    """
    import json

    from benchmark.gold.generator import GoldGenerator, GoldGeneratorConfig
    from benchmark.gold.schema import GoldDataset
    from benchmark.time.simulated_clock import SimulatedClock

    click.echo(
        f"🎲 Generating gold dataset: seed={seed}, users={users}, days={days}, "
        f"events_per_day={events_per_day}"
    )

    # Create config and generator
    config = GoldGeneratorConfig(seed=seed, users=users, days=days, events_per_day=events_per_day)
    time_provider = SimulatedClock()
    generator = GoldGenerator(config, time_provider)

    # Generate dataset
    try:
        dataset = generator.generate()
    except Exception as e:
        click.echo(f"❌ Generation failed: {e}", err=True)
        raise SystemExit(1)

    click.echo(
        f"✅ Generated {len(dataset.events)} days × {len(dataset.user_ids)} users "
        f"= {sum(len(d.memory_events) for d in dataset.events)} total events"
    )

    # Validate if requested
    if validate:
        try:
            data_dict = (
                dataset.model_dump(mode="python")
                if hasattr(dataset, "model_dump")
                else dataset.dict()
            )
            GoldDataset.model_validate(data_dict)
            click.echo("✅ Schema validation passed")
        except Exception as e:
            click.echo(f"❌ Schema validation failed: {e}", err=True)
            raise SystemExit(1)

    if dry_run:
        click.echo("✓ Dry run OK — dataset valid (not written to disk)")
        return

    # Write to file
    output_path = Path(output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_dict = (
            dataset.model_dump(mode="python") if hasattr(dataset, "model_dump") else dataset.dict()
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
        click.echo(f"💾 Output written to {output_path}")
    except Exception as e:
        click.echo(f"❌ Write failed: {e}", err=True)
        raise SystemExit(1)
