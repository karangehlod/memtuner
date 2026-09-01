"""CLI entry point — top-level click group.

Translates user intent into benchmark execution. Contains NO business logic.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from benchmark.cli.commands.analyze_command import analyze_benchmark
from benchmark.cli.commands.compare_command import compare_runs
from benchmark.cli.commands.explore_command import explore_results
from benchmark.cli.commands.generate_gold_command import generate_gold
from benchmark.cli.commands.locomo_command import locomo_group
from benchmark.cli.commands.migrate_command import migrate_config
from benchmark.cli.commands.report_command import generate_report
from benchmark.cli.commands.run_command import run_benchmark
from benchmark.cli.commands.sweep_command import sweep_benchmark
from benchmark.cli.commands.validate_command import validate_config


def _load_repo_env() -> None:
    """Load environment variables from the repository root .env file only."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_repo_env_without_dependency(env_path)
        return

    load_dotenv(dotenv_path=env_path, override=False)


def _load_repo_env_without_dependency(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without python-dotenv."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


@click.group()
@click.version_option(version="0.0.1", prog_name="memtuner")
def cli() -> None:
    """MemTuner — adaptive benchmarking for AI agent memory retrieval.

    Evaluate memory systems across accuracy, reliability, temporal
    correctness, efficiency, and cost.

    Run `memtuner doctor` first to see what your machine can run
    and get a copy-paste command tailored to your hardware.
    """
    _load_repo_env()


@cli.command("doctor")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed hardware information.")
@click.option("--apply", "-a", is_flag=True, default=False,
              help=(
                  "Write hardware-derived settings to .env so every subsequent run "
                  "automatically uses the correct worker count and model exclusions. "
                  "Safe to re-run — only the doctor-managed block in .env is updated."
              ))
def doctor_cmd(verbose: bool, apply: bool) -> None:
    """Check hardware capabilities and optionally write config to .env.

    Without --apply: prints hardware analysis and copy-paste commands.
    With --apply:    writes BENCHMARK_WORKERS and BENCHMARK_SKIP_MODELS
                     to .env so you never need to pass flags again.

    \b
    Examples:
      memtuner doctor               # analyse hardware, print commands
      memtuner doctor --apply       # analyse + write config to .env
    """
    from benchmark.cli.commands.doctor_command import run_doctor
    run_doctor(verbose=verbose, apply=apply)


cli.add_command(analyze_benchmark, "analyze")
cli.add_command(run_benchmark, "run")
cli.add_command(sweep_benchmark, "sweep")
cli.add_command(validate_config, "validate")
cli.add_command(generate_report, "report")
cli.add_command(compare_runs, "compare")
cli.add_command(explore_results, "explore")
cli.add_command(generate_gold, "generate-gold")
cli.add_command(migrate_config, "migrate-config")
cli.add_command(locomo_group, "locomo")


@cli.command("study", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("study_args", nargs=-1, type=click.UNPROCESSED)
def study_cmd(study_args: tuple) -> None:
    """Run the 5-phase adaptive study benchmark (wraps study_runner.py).

    All flags are forwarded directly to study_runner. Run with --help to
    see the full option list.

    \b
    Examples:
      memtuner study --gold-dataset data/input/locomo10.json --mode quick
      memtuner study --gold-dataset data/input/locomo10.json --mode full --workers 5
      memtuner study --doctor
    """
    import sys
    from pathlib import Path
    # study_runner.py lives in scripts/ relative to the project root.
    # Resolve from this file's location so the installed venv binary works
    # regardless of the current working directory.
    _scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    sys.argv = ["memtuner study"] + list(study_args)
    from study_runner import main
    main()


@cli.command("init")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="configs/study_defaults.yaml",
    help="Output path for default config.",
)
def init_project(output: str) -> None:
    """Initialize a new benchmark project with default config."""
    from benchmark.config.defaults import create_default_config

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = create_default_config()
    import yaml

    with output_path.open("w", encoding="utf-8") as file_handle:
        yaml.dump(
            config.model_dump(mode="python"),
            file_handle,
            default_flow_style=False,
        )

    click.echo(f"✅ Default config written to {output_path}")


if __name__ == "__main__":
    cli()
