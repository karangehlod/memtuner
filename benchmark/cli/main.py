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


def _get_version() -> str:
    # Primary: importlib.metadata reads pyproject.toml after `pip install`.
    # Fallback: parse pyproject.toml directly so the source tree (before install)
    # never has a stale hardcoded string — version lives in ONE place only.
    try:
        from importlib.metadata import version
        return version("memtuner")
    except Exception:
        pass
    try:
        import re
        from pathlib import Path
        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        _m = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), re.M)
        if _m:
            return _m.group(1)
    except Exception:
        pass
    return "unknown"


def _scripts_path() -> str:
    """Return the absolute path to scripts/, working for both editable and wheel installs."""
    import sys
    from pathlib import Path
    # Editable install: __file__ is inside the source tree
    # Wheel install: scripts/ is a top-level package installed alongside benchmark/
    candidate = Path(__file__).resolve().parents[2] / "scripts"
    if candidate.is_dir():
        return str(candidate)
    # Try importlib.resources (wheel install puts scripts/ as a package)
    try:
        import importlib.resources as _res
        return str(_res.files("scripts"))
    except Exception:
        pass
    # Last resort: search sys.path for a scripts/ directory containing study_runner.py
    for p in sys.path:
        s = Path(p) / "scripts"
        if (s / "study_runner.py").exists():
            return str(s)
    return str(candidate)  # return best guess; import will fail with a clear error


@click.group()
@click.version_option(version=_get_version(), prog_name="memtuner")
def cli() -> None:
    """MemTuner — adaptive benchmarking for AI agent memory retrieval.

    \b
    Typical workflow (datasets download automatically when needed):
      memtuner doctor                  # 1. check hardware, get a tuned command
      memtuner study --mode quick      # 2. fast sanity check (~1 min, no GPU)
      memtuner study --mode full       # 3. all datasets × all 5 phases
      memtuner reports                 # 4. HTML dashboard + PNG plots

    \b
    Scope a run down:
      memtuner study --gold-dataset data/input/locomo10.json --mode quick
      memtuner study --mode custom --phases 1 2

    Run `memtuner <command> --help` for details. Advanced/legacy commands
    are hidden from this list but still available (see docs/RUNBOOK.md).
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


# Core commands, shown in `memtuner --help`
cli.add_command(validate_config, "validate")
cli.add_command(compare_runs, "compare")
cli.add_command(explore_results, "explore")

# Advanced / legacy commands — fully functional but hidden from the top-level
# help to keep the everyday surface small (doctor → study → reports).
for _advanced in (analyze_benchmark, run_benchmark, sweep_benchmark,
                  generate_report, generate_gold, migrate_config, locomo_group):
    _advanced.hidden = True
cli.add_command(analyze_benchmark, "analyze")
cli.add_command(run_benchmark, "run")
cli.add_command(sweep_benchmark, "sweep")
cli.add_command(generate_report, "report")
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
    # study_runner.py lives in scripts/ relative to the project root.
    # Resolve from this file's location so the installed venv binary works
    # regardless of the current working directory.
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    sys.argv = ["memtuner study"] + list(study_args)
    from study_runner import main
    main()


@cli.command("prepare-datasets")
@click.option("--download", is_flag=True, default=False,
              help="Download missing dataset source files.")
@click.option("--convert", is_flag=True, default=False,
              help="Convert downloaded source files to gold format.")
@click.option("--status", is_flag=True, default=False,
              help="Show dataset status only (default if no flags given).")
def prepare_datasets_cmd(download: bool, convert: bool, status: bool) -> None:
    """Download and convert benchmark datasets to gold format.

    Datasets are stored in data/input/ and are never committed — each stays
    under its original license (see NOTICE). With HF_TOKEN in .env,
    HuggingFace sources download authenticated; without it they are tried
    anonymously and the ones that fail are logged with a fix hint.

    You rarely need this command: `memtuner study` auto-prepares any
    dataset it is asked for.

    \b
    Core datasets (this command):
      locomo       GitHub snap-research (~3 MB)
      longmemeval  HuggingFace (~30 MB) — HF_TOKEN recommended
      squad        Stanford NLP (~36 MB)
      coqa         Stanford NLP (~55 MB)
      personachat  HuggingFace (~20 MB) — HF_TOKEN recommended
      hotpotqa     CMU server (~54 MB) — often down; HF parquet mirror
                   works via scripts/prepare_extended_datasets.py patterns
      synthetic    Generated on demand — no download

    \b
    Extended datasets (scripts/prepare_extended_datasets.py):
      fever, msmarco, multiwoz, narrativeqa, nq, webquestions, wizard

    \b
    Examples:
      memtuner prepare-datasets                    # show status
      memtuner prepare-datasets --download         # fetch missing files
      memtuner prepare-datasets --convert          # convert to gold format
      memtuner prepare-datasets --download --convert  # do both
    """
    import sys
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import prepare_datasets as _pd
    if download:
        _pd.do_download()
    if convert:
        _pd.do_convert()
    if not download and not convert:
        _pd.print_status()
        click.echo("Run with --download to fetch files, --convert to convert them.")
        click.echo("Both flags can be combined: --download --convert")


@cli.command("reports")
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Output directory (default: data/output/).")
@click.option("--no-plots", is_flag=True, default=False,
              help="Skip PNG chart generation (useful on headless servers).")
def reports_cmd(output_dir: str | None, no_plots: bool) -> None:
    """Generate HTML dashboard and PNG plots from all past benchmark runs.

    Scans data/output/ for completed study runs, builds:
      - master_results.csv  — merged table of all cells
      - reports_data.js     — dashboard data
      - plots/              — per-dataset PNG charts (skipped with --no-plots)
      - DATASET_RECOMMENDATIONS.md

    \b
    Example:
      memtuner reports
      memtuner reports --no-plots        # headless / CI environments
      memtuner reports -o /path/to/output
    """
    import sys
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import generate_reports as _gr
    if output_dir:
        import os
        os.environ["BENCHMARK_OUTPUT_DIR"] = output_dir
    _gr.generate(skip_plots=no_plots)


@cli.command("grid-search", hidden=True)
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True,
              help="Gold dataset JSON path.")
@click.option("--mode", type=click.Choice(["quick", "core3x3", "full"]), default="quick",
              help="Search mode (default: quick).")
@click.option("--workers", "-w", type=int, default=None,
              help="Parallel workers (default: cpu_count-1).")
@click.option("--output-dir", "-o", type=click.Path(), default="data/output",
              help="Output directory.")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def grid_search_cmd(dataset: str, mode: str, workers: int | None,
                    output_dir: str, extra_args: tuple) -> None:
    """Run a full 3D grid search (memory × strategy × decay).

    Tests ALL memory types × ALL retrieval strategies × ALL decay policies
    on the same dataset so results are directly comparable.

    \b
    Examples:
      memtuner grid-search -d data/input/locomo10.json
      memtuner grid-search -d data/input/locomo10.json --mode full
      memtuner grid-search -d data/input/locomo10.json --mode full --workers 8
    """
    import sys
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    args = ["--dataset", dataset, "--mode", mode, "--output-dir", output_dir]
    if workers:
        args += ["--workers", str(workers)]
    args += list(extra_args)
    sys.argv = ["memtuner grid-search"] + args
    import grid_search as _gs
    _gs.main()


@cli.command("matrix", hidden=True)
@click.option("--gold-dataset", "-d", type=click.Path(exists=True), required=True,
              help="Gold dataset JSON path.")
@click.option("--mode", type=click.Choice(["core3x3", "full", "lambda-sweep"]),
              default="core3x3", help="Run mode.")
@click.option("--workers", "-w", type=int, default=None)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def matrix_cmd(gold_dataset: str, mode: str, workers: int | None,
               extra_args: tuple) -> None:
    """Run the full 3D benchmark matrix (memory × strategy × decay).

    \b
    Examples:
      memtuner matrix -d data/input/locomo10.json --mode core3x3
      memtuner matrix -d data/input/locomo10.json --mode full --workers 8
    """
    import sys
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    args = ["--mode", mode, "--gold-dataset", gold_dataset]
    if workers:
        args += ["--workers", str(workers)]
    args += list(extra_args)
    sys.argv = ["memtuner matrix"] + args
    import matrix_runner as _mr
    _mr.main()


@cli.command("plots")
@click.option("--output-dir", "-o", type=click.Path(), default="data/output/plots",
              help="Output directory for PNG files.")
@click.option("--dpi", type=int, default=150,
              help="Plot resolution (default: 150 dpi; use 300 for publication).")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def plots_cmd(output_dir: str, dpi: int, extra_args: tuple) -> None:
    """Generate publication-quality PNG charts from benchmark results.

    Scans data/output/ for completed runs and writes PNG charts to
    the output directory. Use --dpi 300 for print/publication quality.

    \b
    Examples:
      memtuner plots
      memtuner plots --dpi 300 -o docs/figures/
    """
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parents[2]
    _scripts_dir = str(_project_root / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    args = ["--out", output_dir, "--dpi", str(dpi)] + list(extra_args)
    sys.argv = ["memtuner plots"] + args
    import plot_benchmark as _pb
    _pb.main()


@cli.command("compare-strategies", hidden=True)
@click.option("--gold-dataset", "-d", type=click.Path(exists=True), required=True,
              help="Gold dataset JSON path.")
@click.option("--output-dir", "-o", type=click.Path(), default="data/output/comparison",
              help="Output directory.")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def compare_strategies_cmd(gold_dataset: str, output_dir: str,
                            extra_args: tuple) -> None:
    """Compare retrieval strategies head-to-head on the same dataset.

    Runs all available strategies on the same gold dataset and produces
    a side-by-side comparison table and charts.

    \b
    Example:
      memtuner compare-strategies -d data/input/locomo10.json
    """
    import sys
    _scripts_dir = _scripts_path()
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    args = ["--gold-dataset", gold_dataset, "--output-dir", output_dir]
    args += list(extra_args)
    sys.argv = ["memtuner compare-strategies"] + args
    import compare_retrieval_strategies as _crs
    _crs.main()


@cli.command("diagnose", hidden=True)
@click.option("--gold-dataset", "-d", type=click.Path(exists=True), required=True,
              help="Gold dataset JSON path.")
@click.option("--memory-type", default="episodic",
              help="Memory type to test (default: episodic).")
@click.option("--strategy", default="bm25",
              help="Retrieval strategy to test (default: bm25).")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def diagnose_cmd(gold_dataset: str, memory_type: str, strategy: str,
                 extra_args: tuple) -> None:
    """Run a single benchmark cell and print full diagnostic output.

    Useful for debugging why a specific configuration fails or
    produces unexpected results.

    \b
    Examples:
      memtuner diagnose -d data/input/locomo10.json
      memtuner diagnose -d data/input/locomo10.json --memory-type episodic --strategy hybrid
    """
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parents[2]
    _scripts_dir = str(_project_root / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    args = ["--gold-dataset", gold_dataset,
            "--memory-type", memory_type,
            "--strategy", strategy]
    args += list(extra_args)
    sys.argv = ["memtuner diagnose"] + args
    import diagnose_cell as _dc
    _dc.main()


@cli.command("init", hidden=True)
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
