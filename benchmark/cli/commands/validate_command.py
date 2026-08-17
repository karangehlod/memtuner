"""Validate command — validates benchmark configuration.

Loads config and checks for errors without running the benchmark.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import click


@click.command("validate")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to benchmark config YAML.",
)
@click.option(
    "--check-api",
    is_flag=True,
    default=False,
    help="Probe the configured API embeddings endpoint after config validation.",
)
@click.option(
    "--check-environment",
    is_flag=True,
    default=False,
    help="Validate the local Python runtime and required benchmark imports.",
)
@click.option(
    "--environment-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write the environment validation report as JSON.",
)
def validate_config(
    config: str,
    check_api: bool,
    check_environment: bool,
    environment_output: Path | None,
) -> None:
    """Validate a benchmark configuration file."""
    from benchmark.config.loader import load_config_from_path
    from benchmark.exceptions.config_errors import BenchmarkError

    config_path = Path(config)
    click.echo(f"📄 Validating config: {config_path}")

    try:
        benchmark_config = load_config_from_path(config_path)
    except BenchmarkError as error:
        click.echo(f"❌ Validation failed: {error}", err=True)
        raise SystemExit(1) from None

    click.echo("✅ Configuration is valid.")
    click.echo(f"   Scenarios: {benchmark_config.benchmark.scenarios}")
    click.echo(f"   Evaluation horizon: {benchmark_config.benchmark.evaluation_horizon}")
    click.echo(f"   Seed: {benchmark_config.benchmark.seed}")
    click.echo(f"   STM modules: {benchmark_config.memory.enabled.short_term}")
    click.echo(f"   LTM modules: {benchmark_config.memory.enabled.long_term}")

    if check_environment or environment_output is not None:
        report = _build_environment_validation_report()
        _emit_environment_validation(report, environment_output)

    if check_api:
        _validate_api_connectivity(benchmark_config)


def _build_environment_validation_report() -> dict[str, Any]:
    checks = [_module_check(module_name) for module_name in _required_module_names()]
    report = {
        "schema_version": "1.0",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "checks": checks,
        "status": "ok" if all(check["status"] == "ok" for check in checks) else "failed",
    }
    return report


def _emit_environment_validation(
    report: dict[str, Any],
    environment_output: Path | None,
) -> None:
    click.echo("🧪 Environment validation")
    click.echo(f"   Python: {report['python']['executable']}")
    click.echo(f"   Version: {report['python']['version']}")

    for check in report["checks"]:
        marker = "✅" if check["status"] == "ok" else "❌"
        message = check.get("error") or "import ok"
        click.echo(f"   {marker} {check['module']}: {message}")

    if environment_output is not None:
        environment_output.parent.mkdir(parents=True, exist_ok=True)
        environment_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo(f"   Report written: {environment_output}")

    if report["status"] != "ok":
        raise click.ClickException(
            "Environment validation failed. Review the reported module import errors and Python runtime details."
        )


def _required_module_names() -> tuple[str, ...]:
    return (
        "benchmark.cli.main",
        "benchmark.cli.commands.analyze_command",
        "benchmark.config.loader",
        "benchmark.gold.oracle",
    )


def _module_check(module_name: str) -> dict[str, str]:
    try:
        importlib.import_module(module_name)
    except Exception as error:
        return {
            "module": module_name,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }

    return {
        "module": module_name,
        "status": "ok",
    }


def _validate_api_connectivity(benchmark_config) -> None:
    import os

    api_config = benchmark_config.benchmark.retrieval.api_embeddings
    base_url = api_config.base_url or os.environ.get("BENCHMARK_OPENAI_BASE_URL")
    model_name = api_config.model_name

    if not base_url:
        raise click.ClickException(
            "API embeddings validation requires benchmark.retrieval.api_embeddings.base_url "
            "or BENCHMARK_OPENAI_BASE_URL env var."
        )

    if not model_name:
        raise click.ClickException(
            "API embeddings validation requires benchmark.retrieval.api_embeddings.model_name."
        )

    try:
        import openai  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "API embeddings validation requires the openai package. "
            "Install: pip install openai"
        ) from None

    click.echo("🔎 API embeddings connectivity check")
    click.echo(f"   Model:    {model_name}")
    click.echo(f"   Base URL: {base_url}")
    click.echo(f"   Timeout:  {api_config.timeout}")

    try:
        from benchmark.memory.strategies.api_embeddings_strategy import ApiEmbeddingsStrategy

        strategy = ApiEmbeddingsStrategy(
            model_name=model_name,
            base_url=base_url,
            api_key=api_config.api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=api_config.timeout,
        )
        embeddings = strategy._embed_texts(["connectivity probe"])
        dimension = int(embeddings[0].shape[0]) if embeddings else 0
        click.echo(f"✅ API embeddings probe succeeded. Embedding dimension: {dimension}")
    except Exception as error:
        raise click.ClickException(f"API embeddings probe failed: {error}") from None
