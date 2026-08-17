"""Configuration loader — YAML to BenchmarkConfig.

Handles safe YAML loading and pydantic validation.
Fails fast on any config issues.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from benchmark.config.schema import BenchmarkConfig
from benchmark.exceptions.config_errors import ConfigLoadError, ConfigValidationError


def load_config_from_path(config_path: Path) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated BenchmarkConfig instance.

    Raises:
        ConfigLoadError: If the file cannot be read or parsed.
        ConfigValidationError: If the config fails schema validation.
    """
    raw_data = _read_yaml_file(config_path)
    return _validate_config(raw_data, source=str(config_path))


def load_config_from_dict(data: dict[str, Any]) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a dictionary.

    Args:
        data: Raw configuration dictionary.

    Returns:
        Validated BenchmarkConfig instance.

    Raises:
        ConfigValidationError: If the config fails schema validation.
    """
    return _validate_config(data, source="<dict>")


def _read_yaml_file(config_path: Path) -> dict[str, Any]:
    """Read and parse a YAML file safely.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed YAML data as a dictionary.

    Raises:
        ConfigLoadError: If the file cannot be read or parsed.
    """
    if not config_path.exists():
        raise ConfigLoadError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigLoadError(f"Configuration path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle)
    except yaml.YAMLError as yaml_error:
        raise ConfigLoadError(f"Failed to parse YAML configuration: {config_path}") from yaml_error
    except OSError as io_error:
        raise ConfigLoadError(f"Failed to read configuration file: {config_path}") from io_error

    if not isinstance(data, dict):
        raise ConfigLoadError(
            f"Configuration file must contain a YAML mapping, got: {type(data).__name__}"
        )

    return data


def _validate_config(data: dict[str, Any], source: str) -> BenchmarkConfig:
    """Validate raw config data against the pydantic schema.

    Args:
        data: Raw configuration dictionary.
        source: Description of where the config came from (for error messages).

    Returns:
        Validated BenchmarkConfig instance.

    Raises:
        ConfigValidationError: If validation fails.
    """
    try:
        hydrated_data = _hydrate_env_defaults(data)
        return BenchmarkConfig.model_validate(hydrated_data)
    except Exception as validation_error:
        raise ConfigValidationError(
            f"Configuration validation failed ({source}): {validation_error}"
        ) from validation_error


def _hydrate_env_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Populate missing retrieval config from environment defaults."""
    hydrated = deepcopy(data)
    benchmark_section = hydrated.setdefault("benchmark", {})
    retrieval_section = benchmark_section.setdefault("retrieval", {})

    _set_missing(retrieval_section, ["embeddings", "model_name"], "BENCHMARK_EMBEDDING_MODEL")
    _set_missing(
        retrieval_section,
        ["embeddings", "comparison_models"],
        "BENCHMARK_EMBEDDING_MODELS",
    )
    _set_missing(retrieval_section, ["embeddings", "cache_dir"], "BENCHMARK_EMBED_CACHE_DIR")
    _set_missing(benchmark_section, ["reranker", "strategy"], "BENCHMARK_RERANKER_STRATEGY")
    _set_missing(benchmark_section, ["reranker", "model_name"], "BENCHMARK_RERANKER_MODEL")
    _set_missing(benchmark_section, ["reranker", "cache_dir"], "BENCHMARK_RERANKER_CACHE_DIR")
    _set_missing_int(
        benchmark_section,
        ["reranker", "local_size_threshold_mb"],
        "BENCHMARK_RERANKER_LOCAL_SIZE_THRESHOLD_MB",
    )
    _set_missing_csv(
        benchmark_section,
        ["reranker", "api_provider_order"],
        "BENCHMARK_RERANKER_API_PROVIDER_ORDER",
    )
    return hydrated


def _set_missing(section: dict[str, Any], path: list[str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None or value == "":
        return

    target = section
    for key in path[:-1]:
        target = target.setdefault(key, {})
    leaf_key = path[-1]
    if target.get(leaf_key) in (None, ""):
        target[leaf_key] = value


def _set_missing_with_fallbacks(
    section: dict[str, Any],
    path: list[str],
    env_names: list[str],
) -> None:
    for env_name in env_names:
        _set_missing(section, path, env_name)
        target = section
        for key in path[:-1]:
            target = target.setdefault(key, {})
        if target.get(path[-1]) not in (None, ""):
            return


def _set_missing_float(section: dict[str, Any], path: list[str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None or value == "":
        return

    target = section
    for key in path[:-1]:
        target = target.setdefault(key, {})
    leaf_key = path[-1]
    if target.get(leaf_key) in (None, ""):
        target[leaf_key] = float(value)


def _set_missing_int(section: dict[str, Any], path: list[str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None or value == "":
        return

    target = section
    for key in path[:-1]:
        target = target.setdefault(key, {})
    leaf_key = path[-1]
    if target.get(leaf_key) in (None, ""):
        target[leaf_key] = int(value)


def _set_missing_csv(section: dict[str, Any], path: list[str], env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is None or value == "":
        return

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return

    target = section
    for key in path[:-1]:
        target = target.setdefault(key, {})
    leaf_key = path[-1]
    if target.get(leaf_key) in (None, ""):
        target[leaf_key] = items
