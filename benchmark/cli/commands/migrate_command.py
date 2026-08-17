"""Migrate command — converts legacy configs to strict schema format.

Reads a YAML config that may contain deprecated or unknown fields,
strips them, applies correct nesting, and writes a valid strict config.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)

# Fields that were silently ignored in the old schema
_LEGACY_BENCHMARK_FIELDS = frozenset(
    {
        "name",
        "description",
        "recall_k",
        "temporal_tolerance_days",
        "gold_dataset",
        "metrics",
    }
)

_LEGACY_OBSERVABILITY_FIELDS = frozenset(
    {
        "enabled",
        "trace_sample_rate",
    }
)

_LEGACY_TOP_LEVEL_FIELDS = frozenset(
    {
        "cost",
    }
)

_LEGACY_MEMORY_FIELDS = frozenset(
    {
        "short_term",
        "long_term",
        "decay",
    }
)


@click.command("migrate-config")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to legacy config YAML.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Path to write migrated config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would change without writing.",
)
def migrate_config(input_path: str, output_path: str, dry_run: bool) -> None:
    """Migrate a legacy config to the strict schema format.

    Strips unknown fields, applies correct nesting (memory.enabled.*),
    and writes a config that passes strict validation.
    """
    source = Path(input_path)
    with source.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    changes: list[str] = []
    migrated = _migrate_config_dict(raw, changes)

    if not changes:
        click.echo("✅ Config is already valid — no migration needed.")
        return

    click.echo(f"🔄 Migration changes ({len(changes)}):")
    for change in changes:
        click.echo(f"   • {change}")

    if dry_run:
        click.echo("\n(dry-run — no file written)")
        return

    # Validate the migrated config before writing
    from benchmark.config.loader import load_config_from_dict

    try:
        load_config_from_dict(migrated)
    except Exception as exc:
        click.echo(f"\n❌ Migrated config still invalid: {exc}", err=True)
        raise SystemExit(1) from exc

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        yaml.dump(migrated, fh, default_flow_style=False, sort_keys=False)

    click.echo(f"\n✅ Migrated config written to {dest}")


def _migrate_config_dict(raw: dict, changes: list[str]) -> dict:
    """Apply migration rules to a raw config dictionary.

    Args:
        raw: The raw YAML dictionary.
        changes: Mutable list that accumulates change descriptions.

    Returns:
        A migrated dictionary compatible with the strict schema.
    """
    migrated: dict = {}

    # --- Migrate memory section ---
    if "memory" in raw:
        memory_raw = raw["memory"]
        if isinstance(memory_raw, dict):
            if "enabled" in memory_raw:
                # Already correct nesting
                migrated["memory"] = {"enabled": memory_raw["enabled"]}
            else:
                # Legacy flat nesting: memory.short_term / memory.long_term
                enabled: dict = {}
                if "short_term" in memory_raw:
                    enabled["short_term"] = memory_raw["short_term"]
                    changes.append("Moved memory.short_term → memory.enabled.short_term")
                if "long_term" in memory_raw:
                    enabled["long_term"] = memory_raw["long_term"]
                    changes.append("Moved memory.long_term → memory.enabled.long_term")
                if enabled:
                    migrated["memory"] = {"enabled": enabled}

            # Remove legacy memory.decay
            if "decay" in memory_raw:
                changes.append("Removed memory.decay (use policies.module_policies instead)")
    else:
        migrated["memory"] = raw.get("memory")

    # --- Migrate benchmark section ---
    if "benchmark" in raw:
        bench_raw = raw["benchmark"]
        if isinstance(bench_raw, dict):
            bench_clean: dict = {}
            for key, value in bench_raw.items():
                if key in _LEGACY_BENCHMARK_FIELDS:
                    changes.append(f"Removed benchmark.{key} (not in strict schema)")
                else:
                    bench_clean[key] = value
            if bench_clean:
                migrated["benchmark"] = bench_clean

    # --- Migrate observability section ---
    if "observability" in raw:
        obs_raw = raw["observability"]
        if isinstance(obs_raw, dict):
            obs_clean: dict = {}
            for key, value in obs_raw.items():
                if key in _LEGACY_OBSERVABILITY_FIELDS:
                    changes.append(f"Removed observability.{key} (not in strict schema)")
                else:
                    obs_clean[key] = value
            # Ensure required fields have defaults
            if "exporter" not in obs_clean:
                obs_clean["exporter"] = "none"
                changes.append("Added observability.exporter = 'none' (required field)")
            if "endpoint" not in obs_clean:
                obs_clean["endpoint"] = "http://localhost:4317"
            if "log_level" not in obs_clean:
                obs_clean["log_level"] = "WARNING"
            migrated["observability"] = obs_clean

    # --- Migrate top-level fields ---
    if "policies" in raw:
        migrated["policies"] = raw["policies"]

    if "answering" in raw:
        migrated["answering"] = raw["answering"]

    # Remove legacy top-level fields
    for field in _LEGACY_TOP_LEVEL_FIELDS:
        if field in raw:
            changes.append(f"Removed top-level '{field}' section (not in strict schema)")

    return migrated
