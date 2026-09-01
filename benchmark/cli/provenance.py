"""Helpers for emitting benchmark protocol provenance artifacts.

These helpers keep `benchmark analyze` focused on orchestration while making
protocol metadata emission consistent and auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.resources.tracker import ResourceReport


def collect_environment_metadata() -> dict[str, Any]:
    """Collect a small, stable snapshot of the execution environment."""
    try:
        git_metadata = {
            "commit": _git_output("rev-parse", "HEAD"),
            "short_commit": _git_output("rev-parse", "--short", "HEAD"),
            "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": _git_is_dirty(),
        }
    except BaseException as exc:
        git_metadata = {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "is_dirty": None,
            "error": f"git metadata unavailable: {exc}",
        }

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "process_environment": {
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
        },
        "git": git_metadata,
    }


def build_run_metadata(
    *,
    command_name: str,
    dataset_path: str | None,
    output_dir: Path,
    pack_name: str | None,
    pack_metadata: dict[str, Any] | None,
    max_queries: int | None,
    seed: int,
    with_llm_judge: bool,
    run_plan: dict[str, Any] | None,
    resource_report: ResourceReport | None,
) -> dict[str, Any]:
    """Build a structured run metadata payload for protocol auditing."""
    dataset_info = dict(run_plan.get("dataset", {})) if run_plan else {}
    dataset_info["input_path"] = dataset_path

    run_identity = {
        "schema_version": "1.0",
        "command": command_name,
        "seed": seed,
        "max_queries": max_queries,
        "dataset": {
            "input_path": dataset_path,
            "fingerprint": dataset_info.get("fingerprint"),
            "query_count": dataset_info.get("query_count"),
            "memory_count": dataset_info.get("memory_count"),
            "user_count": dataset_info.get("user_count"),
            "event_day_count": dataset_info.get("event_day_count"),
        },
        "pack": {
            "name": pack_name,
            "version": (pack_metadata or {}).get("version"),
        },
        "run_plan": {
            "config_hash": (run_plan or {}).get("config_hash"),
            "metric_semantics_version": (run_plan or {}).get("metric_semantics_version"),
            "strategy": (run_plan or {}).get("strategy"),
            "memory_modules": (run_plan or {}).get("memory_modules"),
            "lifecycle_policies": (run_plan or {}).get("lifecycle_policies"),
            "evaluation": (run_plan or {}).get("evaluation"),
            "horizon": (run_plan or {}).get("horizon"),
            "normalization": (run_plan or {}).get("normalization"),
        },
        "llm_judge": {
            "enabled": with_llm_judge,
            "method": "llm_judge" if with_llm_judge else None,
        },
    }

    run_hash = _stable_short_hash(run_identity)

    return {
        "schema_version": "1.0",
        "run_hash": run_hash,
        "command": command_name,
        "captured_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "max_queries": max_queries,
        "llm_judge": {
            "enabled": with_llm_judge,
            "method": "llm_judge" if with_llm_judge else None,
        },
        "output_dir": str(output_dir),
        "pack": {
            "name": pack_name,
            "metadata": pack_metadata,
        }
        if pack_name
        else None,
        "dataset": dataset_info,
        "run_plan": run_plan,
        "resources": resource_report.to_dict() if resource_report is not None else None,
    }


def serialize_pack_metadata(pack_metadata: Any | None) -> dict[str, Any] | None:
    """Convert pack metadata dataclasses into JSON-compatible dictionaries."""
    if pack_metadata is None:
        return None

    return {
        "name": pack_metadata.name,
        "version": pack_metadata.version,
        "description": pack_metadata.description,
        "source_url": pack_metadata.source_url,
        "license": pack_metadata.license,
        "citation": pack_metadata.citation,
        "total_queries": pack_metadata.total_queries,
        "total_sessions": pack_metadata.total_sessions,
        "memory_abilities": list(pack_metadata.memory_abilities),
    }


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _stable_short_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
