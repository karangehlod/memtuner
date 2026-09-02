from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmark.application.run_plan import compute_dataset_fingerprint
from benchmark.cli.provenance import (
    build_run_metadata,
    collect_environment_metadata,
    serialize_pack_metadata,
)


@dataclass(frozen=True)
class _PackMetadataStub:
    name: str = "locomo"
    version: str = "1.0"
    description: str = "LoCoMo benchmark pack"
    source_url: str = "https://example.test/locomo"
    license: str = "MIT"
    citation: str = "Example et al."
    total_queries: int = 100
    total_sessions: int = 10
    memory_abilities: tuple[str, ...] = ("episodic", "temporal")


@pytest.mark.unit
def test_collect_environment_metadata_returns_expected_sections() -> None:
    metadata = collect_environment_metadata()

    assert "captured_at" in metadata
    assert metadata["python"]["version"]
    assert metadata["python"]["executable"]
    assert metadata["platform"]["system"]
    assert metadata["process_environment"]["cwd"]
    assert "git" in metadata


@pytest.mark.unit
def test_build_run_metadata_includes_run_plan_dataset_and_resources() -> None:
    run_plan = {
        "dataset": {
            "fingerprint": "abc123def4567890",
            "query_count": 5,
            "memory_count": 12,
            "user_count": 2,
            "event_day_count": 4,
        },
        "config_hash": "cfg-123",
        "seed": 42,
    }
    resource_report = {
        "wall_clock_seconds": 1.25,
        "cpu_peak_percent": 45.0,
    }

    class _ResourceReportStub:
        def to_dict(self) -> dict[str, float]:
            return resource_report

    payload = build_run_metadata(
        command_name="benchmark analyze",
        dataset_path="data/locomo10.json",
        output_dir=Path("analysis_output/sample"),
        pack_name="locomo",
        pack_metadata={"name": "locomo", "version": "1.0"},
        max_queries=5,
        seed=42,
        with_llm_judge=False,
        run_plan=run_plan,
        resource_report=_ResourceReportStub(),
    )

    assert payload["schema_version"] == "1.0"
    assert len(payload["run_hash"]) == 16
    assert payload["command"] == "benchmark analyze"
    assert payload["dataset"]["fingerprint"] == "abc123def4567890"
    assert payload["dataset"]["input_path"] == "data/locomo10.json"
    assert payload["pack"]["name"] == "locomo"
    assert payload["resources"] == resource_report
    assert payload["llm_judge"] == {"enabled": False, "method": None}


@pytest.mark.unit
def test_build_run_metadata_run_hash_is_deterministic_for_same_identity() -> None:
    run_plan = {
        "metric_semantics_version": "2.0",
        "strategy": {
            "requested": "bm25",
            "effective": "bm25",
            "resolved_class": "bm25",
        },
        "memory_modules": ["short_term"],
        "lifecycle_policies": ["keep_all"],
        "dataset": {
            "fingerprint": "abc123def4567890",
            "query_count": 5,
            "memory_count": 12,
            "user_count": 2,
            "event_day_count": 4,
        },
        "evaluation": {"recall_k": 5},
        "horizon": {"requested": None, "effective": 0},
        "normalization": {"applied": False, "delta_days": 0},
        "config_hash": "cfg-123",
        "seed": 42,
    }

    first = build_run_metadata(
        command_name="benchmark analyze",
        dataset_path="data/locomo10.json",
        output_dir=Path("analysis_output/sample-a"),
        pack_name="locomo",
        pack_metadata={"name": "locomo", "version": "1.0"},
        max_queries=5,
        seed=42,
        with_llm_judge=False,
        run_plan=run_plan,
        resource_report=None,
    )
    second = build_run_metadata(
        command_name="benchmark analyze",
        dataset_path="data/locomo10.json",
        output_dir=Path("analysis_output/sample-b"),
        pack_name="locomo",
        pack_metadata={"name": "locomo", "version": "1.0"},
        max_queries=5,
        seed=42,
        with_llm_judge=False,
        run_plan=run_plan,
        resource_report=None,
    )

    assert first["run_hash"] == second["run_hash"]


@pytest.mark.unit
def test_build_run_metadata_run_hash_changes_when_identity_changes() -> None:
    run_plan = {
        "metric_semantics_version": "2.0",
        "strategy": {
            "requested": "bm25",
            "effective": "bm25",
            "resolved_class": "bm25",
        },
        "memory_modules": ["short_term"],
        "lifecycle_policies": ["keep_all"],
        "dataset": {
            "fingerprint": "abc123def4567890",
            "query_count": 5,
            "memory_count": 12,
            "user_count": 2,
            "event_day_count": 4,
        },
        "evaluation": {"recall_k": 5},
        "horizon": {"requested": None, "effective": 0},
        "normalization": {"applied": False, "delta_days": 0},
        "config_hash": "cfg-123",
        "seed": 42,
    }

    base = build_run_metadata(
        command_name="benchmark analyze",
        dataset_path="data/locomo10.json",
        output_dir=Path("analysis_output/sample"),
        pack_name="locomo",
        pack_metadata={"name": "locomo", "version": "1.0"},
        max_queries=5,
        seed=42,
        with_llm_judge=False,
        run_plan=run_plan,
        resource_report=None,
    )
    changed = build_run_metadata(
        command_name="benchmark analyze",
        dataset_path="data/locomo10.json",
        output_dir=Path("analysis_output/sample"),
        pack_name="locomo",
        pack_metadata={"name": "locomo", "version": "1.0"},
        max_queries=10,
        seed=42,
        with_llm_judge=False,
        run_plan=run_plan,
        resource_report=None,
    )

    assert base["run_hash"] != changed["run_hash"]


@pytest.mark.unit
def test_serialize_pack_metadata_returns_json_compatible_shape() -> None:
    serialized = serialize_pack_metadata(_PackMetadataStub())

    assert serialized == {
        "name": "locomo",
        "version": "1.0",
        "description": "LoCoMo benchmark pack",
        "source_url": "https://example.test/locomo",
        "license": "MIT",
        "citation": "Example et al.",
        "total_queries": 100,
        "total_sessions": 10,
        "memory_abilities": ["episodic", "temporal"],
    }


@pytest.mark.unit
def test_compute_dataset_fingerprint_is_deterministic_across_user_order() -> None:
    left = compute_dataset_fingerprint(
        scenario="locomo",
        query_count=5,
        memory_count=12,
        user_ids=["user-b", "user-a"],
    )
    right = compute_dataset_fingerprint(
        scenario="locomo",
        query_count=5,
        memory_count=12,
        user_ids=["user-a", "user-b"],
    )

    assert left == right
    assert len(left) == 16
