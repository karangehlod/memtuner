"""Integration tests for the benchmark migrate-config command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from benchmark.cli.main import cli
from benchmark.config.loader import load_config_from_path

LEGACY_CONFIG = {
    "benchmark": {
        "name": "Legacy Test",
        "seed": 42,
        "evaluation_horizon": 14,
        "retrieval_strategy": "bm25",
        "recall_k": 10,
        "metrics": ["recall_at_k"],
        "gold_dataset": "data/test.json",
        "scenarios": ["delayed_recall"],
    },
    "memory": {
        "short_term": ["episodic_buffer"],
        "long_term": ["episodic_store"],
        "decay": {"type": "exponential", "lambda": 0.05},
    },
    "observability": {"enabled": True, "trace_sample_rate": 1.0},
    "cost": {"enabled": True, "model": "gpt-4o"},
}


@pytest.mark.integration
class TestMigrateCommand:
    """Tests for the migrate-config CLI command."""

    def test_migrates_legacy_config_successfully(self, tmp_path: Path) -> None:
        """A legacy config is migrated and the output passes strict validation."""
        input_path = tmp_path / "legacy.yaml"
        output_path = tmp_path / "migrated.yaml"

        with input_path.open("w") as fh:
            yaml.dump(LEGACY_CONFIG, fh)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["migrate-config", "-i", str(input_path), "-o", str(output_path)]
        )

        assert result.exit_code == 0
        assert "Migration changes" in result.output
        assert output_path.exists()

        # Verify the output passes strict validation
        config = load_config_from_path(output_path)
        assert config.benchmark.seed == 42
        assert config.benchmark.retrieval_strategy == "bm25"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        """Dry-run shows changes but does not create output file."""
        input_path = tmp_path / "legacy.yaml"
        output_path = tmp_path / "migrated.yaml"

        with input_path.open("w") as fh:
            yaml.dump(LEGACY_CONFIG, fh)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["migrate-config", "-i", str(input_path), "-o", str(output_path), "--dry-run"],
        )

        assert result.exit_code == 0
        assert "dry-run" in result.output
        assert not output_path.exists()

    def test_valid_config_reports_no_migration_needed(self, tmp_path: Path) -> None:
        """A config already in strict format reports no changes needed."""
        input_path = tmp_path / "valid.yaml"
        output_path = tmp_path / "output.yaml"

        valid_config = {
            "memory": {"enabled": {"short_term": ["episodic_buffer"], "long_term": ["episodic_store"]}},
            "benchmark": {"evaluation_horizon": 14, "seed": 42, "scenarios": ["delayed_recall"], "retrieval_strategy": "bm25"},
            "observability": {"exporter": "none", "endpoint": "http://localhost:4317", "log_level": "INFO"},
            "answering": {"enabled": False, "model": "gpt-4o", "max_tokens": 500},
        }
        with input_path.open("w") as fh:
            yaml.dump(valid_config, fh)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["migrate-config", "-i", str(input_path), "-o", str(output_path)]
        )

        assert result.exit_code == 0
        assert "already valid" in result.output

    def test_strips_all_known_legacy_fields(self, tmp_path: Path) -> None:
        """All known legacy fields are stripped from the output."""
        input_path = tmp_path / "legacy.yaml"
        output_path = tmp_path / "migrated.yaml"

        with input_path.open("w") as fh:
            yaml.dump(LEGACY_CONFIG, fh)

        runner = CliRunner()
        runner.invoke(
            cli, ["migrate-config", "-i", str(input_path), "-o", str(output_path)]
        )

        with output_path.open() as fh:
            migrated = yaml.safe_load(fh)

        # Legacy fields should be gone
        assert "name" not in migrated.get("benchmark", {})
        assert "recall_k" not in migrated.get("benchmark", {})
        assert "metrics" not in migrated.get("benchmark", {})
        assert "gold_dataset" not in migrated.get("benchmark", {})
        assert "cost" not in migrated
        assert "short_term" not in migrated.get("memory", {})
        assert "long_term" not in migrated.get("memory", {})

        # Correct nesting should exist
        assert "enabled" in migrated.get("memory", {})
        assert "short_term" in migrated["memory"]["enabled"]
