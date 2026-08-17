"""Integration tests for generate-gold CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmark.cli.main import cli
from benchmark.gold.schema import GoldDataset


@pytest.mark.integration
def test_generate_gold_basic(tmp_path: Path) -> None:
    """Test basic gold dataset generation via CLI."""
    runner = CliRunner()
    output_file = tmp_path / "test_gold.json"

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "42",
            "--users",
            "5",
            "--days",
            "3",
            "--events-per-day",
            "5",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_file.exists(), "Output file not created"
    assert "Generated" in result.output
    assert "💾 Output written to" in result.output


@pytest.mark.integration
def test_generate_gold_deterministic_replay(tmp_path: Path) -> None:
    """Test that same seed + config produces identical output."""
    runner = CliRunner()
    output_file1 = tmp_path / "gold1.json"
    output_file2 = tmp_path / "gold2.json"

    # First run
    result1 = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "123",
            "--users",
            "3",
            "--days",
            "2",
            "--events-per-day",
            "4",
            "--output",
            str(output_file1),
        ],
    )
    assert result1.exit_code == 0

    # Second run with same parameters
    result2 = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "123",
            "--users",
            "3",
            "--days",
            "2",
            "--events-per-day",
            "4",
            "--output",
            str(output_file2),
        ],
    )
    assert result2.exit_code == 0

    # Both files should be identical
    with output_file1.open() as f1, output_file2.open() as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    assert data1 == data2, "Same seed should produce identical output"


@pytest.mark.integration
def test_generate_gold_schema_validation(tmp_path: Path) -> None:
    """Test that generated output conforms to GoldDataset schema."""
    runner = CliRunner()
    output_file = tmp_path / "validated_gold.json"

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "99",
            "--users",
            "2",
            "--days",
            "1",
            "--events-per-day",
            "3",
            "--output",
            str(output_file),
            "--validate",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "Schema validation passed" in result.output

    # Verify output can be loaded and validated
    with output_file.open() as f:
        data = json.load(f)

    # Should not raise
    dataset = GoldDataset.model_validate(data)
    assert len(dataset.events) == 1
    assert len(dataset.user_ids) == 2
    assert sum(len(d.memory_events) for d in dataset.events) == 3


@pytest.mark.integration
def test_generate_gold_dry_run(tmp_path: Path) -> None:
    """Test dry-run mode doesn't write file."""
    runner = CliRunner()
    output_file = tmp_path / "never_created.json"

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "55",
            "--users",
            "2",
            "--days",
            "1",
            "--events-per-day",
            "2",
            "--output",
            str(output_file),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert not output_file.exists(), "File should not be created in dry-run mode"
    assert "Dry run OK" in result.output


@pytest.mark.integration
def test_generate_gold_creates_parent_dirs(tmp_path: Path) -> None:
    """Test that parent directories are created if missing."""
    runner = CliRunner()
    nested_dir = tmp_path / "a" / "b" / "c"
    output_file = nested_dir / "gold.json"

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "77",
            "--users",
            "2",
            "--days",
            "1",
            "--events-per-day",
            "2",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert nested_dir.exists()


@pytest.mark.integration
def test_generate_gold_default_values(tmp_path: Path) -> None:
    """Test command with only required seed parameter."""
    runner = CliRunner()
    output_file = tmp_path / "defaults_gold.json"

    # Seed is required, others have defaults
    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "100",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    with output_file.open() as f:
        data = json.load(f)

    dataset = GoldDataset.model_validate(data)
    assert len(dataset.user_ids) == 10  # default
    assert len(dataset.events) == 7  # default days


@pytest.mark.integration
def test_generate_gold_missing_required_seed() -> None:
    """Test that missing --seed fails gracefully."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--users",
            "5",
            "--output",
            "/tmp/test.json",
        ],
    )

    assert result.exit_code != 0
    assert "Error" in result.output or "Missing option" in result.output


@pytest.mark.integration
def test_generate_gold_missing_required_output() -> None:
    """Test that missing --output fails gracefully."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "generate-gold",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code != 0
