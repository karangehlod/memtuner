"""Unit tests for input validation / security hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmark.config.validation import (
    ALLOWED_CONFIG_EXTENSIONS,
    MAX_CONFIG_FILE_SIZE_BYTES,
    sanitize_run_id,
    validate_config_path,
    validate_dataset_path,
    validate_file_path_safe,
    validate_output_directory,
)
from benchmark.exceptions.config_errors import ConfigValidationError
import benchmark.cli.commands.validate_command as validate_module

from benchmark.cli.commands.validate_command import validate_config


@pytest.mark.unit
class TestValidateFilePath:
    """Tests for the validate_file_path_safe function."""

    def test_valid_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("benchmark: {}", encoding="utf-8")
        result = validate_file_path_safe(
            config_file, ALLOWED_CONFIG_EXTENSIONS, MAX_CONFIG_FILE_SIZE_BYTES
        )
        assert result == config_file.resolve()

    def test_disallowed_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.exe"
        bad_file.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Disallowed file extension"):
            validate_file_path_safe(
                bad_file, ALLOWED_CONFIG_EXTENSIONS, MAX_CONFIG_FILE_SIZE_BYTES
            )

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError, match="File not found"):
            validate_file_path_safe(
                tmp_path / "missing.yaml",
                ALLOWED_CONFIG_EXTENSIONS,
                MAX_CONFIG_FILE_SIZE_BYTES,
            )

    def test_directory_instead_of_file_raises(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir.yaml"
        subdir.mkdir()
        with pytest.raises(ConfigValidationError, match="not a regular file"):
            validate_file_path_safe(
                subdir, ALLOWED_CONFIG_EXTENSIONS, MAX_CONFIG_FILE_SIZE_BYTES
            )

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.yaml"
        big_file.write_text("x" * 100, encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="File too large"):
            validate_file_path_safe(
                big_file, ALLOWED_CONFIG_EXTENSIONS, max_size_bytes=50
            )

    def test_path_traversal_detected(self, tmp_path: Path) -> None:
        base = tmp_path / "safe"
        base.mkdir()
        outside = tmp_path / "outside.yaml"
        outside.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Path traversal"):
            validate_file_path_safe(
                outside,
                ALLOWED_CONFIG_EXTENSIONS,
                MAX_CONFIG_FILE_SIZE_BYTES,
                base_directory=base,
            )


@pytest.mark.unit
class TestValidateConfigPath:
    """Tests for the validate_config_path helper."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("benchmark: {}", encoding="utf-8")
        result = validate_config_path(config)
        assert result.exists()

    def test_valid_yml(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text("benchmark: {}", encoding="utf-8")
        result = validate_config_path(config)
        assert result.exists()

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.toml"
        bad.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            validate_config_path(bad)


@pytest.mark.unit
class TestValidateDatasetPath:
    """Tests for the validate_dataset_path helper."""

    def test_valid_json(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset.json"
        dataset.write_text("{}", encoding="utf-8")
        result = validate_dataset_path(dataset)
        assert result.exists()

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "dataset.csv"
        bad.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            validate_dataset_path(bad)


@pytest.mark.unit
class TestValidateOutputDirectory:
    """Tests for the validate_output_directory helper."""

    def test_creates_directory_if_needed(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "new" / "output"
        result = validate_output_directory(output_dir)
        assert result.is_dir()

    def test_existing_directory_passes(self, tmp_path: Path) -> None:
        result = validate_output_directory(tmp_path)
        assert result.is_dir()

    def test_file_instead_of_directory_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="not a directory"):
            validate_output_directory(file_path)


@pytest.mark.unit
class TestSanitizeRunId:
    """Tests for the sanitize_run_id function."""

    def test_valid_run_id(self) -> None:
        assert sanitize_run_id("abc-123_XYZ") == "abc-123_XYZ"

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            sanitize_run_id("")

    def test_too_long_run_id_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="too long"):
            sanitize_run_id("a" * 200)

    def test_invalid_characters_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="invalid characters"):
            sanitize_run_id("run/../../../etc/passwd")

    def test_spaces_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="invalid characters"):
            sanitize_run_id("run with spaces")


@pytest.mark.unit
class TestValidateCommandApiProbe:
    def test_validate_command_reports_api_probe_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import numpy as np

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
benchmark:
  retrieval:
    api_embeddings:
      model_name: text-embedding-3-small
      api_key: config-token
      base_url: https://api.openai.com/v1
      timeout: 12.0
""",
            encoding="utf-8",
        )

        class StubStrategy:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def _embed_texts(self, texts: list[str]):
                assert texts == ["connectivity probe"]
                return [np.array([0.1, 0.2, 0.3], dtype=np.float32)]

        import benchmark.memory.strategies.api_embeddings_strategy as strategy_module

        monkeypatch.setattr(strategy_module, "ApiEmbeddingsStrategy", StubStrategy)

        runner = CliRunner()
        result = runner.invoke(validate_config, ["-c", str(config_path), "--check-api"])

        assert result.exit_code == 0
        assert "API embeddings probe succeeded" in result.output

    def test_validate_command_surfaces_api_probe_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
benchmark:
  retrieval:
    api_embeddings:
      model_name: text-embedding-3-small
      api_key: config-token
      base_url: https://api.openai.com/v1
      timeout: 12.0
""",
            encoding="utf-8",
        )

        class StubStrategy:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def _embed_texts(self, texts: list[str]):
                del texts
                raise RuntimeError("403 Forbidden: invalid API key")

        import benchmark.memory.strategies.api_embeddings_strategy as strategy_module

        monkeypatch.setattr(strategy_module, "ApiEmbeddingsStrategy", StubStrategy)

        runner = CliRunner()
        result = runner.invoke(validate_config, ["-c", str(config_path), "--check-api"])

        assert result.exit_code != 0
        assert "API embeddings probe failed" in result.output
        assert "403 Forbidden" in result.output

    def test_validate_command_rejects_missing_api_base_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark: {}\n", encoding="utf-8")
        monkeypatch.delenv("BENCHMARK_OPENAI_BASE_URL", raising=False)

        runner = CliRunner()
        result = runner.invoke(validate_config, ["-c", str(config_path), "--check-api"])

        assert result.exit_code != 0
        assert "base_url" in result.output


@pytest.mark.unit
class TestValidateCommandEnvironment:
    def test_validate_command_emits_environment_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark: {}\n", encoding="utf-8")
        output_path = tmp_path / "environment-report.json"

        monkeypatch.setattr(
            validate_module,
            "_build_environment_validation_report",
            lambda: {
                "schema_version": "1.0",
                "python": {"version": "3.11.9", "executable": "/usr/bin/python3"},
                "platform": {"system": "Darwin", "release": "24.0", "machine": "arm64"},
                "checks": [{"module": "benchmark.cli.main", "status": "ok"}],
                "status": "ok",
            },
        )

        runner = CliRunner()
        result = runner.invoke(
            validate_config,
            ["-c", str(config_path), "--check-environment", "--environment-output", str(output_path)],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        assert "Environment validation" in result.output
        assert "Report written" in result.output

    def test_validate_command_fails_when_environment_validation_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark: {}\n", encoding="utf-8")

        monkeypatch.setattr(
            validate_module,
            "_build_environment_validation_report",
            lambda: {
                "schema_version": "1.0",
                "python": {"version": "3.14.6", "executable": "/broken/python"},
                "platform": {"system": "Darwin", "release": "24.0", "machine": "arm64"},
                "checks": [
                    {
                        "module": "benchmark.cli.commands.analyze_command",
                        "status": "failed",
                        "error": "ImportError: simulated import failure",
                    }
                ],
                "status": "failed",
            },
        )

        runner = CliRunner()
        result = runner.invoke(validate_config, ["-c", str(config_path), "--check-environment"])

        assert result.exit_code != 0
        assert "Environment validation failed" in result.output
