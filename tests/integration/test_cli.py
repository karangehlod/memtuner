"""Integration tests for CLI commands.

Tests CLI commands end-to-end using click's CliRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from benchmark.cli.main import cli
import benchmark.cli.commands.analyze_command as analyze_module

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.integration
class TestCliInit:
    """Tests for the `benchmark init` command."""

    def test_init_creates_default_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_path = tmp_path / "new_config.yaml"
        result = runner.invoke(cli, ["init", "--output", str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()
        assert "Default config written" in result.output

    def test_init_default_path(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0


@pytest.mark.integration
class TestCliValidate:
    """Tests for the `benchmark validate` command."""

    def test_validate_valid_config(self) -> None:
        runner = CliRunner()
        config_path = CONFIGS_DIR / "default.yaml"
        if not config_path.exists():
            pytest.skip("default.yaml not found")
        result = runner.invoke(cli, ["validate", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output

    def test_validate_nonexistent_config(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_validate_help_includes_environment_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--check-environment" in result.output
        assert "--environment-output" in result.output


@pytest.mark.integration
class TestCliAnalyze:
    def test_analyze_emits_phase1_artifact_set_with_stubbed_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text("{}\n", encoding="utf-8")
        output_dir = tmp_path / "analysis"

        class StubGoldOracle:
            def load_dataset(self, _path: Path):
                query = SimpleNamespace(
                    query="Where did we meet?",
                    day=1,
                    task_id="task-1",
                    user_id="user-1",
                    gold_answer="At the cafe.",
                    expected=SimpleNamespace(memory_ids=["mem-1"]),
                )
                event = SimpleNamespace(
                    day=1,
                    memory_events=[
                        SimpleNamespace(
                            id="mem-1",
                            user_id="user-1",
                            type="episodic",
                            content="Met at the cafe.",
                            importance=0.9,
                            entities=["cafe"],
                            task_id="task-1",
                        )
                    ],
                )
                return SimpleNamespace(
                    scenario="locomo",
                    queries=[query],
                    total_conversation_turns=1,
                    user_ids=["user-1"],
                    events=[event],
                )

        class StubComposer:
            def compose(self, config, dataset_override, answer_evaluator=None, allow_strategy_fallback=False, **kwargs):
                del answer_evaluator
                scenario_result = SimpleNamespace(
                    recall_at_k=1.0,
                    precision_at_k=1.0,
                    contamination_rate=0.0,
                    temporal_accuracy=1.0,
                    mrr=1.0,
                    ndcg=1.0,
                    total_queries=1,
                    llm_judge_score=None,
                    llm_judge_queries=0,
                )
                result = SimpleNamespace(scenario_results=[scenario_result])
                runner = SimpleNamespace(run=lambda scenarios: result)
                run_plan = SimpleNamespace(
                    to_dict=lambda: {
                        "config_hash": "cfg-123",
                        "metric_semantics_version": "1.0",
                        "strategy": {"requested": "bm25", "effective": "bm25"},
                        "memory_modules": ["episodic_store"],
                        "lifecycle_policies": ["score_threshold"],
                        "evaluation": {"recall_k": 10},
                        "horizon": {"requested": None, "effective": 30},
                        "normalization": {"applied": False, "delta_days": 0},
                        "dataset": {
                            "fingerprint": "abc123def4567890",
                            "query_count": len(dataset_override.queries),
                            "memory_count": dataset_override.total_conversation_turns,
                            "user_count": len(dataset_override.user_ids),
                            "event_day_count": len(dataset_override.events),
                        },
                    }
                )
                return SimpleNamespace(runner=runner, scenarios=[object()], run_plan=run_plan)

        class StubResourceTracker:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def report(self):
                return SimpleNamespace(to_dict=lambda: {"wall_clock_seconds": 0.1})

        class StubStore:
            def __init__(self, retrieval_strategy=None):
                del retrieval_strategy
                self._count = 0

            def write_on_day(self, mem, day):
                del mem, day
                self._count += 1

            def read(self, query):
                del query
                return SimpleNamespace(retrieved_memories=[SimpleNamespace(memory_id="mem-1")])

            def count(self):
                return self._count

        monkeypatch.setattr(analyze_module, "GoldOracle", StubGoldOracle)
        monkeypatch.setattr(analyze_module, "BenchmarkComposer", StubComposer)
        monkeypatch.setattr(analyze_module, "ResourceTracker", StubResourceTracker)
        monkeypatch.setattr(analyze_module, "EpisodicStore", StubStore)
        monkeypatch.setattr(analyze_module, "EntityStore", StubStore)
        monkeypatch.setattr(analyze_module, "PreferenceStore", StubStore)
        monkeypatch.setattr(analyze_module, "SemanticStore", StubStore)
        monkeypatch.setattr(
            analyze_module,
            "run_interference_test",
            lambda *args, **kwargs: SimpleNamespace(isolation_rate=1.0, leaked_queries=0, total_queries=100),
        )
        monkeypatch.setattr(analyze_module, "find_spec", lambda _name: None)
        monkeypatch.setattr(
            analyze_module,
            "collect_environment_metadata",
            lambda: {"python": {"version": "3.11.9"}, "git": {"commit": "abc123"}},
        )
        monkeypatch.setattr(
            analyze_module,
            "build_run_metadata",
            lambda **kwargs: {
                "schema_version": "1.0",
                "run_hash": "abc123def4567890",
                "command": kwargs["command_name"],
            },
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--max-queries",
                "1",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "benchmark_report.json").exists()
        assert (output_dir / "artifact_manifest.json").exists()
        assert (output_dir / "effective_config.json").exists()
        assert (output_dir / "environment.json").exists()
        assert (output_dir / "run_metadata.json").exists()

        report = json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

        assert report["status"] == "completed"
        assert report["run_metadata_artifact"].endswith("run_metadata.json")
        assert report["environment_artifact"].endswith("environment.json")
        assert report["effective_config_artifact"].endswith("effective_config.json")
        assert report["dataset"]["queries"] == 1
        assert report["dataset"]["memories"] == 1
        assert report["run_plan"]["dataset"]["fingerprint"] == "abc123def4567890"
        assert report["artifact_manifest"] == manifest["artifacts"]
        assert manifest["summary"] == "Tagged images and JSON artifacts emitted by benchmark analyze."
        assert any(item["tag"] == "environment" for item in manifest["artifacts"])
        assert any(item["tag"] == "run_metadata" for item in manifest["artifacts"])
        assert any(item["tag"] == "effective_config" for item in manifest["artifacts"])

    def test_analyze_writes_failed_partial_report_when_runtime_errors_after_dataset_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text("{}\n", encoding="utf-8")
        output_dir = tmp_path / "analysis_failed"

        class StubGoldOracle:
            def load_dataset(self, _path: Path):
                query = SimpleNamespace(
                    query="Where did we meet?",
                    day=1,
                    task_id="task-1",
                    user_id="user-1",
                    gold_answer="At the cafe.",
                    expected=SimpleNamespace(memory_ids=["mem-1"]),
                )
                return SimpleNamespace(
                    scenario="locomo",
                    queries=[query],
                    total_conversation_turns=1,
                    user_ids=["user-1"],
                    events=[],
                )

        class FailingComposer:
            def compose(self, config, dataset_override, answer_evaluator=None, allow_strategy_fallback=False, **kwargs):
                del config, dataset_override, answer_evaluator
                raise RuntimeError("forced compose failure")

        monkeypatch.setattr(analyze_module, "GoldOracle", StubGoldOracle)
        monkeypatch.setattr(analyze_module, "BenchmarkComposer", FailingComposer)
        monkeypatch.setattr(analyze_module, "find_spec", lambda _name: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--max-queries",
                "1",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code != 0
        assert "forced compose failure" in str(result.exception)

        report = json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        assert (output_dir / "environment.json").exists()
        assert (output_dir / "run_metadata.json").exists()
        assert report["status"] == "failed"
        assert report["runtime_error"] == "forced compose failure"
        assert report["dataset"]["queries"] == 1
        assert report["environment_artifact"].endswith("environment.json")
        assert report["run_metadata_artifact"].endswith("run_metadata.json")

    def test_analyze_strategy_allowlist_limits_live_strategy_comparison(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text("{}\n", encoding="utf-8")
        output_dir = tmp_path / "analysis_allowlist"

        class StubGoldOracle:
            def load_dataset(self, _path: Path):
                query = SimpleNamespace(
                    query="Where did we meet?",
                    day=1,
                    task_id="task-1",
                    user_id="user-1",
                    gold_answer="At the cafe.",
                    expected=SimpleNamespace(memory_ids=["mem-1"]),
                )
                event = SimpleNamespace(day=1, memory_events=[])
                return SimpleNamespace(
                    scenario="locomo",
                    queries=[query],
                    total_conversation_turns=1,
                    user_ids=["user-1"],
                    events=[event],
                )

        seen_strategies: list[str] = []

        class StubComposer:
            def compose(self, config, dataset_override, answer_evaluator=None, allow_strategy_fallback=False, **kwargs):
                del dataset_override, answer_evaluator
                seen_strategies.append(config.benchmark.retrieval_strategy)
                scenario_result = SimpleNamespace(
                    recall_at_k=1.0,
                    precision_at_k=1.0,
                    contamination_rate=0.0,
                    temporal_accuracy=1.0,
                    mrr=1.0,
                    ndcg=1.0,
                    total_queries=1,
                    llm_judge_score=None,
                    llm_judge_queries=0,
                )
                result = SimpleNamespace(scenario_results=[scenario_result])
                runner = SimpleNamespace(run=lambda scenarios: result)
                run_plan = SimpleNamespace(
                    to_dict=lambda: {
                        "config_hash": "cfg-allowlist",
                        "metric_semantics_version": "1.0",
                        "strategy": {"requested": "bm25", "effective": "bm25"},
                        "memory_modules": ["episodic_store"],
                        "lifecycle_policies": ["score_threshold"],
                        "evaluation": {"recall_k": 10},
                        "horizon": {"requested": None, "effective": 30},
                        "normalization": {"applied": False, "delta_days": 0},
                        "dataset": {
                            "fingerprint": "abc123def4567890",
                            "query_count": 1,
                            "memory_count": 1,
                            "user_count": 1,
                            "event_day_count": 1,
                        },
                    }
                )
                return SimpleNamespace(runner=runner, scenarios=[object()], run_plan=run_plan)

        class StubResourceTracker:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def report(self):
                return SimpleNamespace(to_dict=lambda: {"wall_clock_seconds": 0.1})

        class StubStore:
            def __init__(self, retrieval_strategy=None):
                del retrieval_strategy
                self._count = 0

            def write_on_day(self, mem, day):
                del mem, day
                self._count += 1

            def read(self, query):
                del query
                return SimpleNamespace(retrieved_memories=[])

            def count(self):
                return self._count

        monkeypatch.setattr(analyze_module, "GoldOracle", StubGoldOracle)
        monkeypatch.setattr(analyze_module, "BenchmarkComposer", StubComposer)
        monkeypatch.setattr(analyze_module, "ResourceTracker", StubResourceTracker)
        monkeypatch.setattr(analyze_module, "EpisodicStore", StubStore)
        monkeypatch.setattr(analyze_module, "EntityStore", StubStore)
        monkeypatch.setattr(analyze_module, "PreferenceStore", StubStore)
        monkeypatch.setattr(analyze_module, "SemanticStore", StubStore)
        monkeypatch.setattr(
            analyze_module,
            "run_interference_test",
            lambda *args, **kwargs: SimpleNamespace(isolation_rate=1.0, leaked_queries=0, total_queries=10),
        )
        monkeypatch.setattr(analyze_module, "find_spec", lambda _name: None)
        monkeypatch.setattr(
            analyze_module,
            "collect_environment_metadata",
            lambda: {"python": {"version": "3.11.9"}, "git": {"commit": "abc123"}},
        )
        monkeypatch.setattr(
            analyze_module,
            "build_run_metadata",
            lambda **kwargs: {
                "schema_version": "1.0",
                "run_hash": "allowlist12345678",
                "command": kwargs["command_name"],
            },
        )
        monkeypatch.setenv("BENCHMARK_ANALYZE_STRATEGIES", "bm25")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_dir),
                "--max-queries",
                "1",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert seen_strategies
        assert set(seen_strategies) == {"bm25"}


@pytest.mark.integration
class TestCliRun:
    """Tests for the `benchmark run` command."""

    def test_run_produces_output(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config_path = CONFIGS_DIR / "default.yaml"
        if not config_path.exists():
            pytest.skip("default.yaml not found")
        output_dir = tmp_path / "results"
        result = runner.invoke(
            cli,
            ["run", "--config", str(config_path), "--output-dir", str(output_dir)],
        )
        assert result.exit_code == 0
        assert "BENCHMARK RESULTS" in result.output
        assert "Results written to" in result.output
        # Verify JSON file was created
        json_files = list(output_dir.glob("run_*.json"))
        assert len(json_files) == 1

    def test_run_nonexistent_config(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", "/nonexistent.yaml"])
        assert result.exit_code != 0


@pytest.mark.integration
class TestCliReport:
    """Tests for the `benchmark report` command."""

    def _create_result_file(self, tmp_path: Path) -> Path:
        """Run a benchmark and return the path to the result JSON."""
        runner = CliRunner()
        config_path = CONFIGS_DIR / "default.yaml"
        output_dir = tmp_path / "run_output"
        runner.invoke(
            cli,
            ["run", "--config", str(config_path), "--output-dir", str(output_dir)],
        )
        json_files = list(output_dir.glob("run_*.json"))
        assert len(json_files) == 1
        return json_files[0]

    def test_report_text_format(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_file = self._create_result_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--input", str(result_file)])
        assert result.exit_code == 0
        assert "BENCHMARK RESULTS" in result.output

    def test_report_json_format(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_file = self._create_result_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["report", "--input", str(result_file), "--format", "json"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output.split("📄")[0] + result.output.split("\n", 1)[1])

    def test_report_csv_format(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_file = self._create_result_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["report", "--input", str(result_file), "--format", "csv"]
        )
        assert result.exit_code == 0
        assert "run_id" in result.output

    def test_report_to_file(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_file = self._create_result_file(tmp_path)
        output_file = tmp_path / "report.txt"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["report", "--input", str(result_file), "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()


@pytest.mark.integration
class TestCliCompare:
    """Tests for the `benchmark compare` command."""

    def _run_two_benchmarks(self, tmp_path: Path) -> list[Path]:
        """Run two benchmarks and return paths to result JSONs."""
        runner = CliRunner()
        config_path = CONFIGS_DIR / "default.yaml"
        results: list[Path] = []
        for i in range(2):
            output_dir = tmp_path / f"run_{i}"
            runner.invoke(
                cli,
                ["run", "--config", str(config_path), "--output-dir", str(output_dir)],
            )
            json_files = list(output_dir.glob("run_*.json"))
            results.append(json_files[0])
        return results

    def test_compare_two_runs(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_files = self._run_two_benchmarks(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["compare", "-r", str(result_files[0]), "-r", str(result_files[1])],
        )
        assert result.exit_code == 0
        assert "recall_at_k" in result.output

    def test_compare_single_run_fails(self, tmp_path: Path) -> None:
        if not (CONFIGS_DIR / "default.yaml").exists():
            pytest.skip("default.yaml not found")
        result_files = self._run_two_benchmarks(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "-r", str(result_files[0])])
        assert result.exit_code != 0


@pytest.mark.integration
class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Agentic Memory Benchmarking Tool" in result.output

    def test_run_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_report_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0

    def test_compare_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.0.1" in result.output
