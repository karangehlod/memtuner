"""Unit tests for GoldOracle edge cases and error paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.exceptions.evaluation_errors import GoldDatasetError
from benchmark.gold.oracle import GoldOracle


@pytest.mark.unit
class TestGoldOracleEdgeCases:
    """Edge case and error handling tests for GoldOracle."""

    def test_load_dataset_file_not_found(self) -> None:
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError, match="file not found"):
            oracle.load_dataset(Path("/nonexistent/dataset.json"))

    def test_load_dataset_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{", encoding="utf-8")
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError, match="Failed to parse"):
            oracle.load_dataset(bad_file)

    def test_load_dataset_not_a_dict(self, tmp_path: Path) -> None:
        # A list of bare integers is neither a LoCoMo conversation array
        # nor a native GoldDataset dict, so loading must raise GoldDatasetError.
        bad_file = tmp_path / "list.json"
        bad_file.write_text("[1, 2, 3]", encoding="utf-8")
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError):
            oracle.load_dataset(bad_file)

    def test_load_dataset_invalid_schema(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "incomplete.json"
        bad_file.write_text(json.dumps({"incomplete": True}), encoding="utf-8")
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError, match="validation failed"):
            oracle.load_dataset(bad_file)

    def test_get_dataset_not_loaded(self) -> None:
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError, match="not loaded"):
            oracle.get_dataset("nonexistent")

    def test_get_expected_result_not_found(self, tmp_path: Path) -> None:
        """Test querying for a query/day combination that doesn't exist."""
        dataset_file = tmp_path / "test.json"
        dataset_file.write_text(
            json.dumps({
                "scenario": "test_scenario",
                "description": "Test scenario",
                "events": [
                    {
                        "day": 0,
                        "memory_events": [
                            {
                                "id": "M-001",
                                "type": "episodic",
                                "content": "Test event",
                                "importance": 0.8,
                                "task_id": "test",
                            }
                        ],
                    }
                ],
                "queries": [
                    {
                        "day": 1,
                        "query": "existing query",
                        "task_id": "test",
                        "expected": {
                            "memory_ids": ["M-001"],
                        },
                    }
                ],
            }),
            encoding="utf-8",
        )
        oracle = GoldOracle()
        oracle.load_dataset(dataset_file)
        with pytest.raises(GoldDatasetError, match="No gold query found"):
            oracle.get_expected_result("test_scenario", "nonexistent query", day=1)

    def test_list_loaded_scenarios_empty(self) -> None:
        oracle = GoldOracle()
        assert oracle.list_loaded_scenarios() == []

    def test_get_queries_for_day_no_matches(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "test.json"
        dataset_file.write_text(
            json.dumps({
                "scenario": "test_scenario",
                "description": "Test scenario",
                "events": [
                    {
                        "day": 0,
                        "memory_events": [
                            {
                                "id": "M-001",
                                "type": "episodic",
                                "content": "Test event",
                                "importance": 0.8,
                                "task_id": "test",
                            }
                        ],
                    }
                ],
                "queries": [
                    {
                        "day": 1,
                        "query": "query on day 1",
                        "task_id": "test",
                        "expected": {
                            "memory_ids": ["M-001"],
                        },
                    }
                ],
            }),
            encoding="utf-8",
        )
        oracle = GoldOracle()
        oracle.load_dataset(dataset_file)
        queries = oracle.get_queries_for_day("test_scenario", day=99)
        assert queries == []

    def test_load_dataset_directory_not_file(self, tmp_path: Path) -> None:
        """Path is a directory, not a file."""
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError):
            oracle.load_dataset(tmp_path)
