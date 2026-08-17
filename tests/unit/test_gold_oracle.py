"""Unit tests for gold oracle and gold dataset schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.exceptions.evaluation_errors import GoldDatasetError
from benchmark.gold.oracle import GoldOracle
from benchmark.gold.schema import GoldDataset

DATASETS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"


@pytest.mark.unit
class TestGoldOracle:
    """Tests for the GoldOracle repository."""

    def test_load_delayed_recall_dataset(self) -> None:
        oracle = GoldOracle()
        dataset = oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        assert isinstance(dataset, GoldDataset)
        assert dataset.scenario == "delayed_recall"

    def test_get_loaded_dataset(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        dataset = oracle.get_dataset("delayed_recall")
        assert dataset.scenario == "delayed_recall"

    def test_get_unloaded_dataset_raises(self) -> None:
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError, match="not loaded"):
            oracle.get_dataset("nonexistent")

    def test_get_expected_result(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        result = oracle.get_expected_result(
            "delayed_recall",
            "Which database did Alice prefer?",
            day=3,
        )
        assert "M-001" in result.memory_ids

    def test_get_expected_result_no_match_raises(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        with pytest.raises(GoldDatasetError, match="No gold query found"):
            oracle.get_expected_result("delayed_recall", "nonexistent query", day=999)

    def test_get_queries_for_day(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        queries = oracle.get_queries_for_day("delayed_recall", day=3)
        assert len(queries) >= 1
        assert queries[0].query == "Which database did Alice prefer?"

    def test_get_queries_for_day_no_queries(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        queries = oracle.get_queries_for_day("delayed_recall", day=1)
        assert queries == []

    def test_list_loaded_scenarios(self) -> None:
        oracle = GoldOracle()
        oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        scenarios = oracle.list_loaded_scenarios()
        assert "delayed_recall" in scenarios

    def test_load_nonexistent_file_raises(self) -> None:
        oracle = GoldOracle()
        with pytest.raises(GoldDatasetError):
            oracle.load_dataset(Path("/nonexistent/file.json"))

    def test_dataset_events_have_memory_events(self) -> None:
        oracle = GoldOracle()
        dataset = oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        assert len(dataset.events) > 0
        for day_events in dataset.events:
            assert len(day_events.memory_events) > 0

    def test_dataset_queries_have_expected_results(self) -> None:
        oracle = GoldOracle()
        dataset = oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")
        for query in dataset.queries:
            assert len(query.expected.memory_ids) > 0
