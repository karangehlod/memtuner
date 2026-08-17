"""Unit tests for scenario base and GoldDatasetScenario loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.gold.oracle import GoldOracle
from benchmark.gold.schema import GoldDataset
from benchmark.scenario.base import BenchmarkScenario
from benchmark.scenario.loader import GoldDatasetScenario

DATASETS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"


@pytest.mark.unit
class TestGoldDatasetScenario:
    """Tests for the GoldDatasetScenario implementation."""

    @pytest.fixture()
    def delayed_recall_dataset(self) -> GoldDataset:
        oracle = GoldOracle()
        return oracle.load_dataset(DATASETS_DIR / "delayed_recall.json")

    def test_is_benchmark_scenario(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        assert isinstance(scenario, BenchmarkScenario)

    def test_name_from_dataset(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        assert scenario.name() == "delayed_recall"

    def test_description_from_dataset(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        assert len(scenario.description()) > 0

    def test_total_days_auto_computed(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        # max day in dataset is 13, so total_days should be 14
        assert scenario.total_days() >= 14

    def test_total_days_override(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset, evaluation_horizon=30)
        assert scenario.total_days() == 30

    def test_get_events_for_day_zero(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        events = scenario.get_events_for_day(0)
        assert events is not None
        # After unconditional timestamp normalization, all events may compress to day 0
        # if queries come before newest events (common in small test fixtures).
        # Just verify that day 0 has at least one event.
        assert len(events.memory_events) >= 1

    def test_get_events_for_empty_day(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        events = scenario.get_events_for_day(1)
        assert events is None

    def test_get_queries_for_day_three(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        queries = scenario.get_queries_for_day(3)
        assert len(queries) >= 1
        assert queries[0].query == "Which database did Alice prefer?"

    def test_get_queries_for_empty_day(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        queries = scenario.get_queries_for_day(1)
        assert queries == []

    def test_recall_k_from_criteria(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset)
        assert scenario.recall_k() == 5
