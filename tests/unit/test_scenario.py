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

    def test_tail_holdout_only_injects_training_events(self, delayed_recall_dataset: GoldDataset) -> None:
        scenario = GoldDatasetScenario(delayed_recall_dataset, test_frac=0.2)
        split_day = int(scenario.total_days() * 0.8)
        assert scenario.get_events_for_day(split_day) is None
        assert scenario.get_queries_for_day(0) == []

    def test_query_window_only_scores_its_bounded_interval(
        self, delayed_recall_dataset: GoldDataset
    ) -> None:
        scenario = GoldDatasetScenario(
            delayed_recall_dataset,
            test_frac=0.5,
            query_start_fraction=0.2,
            query_end_fraction=0.5,
        )
        assert scenario.get_queries_for_day(0) == []
        assert scenario.get_queries_for_day(3)
        assert scenario.get_queries_for_day(scenario.total_days() - 1) == []

    def test_padded_horizon_keeps_holdout_window_on_data(
        self, delayed_recall_dataset: GoldDataset
    ) -> None:
        # Regression: an evaluation horizon larger than the dataset's day span
        # (e.g. --evaluation-horizon 50 on a 30-day dataset) must not push the
        # holdout query window past every query — that scored 0 queries and
        # recorded recall=0.0 as a successful cell.
        scenario = GoldDatasetScenario(
            delayed_recall_dataset, evaluation_horizon=30, test_frac=0.2
        )
        assert scenario.total_days() == 30  # replay horizon still honoured
        scored = [
            q for day in range(scenario.total_days()) for q in scenario.get_queries_for_day(day)
        ]
        assert scored, "padded horizon must not empty the holdout query window"

    def test_padded_horizon_matches_natural_span_split(
        self, delayed_recall_dataset: GoldDataset
    ) -> None:
        padded = GoldDatasetScenario(
            delayed_recall_dataset, evaluation_horizon=30, test_frac=0.2
        )
        natural = GoldDatasetScenario(delayed_recall_dataset, test_frac=0.2)
        for day in range(padded.total_days()):
            assert [q.query for q in padded.get_queries_for_day(day)] == [
                q.query for q in natural.get_queries_for_day(day)
            ]
            assert (padded.get_events_for_day(day) is None) == (
                natural.get_events_for_day(day) is None
            )
        assert padded.test_frac_applied == natural.test_frac_applied > 0.0

    def test_empty_query_window_logs_warning(
        self, delayed_recall_dataset: GoldDataset, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="benchmark.scenario.loader"):
            GoldDatasetScenario(
                delayed_recall_dataset,
                query_start_fraction=0.01,
                query_end_fraction=0.02,
            )
        assert any("contains none" in r.message for r in caplog.records)

    def test_same_day_gold_holdout_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A dataset whose holdout queries only reference same-day memories can
        # never be scored by the day-based holdout — the loader must say so.
        from benchmark.gold.schema import (
            GoldDayEvents,
            GoldEvaluationCriteria,
            GoldExpectedResult,
            GoldMemoryEvent,
            GoldQuery,
        )

        def _mem(mid: str) -> GoldMemoryEvent:
            return GoldMemoryEvent(
                id=mid, type="episodic", content=f"fact {mid}", importance=0.5, task_id="t"
            )

        dataset = GoldDataset(
            scenario="same_day",
            description="queries land on the same day as their gold memories",
            events=[
                GoldDayEvents(day=d, memory_events=[_mem(f"M-{d}")]) for d in range(10)
            ],
            queries=[
                GoldQuery(
                    day=d,
                    query=f"what is fact M-{d}?",
                    task_id="t",
                    expected=GoldExpectedResult(memory_ids=[f"M-{d}"]),
                )
                for d in range(10)
            ],
            evaluation_criteria=GoldEvaluationCriteria(recall_k=5),
        )
        with caplog.at_level("WARNING", logger="benchmark.scenario.loader"):
            GoldDatasetScenario(dataset, test_frac=0.2)
        assert any("structurally" in r.message for r in caplog.records)
