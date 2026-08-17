"""Tests for determinism and temporal boundaries."""

from __future__ import annotations

import pytest

from benchmark.gold.generator import GoldGenerator, GoldGeneratorConfig
from benchmark.time.simulated_clock import SimulatedClock


@pytest.mark.unit
class TestDeterminismBoundaries:
    """Determinism tests with numeric boundaries."""

    def test_gold_generator_deterministic_replay(self) -> None:
        """Same seed and config should produce identical datasets."""
        config = GoldGeneratorConfig(seed=42, users=5, days=3, events_per_day=4)
        time1 = SimulatedClock()
        time2 = SimulatedClock()

        gen1 = GoldGenerator(config, time1)
        gen2 = GoldGenerator(config, time2)

        dataset1 = gen1.generate()
        dataset2 = gen2.generate()

        # Should be identical
        assert dataset1.scenario == dataset2.scenario
        assert len(dataset1.events) == len(dataset2.events)
        assert len(dataset1.user_ids) == len(dataset2.user_ids)

    def test_gold_generator_deterministic_content(self) -> None:
        """Generated content should be deterministically the same."""
        config = GoldGeneratorConfig(seed=100, users=3, days=2, events_per_day=2)
        time1 = SimulatedClock()
        time2 = SimulatedClock()

        gen1 = GoldGenerator(config, time1)
        gen2 = GoldGenerator(config, time2)

        dataset1 = gen1.generate()
        dataset2 = gen2.generate()

        # Check memory events
        for d1, d2 in zip(dataset1.events, dataset2.events):
            assert d1.day == d2.day
            for m1, m2 in zip(d1.memory_events, d2.memory_events):
                assert m1.id == m2.id
                assert m1.content == m2.content
                assert m1.importance == m2.importance

    def test_gold_generator_different_seeds_different_output(self) -> None:
        """Different seeds should produce different datasets."""
        config1 = GoldGeneratorConfig(seed=1, users=5, days=2, events_per_day=3)
        config2 = GoldGeneratorConfig(seed=2, users=5, days=2, events_per_day=3)

        time = SimulatedClock()
        gen1 = GoldGenerator(config1, time)
        gen2 = GoldGenerator(config2, time)

        dataset1 = gen1.generate()
        dataset2 = gen2.generate()

        # At least one event should differ
        differ = False
        for d1, d2 in zip(dataset1.events, dataset2.events):
            for m1, m2 in zip(d1.memory_events, d2.memory_events):
                if m1.content != m2.content or m1.importance != m2.importance:
                    differ = True
                    break

        assert differ, "Different seeds should produce different content"

    def test_simulated_clock_deterministic(self) -> None:
        """Simulated clock should be deterministic."""
        clock1 = SimulatedClock()
        clock2 = SimulatedClock()

        # Both start at day 0
        assert clock1.current_day() == clock2.current_day()

    def test_gold_generator_large_parameters(self) -> None:
        """Large parameters should not cause overflow."""
        config = GoldGeneratorConfig(
            seed=999, users=100, days=10, events_per_day=50
        )
        time = SimulatedClock()
        gen = GoldGenerator(config, time)
        dataset = gen.generate()

        assert len(dataset.user_ids) == 100
        assert len(dataset.events) == 10
        total_events = sum(len(d.memory_events) for d in dataset.events)
        assert total_events == 500  # 10 days * 50 events

    def test_gold_generator_minimum_parameters(self) -> None:
        """Minimum parameters should work."""
        config = GoldGeneratorConfig(seed=1, users=1, days=1, events_per_day=1)
        time = SimulatedClock()
        gen = GoldGenerator(config, time)
        dataset = gen.generate()

        assert len(dataset.user_ids) == 1
        assert len(dataset.events) == 1
        assert len(dataset.events[0].memory_events) == 1


@pytest.mark.unit
class TestTemporalBoundaryConditions:
    """Temporal boundary conditions and edge cases."""

    def test_temporal_query_at_boundary_day_0(self) -> None:
        """Query on day 0 should work."""
        from benchmark.gold.schema import GoldQuery, GoldExpectedResult, TemporalWindow

        query = GoldQuery(
            day=0,
            query="test",
            task_id="t1",
            user_id="u1",
            expected=GoldExpectedResult(
                memory_ids=["M1"],
                temporal_window=TemporalWindow(not_before_day=0, not_after_day=1),
            ),
        )
        assert query.day == 0

    def test_temporal_window_zero_tolerance(self) -> None:
        """Temporal window with zero tolerance."""
        from benchmark.gold.schema import TemporalWindow

        window = TemporalWindow(not_before_day=5, not_after_day=5)
        assert window.not_before_day == window.not_after_day

    def test_temporal_window_large_range(self) -> None:
        """Temporal window with large range."""
        from benchmark.gold.schema import TemporalWindow

        window = TemporalWindow(not_before_day=0, not_after_day=10_000)
        assert window.not_after_day - window.not_before_day == 10_000

    def test_gold_memory_event_boundaries(self) -> None:
        """Memory event with boundary importance values."""
        from benchmark.gold.schema import GoldMemoryEvent, MemoryType

        # Minimum importance
        mem_min = GoldMemoryEvent(
            id="M1",
            user_id="u1",
            type=MemoryType("preference"),
            content="test",
            importance=0.0,
            task_id="t1",
        )
        assert mem_min.importance == 0.0

        # Maximum importance
        mem_max = GoldMemoryEvent(
            id="M2",
            user_id="u1",
            type=MemoryType("preference"),
            content="test",
            importance=1.0,
            task_id="t1",
        )
        assert mem_max.importance == 1.0


@pytest.mark.unit
class TestNumericPrecision:
    """Numeric precision and float handling."""

    def test_importance_precision(self) -> None:
        """Importance values with various precisions."""
        from benchmark.gold.schema import GoldMemoryEvent, MemoryType

        # Test various precision values
        for importance in [0.1, 0.33, 0.5, 0.99, 0.333333]:
            mem = GoldMemoryEvent(
                id="M1",
                user_id="u1",
                type=MemoryType("preference"),
                content="test",
                importance=importance,
                task_id="t1",
            )
            assert 0.0 <= mem.importance <= 1.0

    def test_recall_float_precision(self) -> None:
        """Recall computation with various float precisions."""
        from benchmark.evaluation.recall import RecallEvaluator

        evaluator = RecallEvaluator(top_k=5)

        # Test case: 1/3 recall
        result = evaluator.evaluate(["M1"], ["M1", "M2", "M3"])
        expected = 1.0 / 3.0
        assert abs(result.value - expected) < 1e-9

    def test_fpr_float_precision(self) -> None:
        """FPR computation with various float precisions."""
        from benchmark.evaluation.false_positive import FalsePositiveEvaluator

        evaluator = FalsePositiveEvaluator()

        # Test case: 1/3 FPR
        result = evaluator.evaluate(["M1", "N1", "N2"], ["M1"])
        expected = 2.0 / 3.0
        assert abs(result.value - expected) < 1e-9
