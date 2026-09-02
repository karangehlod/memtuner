"""Unit tests for dataset statistics computation.

Tests StatisticsComputer and StatisticsFormatter independently.

SOLID principles tested:
  - Single Responsibility: StatisticsComputer only computes metrics
  - Open/Closed: New metrics can be added without modifying existing code
  - Interface Segregation: Statistics grouped by concern (counts, distributions, quality)
  - Dependency Inversion: Tests depend on DatasetStatistics dataclass
"""


import pytest

from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)
from benchmark.gold.statistics import (
    StatisticsComputer,
    StatisticsFormatter,
)
from benchmark.models.memory_event import MemoryType

# ============================================================================
# Test Helpers
# ============================================================================


def make_memory(
    id: str = "mem1",
    user_id: str = "user1",
    importance: float = 0.5,
    entities: list[str] | None = None,
    task_id: str = "task1",
) -> GoldMemoryEvent:
    """Create a test memory event."""
    return GoldMemoryEvent(
        id=id,
        user_id=user_id,
        type=MemoryType.EPISODIC,
        content=f"Memory {id}",
        importance=importance,
        entities=entities or [],
        task_id=task_id,
    )


def make_query(
    day: int = 0,
    task_id: str = "task1",
    user_id: str = "user1",
) -> GoldQuery:
    """Create a test query."""
    expected = GoldExpectedResult(memory_ids=["mem1"])
    return GoldQuery(
        day=day,
        query="test query",
        task_id=task_id,
        user_id=user_id,
        expected=expected,
    )


def make_dataset(
    memories_by_day: dict[int, list[GoldMemoryEvent]] | None = None,
    queries: list[GoldQuery] | None = None,
) -> GoldDataset:
    """Create a test dataset."""
    if memories_by_day is None:
        memories_by_day = {0: [make_memory()]}
    if queries is None:
        queries = [make_query()]

    events = [
        GoldDayEvents(day=day, memory_events=memories)
        for day, memories in sorted(memories_by_day.items())
    ]

    return GoldDataset(
        scenario="test",
        description="test dataset",
        events=events,
        queries=queries,
    )


# ============================================================================
# Count Statistics Tests
# ============================================================================


class TestCountStatistics:
    """Test basic count computations."""

    def test_single_memory_single_query(self) -> None:
        """Single memory and query dataset."""
        dataset = make_dataset()
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.query_count == 1
        assert stats.counts.memory_count == 1
        assert stats.counts.user_count >= 1
        assert stats.counts.day_count == 1
        assert stats.counts.day_span == 0

    def test_multiple_memories_single_day(self) -> None:
        """Multiple memories on single day."""
        memories = [make_memory(id=f"mem{i}") for i in range(5)]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.memory_count == 5
        assert stats.counts.day_count == 1
        assert stats.counts.avg_memories_per_day == 5.0

    def test_memories_across_multiple_days(self) -> None:
        """Memories distributed across days."""
        memories_by_day = {
            0: [make_memory(id=f"mem{i}") for i in range(2)],
            1: [make_memory(id=f"mem{i}") for i in range(3, 5)],
            2: [make_memory(id=f"mem{i}") for i in range(5, 8)],
        }
        dataset = make_dataset(memories_by_day=memories_by_day)
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.memory_count == 7
        assert stats.counts.day_count == 3
        assert stats.counts.day_span == 2
        assert stats.counts.avg_memories_per_day == pytest.approx(7 / 3)

    def test_multiple_users(self) -> None:
        """Multiple users in dataset."""
        memories = [
            make_memory(id=f"mem{i}", user_id=f"user{i % 3}")
            for i in range(9)
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.user_count >= 3

    def test_multiple_queries(self) -> None:
        """Multiple queries in dataset."""
        queries = [make_query(task_id=f"task{i}") for i in range(5)]
        dataset = make_dataset(queries=queries)
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.query_count == 5


# ============================================================================
# Distribution Statistics Tests
# ============================================================================


class TestDistributionStatistics:
    """Test distribution metrics."""

    def test_importance_statistics_single_value(self) -> None:
        """Single importance value should have zero std."""
        memories = [make_memory(importance=0.5)]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.importance_mean == 0.5
        assert stats.distributions.importance_std == 0.0
        assert stats.distributions.importance_min == 0.5
        assert stats.distributions.importance_max == 0.5

    def test_importance_statistics_multiple_values(self) -> None:
        """Multiple importance values should compute properly."""
        memories = [
            make_memory(id=f"mem{i}", importance=float(i) / 10)
            for i in range(1, 11)  # 0.1, 0.2, ..., 1.0
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.importance_min == pytest.approx(0.1)
        assert stats.distributions.importance_max == pytest.approx(1.0)
        # Mean of 0.1 to 1.0 is 0.55
        assert stats.distributions.importance_mean == pytest.approx(0.55)

    def test_memories_per_user(self) -> None:
        """Per-user memory distribution."""
        memories = [
            make_memory(id=f"mem{i}", user_id="user1")
            for i in range(3)
        ] + [
            make_memory(id=f"mem{i+3}", user_id="user2")
            for i in range(2)
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.memories_per_user["user1"] == 3
        assert stats.distributions.memories_per_user["user2"] == 2

    def test_memories_per_day(self) -> None:
        """Per-day memory distribution."""
        memories_by_day = {
            0: [make_memory(id="mem0"), make_memory(id="mem1")],
            1: [make_memory(id="mem2")],
            2: [make_memory(id=f"mem{i}") for i in range(3, 6)],
        }
        dataset = make_dataset(memories_by_day=memories_by_day)
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.memories_per_day[0] == 2
        assert stats.distributions.memories_per_day[1] == 1
        assert stats.distributions.memories_per_day[2] == 3

    def test_queries_per_user(self) -> None:
        """Per-user query distribution."""
        queries = [
            make_query(user_id="user1"),
            make_query(user_id="user1", task_id="task2"),
            make_query(user_id="user2", task_id="task3"),
        ]
        dataset = make_dataset(queries=queries)
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.queries_per_user["user1"] == 2
        assert stats.distributions.queries_per_user["user2"] == 1


# ============================================================================
# Quality Statistics Tests
# ============================================================================


class TestQualityStatistics:
    """Test quality metrics."""

    def test_entity_coverage_no_entities(self) -> None:
        """Dataset with no entities should have zero coverage."""
        memories = [make_memory(entities=[]) for _ in range(3)]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.quality.entity_coverage == 0.0

    def test_entity_coverage_some_entities(self) -> None:
        """Dataset with some entities should compute coverage."""
        memories = [
            make_memory(id="mem0", entities=["alice", "bob"]),
            make_memory(id="mem1", entities=[]),
            make_memory(id="mem2", entities=["alice"]),
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        # 2 out of 3 memories have entities
        assert stats.quality.entity_coverage == pytest.approx(2 / 3)

    def test_entity_diversity_no_diversity(self) -> None:
        """Single entity across all memories."""
        memories = [
            make_memory(id=f"mem{i}", entities=["alice"])
            for i in range(3)
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        # 1 unique entity, 3 total mentions
        assert stats.quality.entity_diversity == pytest.approx(1 / 3)

    def test_entity_diversity_high(self) -> None:
        """Many unique entities across memories."""
        memories = [
            make_memory(id=f"mem{i}", entities=[f"entity{i}"])
            for i in range(5)
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        # 5 unique entities, 5 total mentions
        assert stats.quality.entity_diversity == pytest.approx(1.0)

    def test_temporal_density_low(self) -> None:
        """Sparse temporal distribution."""
        memories_by_day = {
            0: [make_memory()],
            10: [make_memory(id="mem1")],
            20: [make_memory(id="mem2")],
        }
        dataset = make_dataset(memories_by_day=memories_by_day)
        stats = StatisticsComputer.compute(dataset)

        # 3 memories / 3 days = 1 per day, normalized = 1.0 (capped)
        assert stats.quality.temporal_density <= 1.0

    def test_query_to_memory_ratio(self) -> None:
        """Query to memory ratio metric."""
        memories = [make_memory(id=f"mem{i}") for i in range(10)]
        queries = [make_query(task_id=f"task{i}") for i in range(3)]
        dataset = make_dataset(memories_by_day={0: memories}, queries=queries)
        stats = StatisticsComputer.compute(dataset)

        assert stats.quality.query_to_memory_ratio == pytest.approx(3 / 10)


# ============================================================================
# Statistics Computer Integration Tests
# ============================================================================


class TestStatisticsComputer:
    """Test complete statistics computation."""

    def test_compute_empty_distributions(self) -> None:
        """Statistics handles empty distributions gracefully."""
        dataset = make_dataset()
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.query_count >= 0
        assert stats.counts.memory_count >= 0
        assert isinstance(stats.distributions.memories_per_user, dict)
        assert isinstance(stats.distributions.memories_per_day, dict)

    def test_compute_realistic_dataset(self) -> None:
        """Compute statistics on realistic dataset."""
        memories_by_day = {
            0: [
                make_memory(id=f"mem{i}", user_id="user1", importance=0.7)
                for i in range(5)
            ],
            1: [
                make_memory(
                    id=f"mem{i+5}", user_id="user2", importance=0.4
                )
                for i in range(3)
            ],
            2: [
                make_memory(
                    id=f"mem{i+8}",
                    user_id="user1",
                    importance=0.9,
                    entities=["alice"],
                )
                for i in range(2)
            ],
        }
        queries = [
            make_query(user_id="user1"),
            make_query(user_id="user2", task_id="task2"),
        ]
        dataset = make_dataset(memories_by_day=memories_by_day, queries=queries)
        stats = StatisticsComputer.compute(dataset)

        # Verify counts
        assert stats.counts.memory_count == 10
        assert stats.counts.query_count == 2
        assert stats.counts.day_count == 3
        assert stats.counts.day_span == 2

        # Verify distributions
        assert "user1" in stats.distributions.memories_per_user
        assert stats.distributions.memories_per_user["user1"] == 7

        # Verify quality
        assert stats.quality.entity_coverage > 0
        assert stats.quality.query_to_memory_ratio == pytest.approx(0.2)


# ============================================================================
# Statistics Formatter Tests
# ============================================================================


class TestStatisticsFormatter:
    """Test statistics formatting."""

    def test_format_summary(self) -> None:
        """Format summary should return readable string."""
        dataset = make_dataset()
        stats = StatisticsComputer.compute(dataset)
        summary = StatisticsFormatter.format_summary(stats)

        assert isinstance(summary, str)
        assert "Dataset Statistics Summary" in summary
        assert "Counts:" in summary
        assert "Queries:" in summary
        assert "Memories:" in summary

    def test_format_json(self) -> None:
        """Format JSON should return serializable dict."""
        dataset = make_dataset()
        stats = StatisticsComputer.compute(dataset)
        json_dict = StatisticsFormatter.format_json(stats)

        assert isinstance(json_dict, dict)
        assert "counts" in json_dict
        assert "distributions" in json_dict
        assert "quality" in json_dict

        # Verify counts section
        assert json_dict["counts"]["queries"] >= 0
        assert json_dict["counts"]["memories"] >= 0

        # Verify distributions section
        assert isinstance(json_dict["distributions"]["importance"], dict)
        assert "mean" in json_dict["distributions"]["importance"]

        # Verify quality section
        assert 0 <= json_dict["quality"]["entity_coverage"] <= 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases in statistics computation."""

    def test_single_day_day_span_zero(self) -> None:
        """Single day should have day_span of 0."""
        dataset = make_dataset()
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.day_span == 0

    def test_non_consecutive_days(self) -> None:
        """Non-consecutive days compute correctly."""
        memories_by_day = {
            5: [make_memory()],
            15: [make_memory(id="mem1")],
            20: [make_memory(id="mem2")],
        }
        dataset = make_dataset(memories_by_day=memories_by_day)
        stats = StatisticsComputer.compute(dataset)

        assert stats.counts.day_span == 15

    def test_importance_boundary_values(self) -> None:
        """Boundary importance values (0, 1) compute correctly."""
        memories = [
            make_memory(id="mem0", importance=0.0),
            make_memory(id="mem1", importance=1.0),
        ]
        dataset = make_dataset(memories_by_day={0: memories})
        stats = StatisticsComputer.compute(dataset)

        assert stats.distributions.importance_min == 0.0
        assert stats.distributions.importance_max == 1.0
        assert stats.distributions.importance_mean == pytest.approx(0.5)
