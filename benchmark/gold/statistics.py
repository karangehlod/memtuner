"""Dataset statistics computation framework.

Computes comprehensive metrics on gold datasets:
  - Counts: queries, memories, users, days
  - Distributions: per-user, per-day, importance
  - Coverage: temporal span, memory density
  - Quality: diversity metrics, entity coverage

SOLID principles:
  - Single Responsibility: StatisticsComputer focuses on metric computation
  - Open/Closed: New metrics can be added without modifying existing code
  - Interface Segregation: Metrics grouped by concern (counts, distributions, etc)
  - Dependency Inversion: Consumers depend on DatasetStatistics dataclass
"""

from dataclasses import dataclass, field
from typing import Any

from benchmark.gold.schema import GoldDataset, GoldDayEvents, GoldMemoryEvent


# ============================================================================
# Statistics Dataclasses
# ============================================================================


@dataclass(frozen=True)
class CountStatistics:
    """Basic count statistics."""

    query_count: int
    """Total number of queries."""

    memory_count: int
    """Total number of memory events."""

    user_count: int
    """Number of distinct users."""

    day_count: int
    """Number of days with events."""

    day_span: int
    """Span from first to last day."""

    avg_memories_per_day: float
    """Average memories per day."""


@dataclass(frozen=True)
class DistributionStatistics:
    """Distribution metrics across dataset."""

    importance_mean: float
    """Mean importance score."""

    importance_std: float
    """Standard deviation of importance."""

    importance_min: float
    """Minimum importance score."""

    importance_max: float
    """Maximum importance score."""

    memories_per_user: dict[str, int] = field(default_factory=dict)
    """Memory count per user."""

    memories_per_day: dict[int, int] = field(default_factory=dict)
    """Memory count per day."""

    queries_per_user: dict[str, int] = field(default_factory=dict)
    """Query count per user."""


@dataclass(frozen=True)
class QualityStatistics:
    """Quality and diversity metrics."""

    entity_coverage: float
    """Fraction of memories with entities."""

    entity_diversity: float
    """Number of unique entities / total entities."""

    temporal_density: float
    """Memories per day on average (0-1 normalized)."""

    query_to_memory_ratio: float
    """Queries / memories ratio."""


@dataclass(frozen=True)
class DatasetStatistics:
    """Complete statistics for a dataset."""

    counts: CountStatistics
    """Basic counts."""

    distributions: DistributionStatistics
    """Distribution metrics."""

    quality: QualityStatistics
    """Quality metrics."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


# ============================================================================
# Statistics Computer
# ============================================================================


class StatisticsComputer:
    """Compute comprehensive statistics on gold datasets.

    Computes all metrics in a single pass for efficiency.
    """

    @staticmethod
    def compute(dataset: GoldDataset) -> DatasetStatistics:
        """Compute all statistics on dataset.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Complete DatasetStatistics object.
        """
        # Initialize accumulators
        all_memories = []
        all_importance = []
        memories_per_user: dict[str, int] = {}
        memories_per_day: dict[int, int] = {}
        queries_per_user: dict[str, int] = {}
        entity_count = 0
        unique_entities = set()

        # Process memories
        for day_events in dataset.events:
            day = day_events.day
            memories_per_day[day] = 0

            for memory in day_events.memory_events:
                all_memories.append(memory)
                all_importance.append(memory.importance)
                memories_per_day[day] += 1

                # User stats
                user_id = memory.user_id
                memories_per_user[user_id] = memories_per_user.get(user_id, 0) + 1

                # Entity coverage
                if memory.entities:
                    entity_count += 1
                    unique_entities.update(memory.entities)

        # Process queries
        for query in dataset.queries:
            user_id = query.user_id
            queries_per_user[user_id] = queries_per_user.get(user_id, 0) + 1

        # Compute counts
        query_count = len(dataset.queries)
        memory_count = len(all_memories)
        user_count = len(memories_per_user) + len(
            set(q.user_id for q in dataset.queries) - set(memories_per_user.keys())
        )
        day_count = len(memories_per_day)

        if dataset.events:
            day_span = dataset.events[-1].day - dataset.events[0].day
        else:
            day_span = 0

        avg_memories_per_day = (
            memory_count / day_count if day_count > 0 else 0.0
        )

        counts = CountStatistics(
            query_count=query_count,
            memory_count=memory_count,
            user_count=user_count,
            day_count=day_count,
            day_span=day_span,
            avg_memories_per_day=avg_memories_per_day,
        )

        # Compute distributions
        importance_mean = (
            sum(all_importance) / len(all_importance)
            if all_importance
            else 0.0
        )

        if all_importance:
            variance = sum(
                (x - importance_mean) ** 2 for x in all_importance
            ) / len(all_importance)
            importance_std = variance ** 0.5
            importance_min = min(all_importance)
            importance_max = max(all_importance)
        else:
            importance_std = 0.0
            importance_min = 0.0
            importance_max = 0.0

        distributions = DistributionStatistics(
            importance_mean=importance_mean,
            importance_std=importance_std,
            importance_min=importance_min,
            importance_max=importance_max,
            memories_per_user=memories_per_user,
            memories_per_day=memories_per_day,
            queries_per_user=queries_per_user,
        )

        # Compute quality
        entity_coverage = (
            entity_count / memory_count if memory_count > 0 else 0.0
        )

        entity_diversity = (
            len(unique_entities) / entity_count if entity_count > 0 else 0.0
        )

        # Normalize temporal density (0-1)
        # Perfect density = 1 memory/day
        temporal_density = min(avg_memories_per_day, 1.0)

        query_to_memory_ratio = (
            query_count / memory_count if memory_count > 0 else 0.0
        )

        quality = QualityStatistics(
            entity_coverage=entity_coverage,
            entity_diversity=entity_diversity,
            temporal_density=temporal_density,
            query_to_memory_ratio=query_to_memory_ratio,
        )

        return DatasetStatistics(
            counts=counts,
            distributions=distributions,
            quality=quality,
        )


# ============================================================================
# Statistics Formatter (for reporting)
# ============================================================================


class StatisticsFormatter:
    """Format statistics for human readability."""

    @staticmethod
    def format_summary(stats: DatasetStatistics) -> str:
        """Format statistics as human-readable summary.

        Args:
            stats: Statistics to format.

        Returns:
            Multi-line formatted summary.
        """
        lines = [
            "Dataset Statistics Summary",
            "=" * 50,
            "",
            "Counts:",
            f"  Queries: {stats.counts.query_count}",
            f"  Memories: {stats.counts.memory_count}",
            f"  Users: {stats.counts.user_count}",
            f"  Days: {stats.counts.day_count} (span: {stats.counts.day_span} days)",
            "",
            "Distributions:",
            f"  Importance: μ={stats.distributions.importance_mean:.2f} σ={stats.distributions.importance_std:.2f}",
            f"  Importance range: [{stats.distributions.importance_min:.2f}, {stats.distributions.importance_max:.2f}]",
            f"  Avg memories/day: {stats.counts.avg_memories_per_day:.1f}",
            "",
            "Quality:",
            f"  Entity coverage: {stats.quality.entity_coverage:.1%}",
            f"  Entity diversity: {stats.quality.entity_diversity:.1%}",
            f"  Temporal density: {stats.quality.temporal_density:.1%}",
            f"  Query/memory ratio: {stats.quality.query_to_memory_ratio:.2f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_json(stats: DatasetStatistics) -> dict[str, Any]:
        """Format statistics as JSON-serializable dict.

        Args:
            stats: Statistics to format.

        Returns:
            Dictionary with all statistics.
        """
        return {
            "counts": {
                "queries": stats.counts.query_count,
                "memories": stats.counts.memory_count,
                "users": stats.counts.user_count,
                "days": stats.counts.day_count,
                "day_span": stats.counts.day_span,
                "avg_memories_per_day": stats.counts.avg_memories_per_day,
            },
            "distributions": {
                "importance": {
                    "mean": stats.distributions.importance_mean,
                    "std": stats.distributions.importance_std,
                    "min": stats.distributions.importance_min,
                    "max": stats.distributions.importance_max,
                },
                "memories_per_user": stats.distributions.memories_per_user,
                "memories_per_day": {
                    str(k): v
                    for k, v in stats.distributions.memories_per_day.items()
                },
                "queries_per_user": stats.distributions.queries_per_user,
            },
            "quality": {
                "entity_coverage": stats.quality.entity_coverage,
                "entity_diversity": stats.quality.entity_diversity,
                "temporal_density": stats.quality.temporal_density,
                "query_to_memory_ratio": stats.quality.query_to_memory_ratio,
            },
        }
