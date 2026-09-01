"""Adapter for generating synthetic benchmark datasets.

Synthetic datasets are procedurally generated with configurable properties.
They enable reproducible testing, controlled experimentation, and stress
testing without depending on external datasets.

Key features:
  - Deterministic: same seed → same dataset, always
  - Configurable: queries, users, days, density, diversity
  - Realistic: plausible distributions and entity relationships
  - Fast: generation is O(n) with minimal I/O
"""

import hashlib
import json
import random
from typing import Any

from benchmark.gold.adapters.adapter import (
    DatasetAdapter,
    FingerprintError,
    StatisticsError,
    ValidationError,
    ValidationReport,
)
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class SyntheticAdapter(DatasetAdapter):
    """Adapter for generating synthetic benchmark datasets.

    Generates reproducible datasets suitable for testing, development,
    and controlled experiments.

    Attributes:
        query_count: Number of queries to generate.
        user_count: Number of distinct users.
        day_range: Time span in simulated days.
        memory_density: Distribution density (low/medium/high).
        query_diversity: Query complexity (low/medium/high).
        seed: Random seed for reproducibility.

    Usage:
        >>> # Small dataset for testing
        >>> adapter = SyntheticAdapter(query_count=50, seed=42)
        >>> dataset = adapter.load()

        >>> # Large dataset for stress testing
        >>> adapter = SyntheticAdapter(query_count=1000, day_range=365, seed=42)
        >>> dataset = adapter.load()
    """

    name = "synthetic"

    # Vocabulary for realistic content generation
    ENTITIES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
        "Grace", "Henry", "Ivy", "Jack",
    ]

    TASKS = [
        "analysis", "report", "meeting", "decision", "review",
        "planning", "design", "implementation", "testing", "deployment",
    ]

    MEMORY_VERBS = [
        "discussed", "decided", "reviewed", "analyzed", "proposed",
        "implemented", "tested", "deployed", "improved", "discovered",
    ]

    # Task-specific detail phrases — give retrieval models real signal to match on
    TASK_DETAILS: dict[str, list[str]] = {
        "analysis":       ["performance bottlenecks", "data quality issues", "usage trends", "error rate spikes"],
        "report":         ["quarterly results", "progress metrics", "findings summary", "status update"],
        "meeting":        ["roadmap priorities", "team blockers", "sprint goals", "stakeholder updates"],
        "decision":       ["architecture choices", "vendor selection", "timeline approval", "scope changes"],
        "review":         ["code quality", "security audit", "design feedback", "documentation gaps"],
        "planning":       ["sprint breakdown", "resource allocation", "milestone targets", "risk assessment"],
        "design":         ["system architecture", "API contracts", "database schema", "UI wireframes"],
        "implementation": ["feature development", "bug fixes", "refactoring work", "integration points"],
        "testing":        ["regression suite", "load testing", "edge case coverage", "acceptance criteria"],
        "deployment":     ["production rollout", "staging validation", "rollback procedure", "service health"],
    }

    def __init__(
        self,
        query_count: int = 100,
        user_count: int = 10,
        day_range: int = 30,
        memory_density: str = "medium",
        query_diversity: str = "high",
        seed: int = 42,
    ):
        """Initialize synthetic adapter.

        Args:
            query_count: Number of queries to generate.
            user_count: Number of distinct users.
            day_range: Simulated time span in days.
            memory_density: Memory distribution ("low", "medium", "high").
            query_diversity: Query difficulty ("low", "medium", "high").
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        if query_count < 1:
            raise ValueError("query_count must be >= 1")
        if user_count < 1:
            raise ValueError("user_count must be >= 1")
        if day_range < 1:
            raise ValueError("day_range must be >= 1")
        if memory_density not in ("low", "medium", "high"):
            raise ValueError("memory_density must be low/medium/high")
        if query_diversity not in ("low", "medium", "high"):
            raise ValueError("query_diversity must be low/medium/high")

        self.query_count = query_count
        self.user_count = user_count
        self.day_range = day_range
        self.memory_density = memory_density
        self.query_diversity = query_diversity
        self.seed = seed

        random.seed(seed)

    def load(self, source: Any = None) -> GoldDataset:
        """Generate synthetic dataset.

        The source parameter is ignored (synthetic generation doesn't need input).

        Args:
            source: Ignored.

        Returns:
            Generated GoldDataset.

        Raises:
            ValidationError: If generation fails.
        """
        try:
            # Compute memory count based on density
            memory_density_factor = {
                "low": 0.5,
                "medium": 1.0,
                "high": 2.0,
            }
            total_memories = int(
                self.query_count * memory_density_factor[self.memory_density]
            )

            # Generate users
            users = [f"user_{i}" for i in range(self.user_count)]

            # Generate memories distributed across days.
            # Track injection day and task per memory ID for query generation.
            all_memories: dict[int, list[GoldMemoryEvent]] = {}
            memory_id_to_day: dict[str, int] = {}
            memory_id_to_task: dict[str, str] = {}

            for mem_idx in range(total_memories):
                day = random.randint(0, self.day_range - 1)
                if day not in all_memories:
                    all_memories[day] = []
                memory = self._generate_memory(
                    mem_idx, day, users, list(memory_id_to_day.keys())
                )
                all_memories[day].append(memory)
                memory_id_to_day[memory.id] = day
                memory_id_to_task[memory.id] = memory.task_id

            # Ensure all days have at least one memory (fill sparse days)
            for day in range(self.day_range):
                if day not in all_memories:
                    memory = self._generate_memory(
                        len(memory_id_to_day), day, users, list(memory_id_to_day.keys())
                    )
                    all_memories[day] = [memory]
                    memory_id_to_day[memory.id] = day
                    memory_id_to_task[memory.id] = memory.task_id

            # Generate queries — each query day >= max injection day of its gold memories
            queries = []
            for q_idx in range(self.query_count):
                query = self._generate_query(q_idx, users, memory_id_to_day, memory_id_to_task)
                queries.append(query)

            # Build events
            events = [
                GoldDayEvents(day=day, memory_events=all_memories[day])
                for day in sorted(all_memories.keys())
            ]

            # Create dataset
            return GoldDataset(
                scenario="Synthetic",
                description=f"Synthetic benchmark dataset (seed={self.seed})",
                user_ids=users,
                events=events,
                queries=queries,
            )

        except Exception as e:
            raise ValidationError(f"Failed to generate synthetic dataset: {e}")

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate synthetic dataset.

        Synthetic datasets always pass basic validation (they're generated correctly).
        This uses the standard registry for consistency.

        Args:
            dataset: Dataset to validate.

        Returns:
            Validation report.
        """
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"Synthetic validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint from seed.

        Same parameters + seed always generate same fingerprint.

        Args:
            dataset: Dataset to fingerprint.

        Returns:
            32-character hex string (SHA256).

        Raises:
            FingerprintError: If fingerprint computation fails.
        """
        try:
            # Fingerprint based on generation parameters, not dataset contents
            fp_data = {
                "adapter": "synthetic",
                "seed": self.seed,
                "query_count": self.query_count,
                "user_count": self.user_count,
                "day_range": self.day_range,
                "memory_density": self.memory_density,
                "query_diversity": self.query_diversity,
            }

            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()

        except Exception as e:
            raise FingerprintError(f"Failed to compute synthetic fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute dataset statistics.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Statistics object.

        Raises:
            StatisticsError: If computation fails.
        """
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute synthetic statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return metadata for synthetic datasets.

        Returns:
            Metadata dictionary.
        """
        return {
            "name": "Synthetic",
            "version": "1.0",
            "description": "Synthetically generated benchmark dataset with configurable properties",
            "source": f"Generated with seed={self.seed}",
            "format": "Procedural generation",
            "reproducible": True,
            "parameters": {
                "query_count": self.query_count,
                "user_count": self.user_count,
                "day_range": self.day_range,
                "memory_density": self.memory_density,
                "query_diversity": self.query_diversity,
            },
        }

    # ========================================================================
    # Private Generation Methods
    # ========================================================================

    def _generate_memory(
        self,
        mem_idx: int,
        day: int,
        users: list[str],
        memory_ids: list[str],
    ) -> GoldMemoryEvent:
        """Generate a synthetic memory event.

        Args:
            mem_idx: Memory index.
            day: Day assignment.
            users: Available users.
            memory_ids: Existing memory IDs (for reference).

        Returns:
            Generated GoldMemoryEvent.
        """
        user_id = random.choice(users)
        task_id = random.choice(self.TASKS)
        verb = random.choice(self.MEMORY_VERBS)
        entities_count = random.randint(0, 3)
        entities = random.sample(self.ENTITIES, min(entities_count, len(self.ENTITIES)))

        # Generate realistic importance distribution (biased toward 0.5-0.8)
        importance = random.gauss(0.6, 0.15)
        importance = max(0.0, min(1.0, importance))

        focus = random.choice(self.TASK_DETAILS.get(task_id, ["general progress"]))
        who = ", ".join(entities) if entities else "the team"
        content = (
            f"{verb.capitalize()} the {task_id} with {who}. "
            f"The session covered {focus}."
        )

        return GoldMemoryEvent(
            id=f"mem_{mem_idx}",
            user_id=user_id,
            type=MemoryType.EPISODIC,
            content=content,
            importance=importance,
            entities=entities,
            task_id=task_id,
            conversation_turn=mem_idx % 10,
        )

    def _generate_query(
        self,
        q_idx: int,
        users: list[str],
        memory_id_to_day: dict[str, int],
        memory_id_to_task: dict[str, str],
    ) -> GoldQuery:
        """Generate a synthetic query with temporally valid gold memories.

        Gold memories are selected from those sharing the query's task so that
        retrieval models have a real semantic signal to match against.

        The query day is always >= the injection day of every gold memory,
        so the memories exist in the store when the query executes.

        Args:
            q_idx: Query index.
            users: Available users.
            memory_id_to_day: Mapping from memory_id → injection day.
            memory_id_to_task: Mapping from memory_id → task_id.

        Returns:
            Generated GoldQuery.
        """
        memory_ids = list(memory_id_to_day.keys())

        # Compute query difficulty based on diversity setting
        num_relevant = {
            "low": 1,
            "medium": random.randint(2, 4),
            "high": random.randint(3, 6),
        }[self.query_diversity]

        # Choose the query's task first, then select gold memories about that task
        # so there is genuine semantic overlap between the query and its answers.
        task = random.choice(self.TASKS)
        task_memories = [mid for mid in memory_ids if memory_id_to_task.get(mid) == task]

        if len(task_memories) >= num_relevant:
            relevant = random.sample(task_memories, num_relevant)
        elif task_memories:
            # Fill remainder from other memories when not enough task matches
            others = [mid for mid in memory_ids if mid not in task_memories]
            n_fill = min(num_relevant - len(task_memories), len(others))
            relevant = task_memories + random.sample(others, n_fill)
        else:
            relevant = random.sample(memory_ids, min(num_relevant, len(memory_ids)))

        # Query day must be >= max injection day of gold memories
        min_query_day = max(memory_id_to_day[mid] for mid in relevant)
        day = random.randint(min_query_day, self.day_range - 1)

        # Generate query text — more templates reduces repetition
        entity = random.choice(self.ENTITIES)
        templates = [
            f"What was decided about the {task} with {entity}?",
            f"When did we complete the {task}?",
            f"Who was involved in the {task}?",
            f"What happened during {entity}'s {task}?",
            f"What was the outcome of the {task}?",
            f"What issues came up during the {task}?",
            f"What did {entity} contribute to the {task}?",
            f"What was covered in the {task} session?",
            f"What did the team decide about the {task}?",
            f"How did the {task} with {entity} go?",
        ]
        query_text = random.choice(templates)
        user_id = random.choice(users)

        expected = GoldExpectedResult(memory_ids=relevant)

        return GoldQuery(
            day=day,
            query=query_text,
            task_id=f"task_{q_idx}",
            user_id=user_id,
            expected=expected,
        )
