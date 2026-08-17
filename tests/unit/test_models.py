"""Unit tests for model data classes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.models.answer import TokenUsage
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory
from benchmark.models.run_result import ScenarioMetrics


@pytest.mark.unit
class TestMemoryEvent:
    """Tests for the MemoryEvent model."""

    def test_create_valid_memory_event(self) -> None:
        event = MemoryEvent(
            id="M-001",
            type=MemoryType.EPISODIC,
            content="User prefers Postgres",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            importance=0.85,
            entities=["user", "postgres"],
            task_id="db_selection",
        )
        assert event.id == "M-001"
        assert event.type == MemoryType.EPISODIC
        assert event.importance == 0.85
        assert event.metadata == {}

    def test_memory_event_is_frozen(self) -> None:
        event = MemoryEvent(
            id="M-001",
            type=MemoryType.PREFERENCE,
            content="Test content",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            importance=0.5,
            task_id="test",
        )
        with pytest.raises(Exception):
            event.importance = 0.9  # type: ignore[misc]

    def test_memory_event_rejects_invalid_importance(self) -> None:
        with pytest.raises(Exception):
            MemoryEvent(
                id="M-001",
                type=MemoryType.EPISODIC,
                content="Test",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                importance=1.5,
                task_id="test",
            )

    def test_memory_event_rejects_empty_content(self) -> None:
        with pytest.raises(Exception):
            MemoryEvent(
                id="M-001",
                type=MemoryType.EPISODIC,
                content="",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                importance=0.5,
                task_id="test",
            )

    def test_memory_type_enum_values(self) -> None:
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.ENTITY.value == "entity"


@pytest.mark.unit
class TestReadQuery:
    """Tests for the ReadQuery model."""

    def test_create_valid_read_query(self) -> None:
        query = ReadQuery(
            query="Which database?",
            top_k=5,
            context=ReadQueryContext(dataset_day=7, task_id="db_selection"),
        )
        assert query.query == "Which database?"
        assert query.top_k == 5
        assert query.context.dataset_day == 7

    def test_read_query_default_filters(self) -> None:
        query = ReadQuery(
            query="Test query",
            context=ReadQueryContext(dataset_day=0, task_id="test"),
        )
        assert query.filters.memory_types == []
        assert query.filters.min_importance == 0.0

    def test_read_query_rejects_empty_query(self) -> None:
        with pytest.raises(Exception):
            ReadQuery(
                query="",
                context=ReadQueryContext(dataset_day=0, task_id="test"),
            )


@pytest.mark.unit
class TestReadResponse:
    """Tests for the ReadResponse model."""

    def test_create_empty_response(self) -> None:
        response = ReadResponse(latency_ms=5.0)
        assert response.retrieved_memories == []
        assert response.latency_ms == 5.0
        assert response.total_candidates == 0

    def test_create_response_with_memories(self) -> None:
        memory = RetrievedMemory(
            memory_id="M-001",
            source_module="preference_store",
            score=0.92,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            tier=MemoryTier.HOT,
            decay_factor=0.95,
        )
        response = ReadResponse(
            retrieved_memories=[memory],
            latency_ms=21.0,
            total_candidates=5,
        )
        assert len(response.retrieved_memories) == 1
        assert response.retrieved_memories[0].memory_id == "M-001"


@pytest.mark.unit
class TestTokenUsage:
    """Tests for the TokenUsage model."""

    def test_total_tokens(self) -> None:
        usage = TokenUsage(prompt=820, completion=100)
        assert usage.total == 920

    def test_zero_tokens(self) -> None:
        usage = TokenUsage(prompt=0, completion=0)
        assert usage.total == 0


@pytest.mark.unit
class TestScenarioMetrics:
    """Tests for the ScenarioMetrics model."""

    def test_create_scenario_metrics(self) -> None:
        metrics = ScenarioMetrics(
            scenario_name="delayed_recall",
            recall_at_k=0.8,
            contamination_rate=0.1,
            temporal_accuracy=0.75,
            total_queries=10,
            correct_recalls=8,
        )
        assert metrics.scenario_name == "delayed_recall"
        assert metrics.recall_at_k == 0.8
