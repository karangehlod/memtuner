"""Unit tests for memory module implementations."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _make_event(
    event_id: str,
    content: str,
    task_id: str = "test",
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.8,
) -> MemoryEvent:
    """Create a test memory event."""
    return MemoryEvent(
        id=event_id,
        type=memory_type,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=importance,
        entities=["test"],
        task_id=task_id,
    )


def _make_query(query_text: str, task_id: str = "test", day: int = 0) -> ReadQuery:
    """Create a test read query."""
    return ReadQuery(
        query=query_text,
        top_k=5,
        context=ReadQueryContext(dataset_day=day, task_id=task_id),
    )


@pytest.mark.unit
class TestEpisodicBuffer:
    """Tests for the EpisodicBuffer short-term memory."""

    def test_write_and_read_single_event(self) -> None:
        buffer = EpisodicBuffer(capacity=10)
        event = _make_event("M-001", "User likes Postgres")
        buffer.write(event)
        response = buffer.read(_make_query("Postgres"))
        assert len(response.retrieved_memories) == 1
        assert response.retrieved_memories[0].memory_id == "M-001"

    def test_capacity_evicts_oldest(self) -> None:
        buffer = EpisodicBuffer(capacity=2)
        buffer.write(_make_event("M-001", "First event"))
        buffer.write(_make_event("M-002", "Second event"))
        buffer.write(_make_event("M-003", "Third event"))
        # M-001 should be evicted
        response = buffer.read(_make_query("First event"))
        ids = [m.memory_id for m in response.retrieved_memories]
        assert "M-001" not in ids

    def test_relevance_scoring_orders_results(self) -> None:
        buffer = EpisodicBuffer(capacity=10)
        buffer.write(_make_event("M-001", "User prefers Postgres for databases"))
        buffer.write(_make_event("M-002", "Weather is sunny today"))
        response = buffer.read(_make_query("Postgres databases"))
        assert response.retrieved_memories[0].memory_id == "M-001"

    def test_latency_is_nonnegative(self) -> None:
        buffer = EpisodicBuffer(capacity=10)
        response = buffer.read(_make_query("test"))
        assert response.latency_ms >= 0.0


@pytest.mark.unit
class TestContextBuffer:
    """Tests for the ContextBuffer task-scoped short-term memory."""

    def test_write_and_read_task_scoped(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Database discussion", task_id="db"))
        buffer.write(_make_event("M-002", "UI theme discussion", task_id="ui"))

        response = buffer.read(_make_query("Database", task_id="db"))
        ids = [m.memory_id for m in response.retrieved_memories]
        assert "M-001" in ids
        # M-002 should not appear (different task)
        assert "M-002" not in ids

    def test_empty_task_returns_empty(self) -> None:
        buffer = ContextBuffer()
        response = buffer.read(_make_query("anything", task_id="nonexistent"))
        assert response.retrieved_memories == []

    def test_clear_task_removes_memories(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Test content", task_id="db"))
        buffer.clear_task("db")
        response = buffer.read(_make_query("Test", task_id="db"))
        assert response.retrieved_memories == []


@pytest.mark.unit
class TestScratchpad:
    """Tests for the Scratchpad temporary memory."""

    def test_write_and_read(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Temporary note about Postgres"))
        response = pad.read(_make_query("Postgres"))
        assert len(response.retrieved_memories) == 1

    def test_overwrite_same_id(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Original content"))
        pad.write(_make_event("M-001", "Updated content"))
        response = pad.read(_make_query("Updated"))
        assert len(response.retrieved_memories) == 1

    def test_clear_removes_all(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Content one"))
        pad.write(_make_event("M-002", "Content two"))
        pad.clear()
        response = pad.read(_make_query("Content"))
        assert response.retrieved_memories == []


@pytest.mark.unit
class TestEpisodicStore:
    """Tests for the long-term EpisodicStore."""

    def test_write_and_read(self) -> None:
        store = EpisodicStore()
        store.write(_make_event("M-001", "Database migration discussed"))
        response = store.read(_make_query("migration", day=0))
        assert len(response.retrieved_memories) >= 1
        assert response.retrieved_memories[0].memory_id == "M-001"

    def test_decay_reduces_scores_over_time(self) -> None:
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Old memory"), day=0)
        response_day0 = store.read(_make_query("Old memory", day=0))
        response_day10 = store.read(_make_query("Old memory", day=10))
        if response_day0.retrieved_memories and response_day10.retrieved_memories:
            assert response_day10.retrieved_memories[0].score <= response_day0.retrieved_memories[0].score

    def test_write_on_day_records_creation_day(self) -> None:
        store = EpisodicStore()
        store.write_on_day(_make_event("M-001", "Created on day 5"), day=5)
        response = store.read(_make_query("Created on day 5", day=5))
        assert len(response.retrieved_memories) >= 1


@pytest.mark.unit
class TestPreferenceStore:
    """Tests for the long-term PreferenceStore."""

    def test_write_and_read_preference(self) -> None:
        store = PreferenceStore()
        event = _make_event(
            "M-001",
            "User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        store.write(event)
        response = store.read(_make_query("dark mode", day=0))
        assert len(response.retrieved_memories) >= 1

    def test_decay_over_time(self) -> None:
        store = PreferenceStore(decay_lambda=0.1)
        store.write_on_day(
            _make_event("M-001", "User likes Python", memory_type=MemoryType.PREFERENCE),
            day=0,
        )
        resp_early = store.read(_make_query("Python", day=0))
        resp_late = store.read(_make_query("Python", day=20))
        if resp_early.retrieved_memories and resp_late.retrieved_memories:
            assert resp_late.retrieved_memories[0].score <= resp_early.retrieved_memories[0].score
