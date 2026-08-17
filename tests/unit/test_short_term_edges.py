"""Unit tests for short-term memory module edge cases."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _make_event(
    event_id: str, content: str, task_id: str = "test", importance: float = 0.8
) -> MemoryEvent:
    return MemoryEvent(
        id=event_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=importance,
        entities=["test"],
        task_id=task_id,
    )


def _make_query(query_text: str, task_id: str = "test", day: int = 0) -> ReadQuery:
    return ReadQuery(
        query=query_text,
        top_k=5,
        context=ReadQueryContext(dataset_day=day, task_id=task_id),
    )


@pytest.mark.unit
class TestContextBufferEdgeCases:
    """Edge case tests for ContextBuffer."""

    def test_read_empty_task(self) -> None:
        buffer = ContextBuffer()
        response = buffer.read(_make_query("anything", task_id="empty"))
        assert len(response.retrieved_memories) == 0
        assert response.total_candidates == 0

    def test_clear_task(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Task A data", task_id="task_a"))
        buffer.write(_make_event("M-002", "Task B data", task_id="task_b"))
        buffer.clear_task("task_a")
        response = buffer.read(_make_query("Task A data", task_id="task_a"))
        assert len(response.retrieved_memories) == 0

    def test_clear_all(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Data", task_id="task_a"))
        buffer.write(_make_event("M-002", "Data", task_id="task_b"))
        buffer.clear()
        assert buffer.read(_make_query("Data", task_id="task_a")).retrieved_memories == []
        assert buffer.read(_make_query("Data", task_id="task_b")).retrieved_memories == []

    def test_task_scoping(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Task A specific", task_id="task_a"))
        buffer.write(_make_event("M-002", "Task B specific", task_id="task_b"))
        response = buffer.read(_make_query("Task A specific", task_id="task_a"))
        ids = [m.memory_id for m in response.retrieved_memories]
        assert "M-001" in ids
        assert "M-002" not in ids


@pytest.mark.unit
class TestEpisodicBufferEdgeCases:
    """Edge case tests for EpisodicBuffer."""

    def test_empty_read(self) -> None:
        buffer = EpisodicBuffer(capacity=10)
        response = buffer.read(_make_query("anything"))
        assert response.retrieved_memories == []

    def test_overwrite_same_id(self) -> None:
        buffer = EpisodicBuffer(capacity=10)
        buffer.write(_make_event("M-001", "Original"))
        buffer.write(_make_event("M-001", "Updated"))
        response = buffer.read(_make_query("Updated"))
        assert len(response.retrieved_memories) >= 1


@pytest.mark.unit
class TestScratchpadEdgeCases:
    """Edge case tests for Scratchpad."""

    def test_overwrite_same_id(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Original content"))
        pad.write(_make_event("M-001", "Updated content"))
        assert pad.count() == 1
        response = pad.read(_make_query("Updated content"))
        assert len(response.retrieved_memories) == 1

    def test_clear_empties(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Data"))
        pad.clear()
        assert pad.count() == 0
