"""Unit tests for multi-user filtering and confidence scoring across all memory modules.

Verifies that:
- Each memory module filters reads by user_id.
- Each memory module computes confidence scores on retrieved memories.
- Cross-user isolation is maintained (user A cannot see user B's memories).
- Confidence values are bounded [0.0, 1.0] and reflect relevance + freshness.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext

FIXED_TIMESTAMP = datetime(2024, 1, 15, tzinfo=UTC)


def _event_type_for_store(store: object) -> MemoryType:
    """Return the MemoryType accepted by the given store instance."""
    if isinstance(store, PreferenceStore):
        return MemoryType.PREFERENCE
    if isinstance(store, SemanticStore):
        return MemoryType.SEMANTIC
    if isinstance(store, EntityStore):
        return MemoryType.ENTITY
    return MemoryType.EPISODIC  # EpisodicStore and all short-term stores


def _make_event(
    event_id: str,
    content: str,
    user_id: str = "user-default",
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.8,
    task_id: str = "test",
    entities: list[str] | None = None,
) -> MemoryEvent:
    """Create a test memory event with user_id support."""
    return MemoryEvent(
        id=event_id,
        user_id=user_id,
        type=memory_type,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=importance,
        entities=entities or ["test"],
        task_id=task_id,
    )


def _make_query(
    query_text: str,
    user_id: str = "user-default",
    day: int = 0,
    task_id: str = "test",
    top_k: int = 5,
) -> ReadQuery:
    """Create a test read query with user_id support."""
    return ReadQuery(
        query=query_text,
        top_k=top_k,
        context=ReadQueryContext(dataset_day=day, task_id=task_id, user_id=user_id),
    )


# =============================================================================
# Multi-user filtering: Long-term stores
# =============================================================================


@pytest.mark.unit
class TestEpisodicStoreMultiUser:
    """Multi-user isolation tests for EpisodicStore."""

    def test_user_isolation_on_read(self) -> None:
        """User A's memories are not returned when user B queries."""
        store = EpisodicStore()
        store.write(_make_event("M-001", "Alice prefers Postgres", user_id="user-alice"))
        store.write(_make_event("M-002", "Bob prefers MySQL", user_id="user-bob"))

        response_alice = store.read(_make_query("prefers", user_id="user-alice"))
        response_bob = store.read(_make_query("prefers", user_id="user-bob"))

        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        bob_ids = [m.memory_id for m in response_bob.retrieved_memories]

        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids
        assert "M-002" in bob_ids
        assert "M-001" not in bob_ids

    def test_unknown_user_returns_empty(self) -> None:
        """Querying as an unrecognized user returns no results."""
        store = EpisodicStore()
        store.write(_make_event("M-001", "Some content", user_id="user-alice"))
        response = store.read(_make_query("Some content", user_id="user-unknown"))
        assert response.retrieved_memories == []

    def test_confidence_values_bounded(self) -> None:
        """All confidence values must be in [0.0, 1.0]."""
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test data", user_id="user-alice"), day=0)
        response = store.read(_make_query("Test data", user_id="user-alice", day=5))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0

    def test_confidence_decreases_with_decay(self) -> None:
        """Confidence should decrease as memories decay over time."""
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(
            _make_event("M-001", "Important fact", user_id="user-alice", importance=1.0),
            day=0,
        )
        response_fresh = store.read(_make_query("Important fact", user_id="user-alice", day=0))
        response_stale = store.read(_make_query("Important fact", user_id="user-alice", day=20))

        assert response_fresh.retrieved_memories[0].confidence >= response_stale.retrieved_memories[0].confidence


@pytest.mark.unit
class TestPreferenceStoreMultiUser:
    """Multi-user isolation tests for PreferenceStore."""

    def test_user_isolation_on_read(self) -> None:
        store = PreferenceStore()
        store.write(_make_event("M-001", "Alice likes dark mode", user_id="user-alice", task_id="ui", memory_type=MemoryType.PREFERENCE))
        store.write(_make_event("M-002", "Bob likes light mode", user_id="user-bob", task_id="ui", memory_type=MemoryType.PREFERENCE))

        response_alice = store.read(_make_query("dark mode", user_id="user-alice", task_id="ui"))
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        store = PreferenceStore()
        store.write(_make_event("M-001", "Prefers Python", user_id="user-alice", memory_type=MemoryType.PREFERENCE))
        response = store.read(_make_query("Python", user_id="user-alice"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0


@pytest.mark.unit
class TestSemanticStoreMultiUser:
    """Multi-user isolation tests for SemanticStore."""

    def test_user_isolation_on_read(self) -> None:
        store = SemanticStore()
        store.write(_make_event("M-001", "Python is a language", user_id="user-alice", memory_type=MemoryType.SEMANTIC))
        store.write(_make_event("M-002", "Java is a language", user_id="user-bob", memory_type=MemoryType.SEMANTIC))

        response_alice = store.read(_make_query("language", user_id="user-alice"))
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        store = SemanticStore()
        store.write(_make_event("M-001", "Fact about Postgres", user_id="user-alice", memory_type=MemoryType.SEMANTIC))
        response = store.read(_make_query("Postgres", user_id="user-alice"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0


@pytest.mark.unit
class TestEntityStoreMultiUser:
    """Multi-user isolation tests for EntityStore."""

    def test_user_isolation_on_read(self) -> None:
        store = EntityStore()
        store.write(_make_event("M-001", "Alice knows Carol", user_id="user-alice", entities=["Carol"], memory_type=MemoryType.ENTITY))
        store.write(_make_event("M-002", "Bob knows Dave", user_id="user-bob", entities=["Dave"], memory_type=MemoryType.ENTITY))

        response_alice = store.read(_make_query("Carol", user_id="user-alice"))
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        store = EntityStore()
        store.write(_make_event("M-001", "Alice info", user_id="user-alice", entities=["Alice"], memory_type=MemoryType.ENTITY))
        response = store.read(_make_query("Alice", user_id="user-alice"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0


# =============================================================================
# Multi-user filtering: Short-term stores
# =============================================================================


@pytest.mark.unit
class TestEpisodicBufferMultiUser:
    """Multi-user isolation tests for EpisodicBuffer."""

    def test_user_isolation_on_read(self) -> None:
        buffer = EpisodicBuffer(capacity=50)
        buffer.write(_make_event("M-001", "Alice data point", user_id="user-alice"))
        buffer.write(_make_event("M-002", "Bob data point", user_id="user-bob"))

        response_alice = buffer.read(_make_query("data point", user_id="user-alice"))
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        buffer = EpisodicBuffer(capacity=50)
        buffer.write(_make_event("M-001", "Test content", user_id="user-alice"))
        response = buffer.read(_make_query("Test content", user_id="user-alice"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0

    def test_unknown_user_returns_empty(self) -> None:
        buffer = EpisodicBuffer(capacity=50)
        buffer.write(_make_event("M-001", "Content", user_id="user-alice"))
        response = buffer.read(_make_query("Content", user_id="user-unknown"))
        assert response.retrieved_memories == []


@pytest.mark.unit
class TestContextBufferMultiUser:
    """Multi-user isolation tests for ContextBuffer."""

    def test_user_isolation_within_same_task(self) -> None:
        """Two users with same task_id should not see each other's memories."""
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Alice task data", user_id="user-alice", task_id="shared_task"))
        buffer.write(_make_event("M-002", "Bob task data", user_id="user-bob", task_id="shared_task"))

        response_alice = buffer.read(
            _make_query("task data", user_id="user-alice", task_id="shared_task")
        )
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Context info", user_id="user-alice", task_id="task_a"))
        response = buffer.read(_make_query("Context info", user_id="user-alice", task_id="task_a"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0


@pytest.mark.unit
class TestScratchpadMultiUser:
    """Multi-user isolation tests for Scratchpad."""

    def test_user_isolation_on_read(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Alice scratch note", user_id="user-alice"))
        pad.write(_make_event("M-002", "Bob scratch note", user_id="user-bob"))

        response_alice = pad.read(_make_query("scratch note", user_id="user-alice"))
        alice_ids = [m.memory_id for m in response_alice.retrieved_memories]
        assert "M-001" in alice_ids
        assert "M-002" not in alice_ids

    def test_confidence_present_on_results(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Note content", user_id="user-alice"))
        response = pad.read(_make_query("Note content", user_id="user-alice"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0

    def test_unknown_user_returns_empty(self) -> None:
        pad = Scratchpad()
        pad.write(_make_event("M-001", "Content", user_id="user-alice"))
        response = pad.read(_make_query("Content", user_id="user-unknown"))
        assert response.retrieved_memories == []


# =============================================================================
# Cross-module confidence contract
# =============================================================================


@pytest.mark.unit
class TestConfidenceContract:
    """Verify confidence contract across all modules."""

    @pytest.mark.parametrize(
        "store_factory",
        [
            lambda: EpisodicStore(),
            lambda: PreferenceStore(),
            lambda: SemanticStore(),
            lambda: EntityStore(),
            lambda: EpisodicBuffer(capacity=50),
            lambda: ContextBuffer(),
            lambda: Scratchpad(),
        ],
        ids=[
            "EpisodicStore",
            "PreferenceStore",
            "SemanticStore",
            "EntityStore",
            "EpisodicBuffer",
            "ContextBuffer",
            "Scratchpad",
        ],
    )
    def test_confidence_always_present(self, store_factory: object) -> None:
        """Every retrieved memory must have a confidence field."""
        store = store_factory()  # type: ignore[operator]
        mem_type = _event_type_for_store(store)
        event = _make_event("M-100", "Confidence test content", user_id="user-default", task_id="test", memory_type=mem_type)
        store.write(event)
        response = store.read(_make_query("Confidence test content"))
        for memory in response.retrieved_memories:
            assert hasattr(memory, "confidence")
            assert 0.0 <= memory.confidence <= 1.0

    @pytest.mark.parametrize(
        "store_factory",
        [
            lambda: EpisodicStore(),
            lambda: PreferenceStore(),
            lambda: SemanticStore(),
            lambda: EntityStore(),
            lambda: EpisodicBuffer(capacity=50),
            lambda: ContextBuffer(),
            lambda: Scratchpad(),
        ],
        ids=[
            "EpisodicStore",
            "PreferenceStore",
            "SemanticStore",
            "EntityStore",
            "EpisodicBuffer",
            "ContextBuffer",
            "Scratchpad",
        ],
    )
    def test_user_isolation_contract(self, store_factory: object) -> None:
        """No module should leak user A's memories to user B."""
        store = store_factory()  # type: ignore[operator]
        mem_type = _event_type_for_store(store)
        store.write(_make_event("M-A", "Secret A", user_id="user-a", task_id="test", memory_type=mem_type))
        store.write(_make_event("M-B", "Secret B", user_id="user-b", task_id="test", memory_type=mem_type))

        response_a = store.read(_make_query("Secret", user_id="user-a"))
        response_b = store.read(_make_query("Secret", user_id="user-b"))

        ids_a = {m.memory_id for m in response_a.retrieved_memories}
        ids_b = {m.memory_id for m in response_b.retrieved_memories}

        assert "M-A" in ids_a
        assert "M-B" not in ids_a
        assert "M-B" in ids_b
        assert "M-A" not in ids_b


# =============================================================================
# Multi-user gold dataset integration
# =============================================================================


@pytest.mark.unit
class TestMultiUserGoldDatasetIntegration:
    """Verify that multi-user gold datasets load and validate correctly."""

    def test_delayed_recall_has_multiple_users(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "delayed_recall.json")
        assert len(dataset.user_ids) >= 2
        assert "user-alice" in dataset.user_ids
        assert "user-bob" in dataset.user_ids

    def test_cross_task_interference_has_multiple_users(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "cross_task_interference.json")
        assert len(dataset.user_ids) >= 2

    def test_preference_stability_has_multiple_users(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "preference_stability.json")
        assert len(dataset.user_ids) >= 2

    def test_gold_events_have_user_ids(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "delayed_recall.json")
        for day_events in dataset.events:
            for event in day_events.memory_events:
                assert event.user_id in ("user-alice", "user-bob")

    def test_gold_queries_have_user_ids(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "delayed_recall.json")
        for query in dataset.queries:
            assert query.user_id in ("user-alice", "user-bob")

    def test_total_conversation_turns_positive(self) -> None:
        from pathlib import Path

        from benchmark.gold.oracle import GoldOracle

        datasets_dir = Path(__file__).resolve().parents[2] / "benchmark" / "gold" / "datasets"
        oracle = GoldOracle()
        dataset = oracle.load_dataset(datasets_dir / "delayed_recall.json")
        assert dataset.total_conversation_turns > 0
