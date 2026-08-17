"""Contract tests for MemoryWriter, MemoryReader, and LifecyclePolicy.

Every concrete implementation must pass these tests to prove interface compliance.
Contract tests ensure LSP — any implementation is substitutable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.memory.policies.decay import ExponentialDecayPolicy, LinearDecayPolicy
from benchmark.memory.policies.promotion import ImportanceBasedPromotionPolicy
from benchmark.memory.policies.pruning import (
    CapacityBasedPruningPolicy,
    ScoreThresholdPruningPolicy,
)
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext
from benchmark.models.response import ReadResponse

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _make_event(event_id: str, content: str, task_id: str = "test") -> MemoryEvent:
    """Create a test memory event."""
    return MemoryEvent(
        id=event_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=0.8,
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


# --- MemoryWriter Contract ---


ALL_WRITERS: list[tuple[str, MemoryWriter]] = [
    ("EpisodicBuffer", EpisodicBuffer(capacity=50)),
    ("ContextBuffer", ContextBuffer()),
    ("Scratchpad", Scratchpad()),
    ("EpisodicStore", EpisodicStore()),
    ("PreferenceStore", PreferenceStore()),
    ("SemanticStore", SemanticStore()),
    ("EntityStore", EntityStore()),
]


@pytest.mark.contract
class TestMemoryWriterContract:
    """Contract tests: any MemoryWriter must satisfy these."""

    @pytest.mark.parametrize("name,writer", ALL_WRITERS, ids=[n for n, _ in ALL_WRITERS])
    def test_is_memory_writer(self, name: str, writer: MemoryWriter) -> None:
        assert isinstance(writer, MemoryWriter)

    @pytest.mark.parametrize("name,writer", ALL_WRITERS, ids=[n for n, _ in ALL_WRITERS])
    def test_write_accepts_valid_event(self, name: str, writer: MemoryWriter) -> None:
        event = _make_event("M-100", "Contract test content")
        writer.write(event)  # Must not raise

    @pytest.mark.parametrize("name,writer", ALL_WRITERS, ids=[n for n, _ in ALL_WRITERS])
    def test_write_is_idempotent_on_same_id(self, name: str, writer: MemoryWriter) -> None:
        event = _make_event("M-200", "Same ID event")
        writer.write(event)
        writer.write(event)  # Writing same ID again must not raise


# --- MemoryReader Contract ---


ALL_READERS: list[tuple[str, MemoryReader]] = [
    ("EpisodicBuffer", EpisodicBuffer(capacity=50)),
    ("ContextBuffer", ContextBuffer()),
    ("Scratchpad", Scratchpad()),
    ("EpisodicStore", EpisodicStore()),
    ("PreferenceStore", PreferenceStore()),
    ("SemanticStore", SemanticStore()),
    ("EntityStore", EntityStore()),
]


@pytest.mark.contract
class TestMemoryReaderContract:
    """Contract tests: any MemoryReader must satisfy these."""

    @pytest.mark.parametrize("name,reader", ALL_READERS, ids=[n for n, _ in ALL_READERS])
    def test_is_memory_reader(self, name: str, reader: MemoryReader) -> None:
        assert isinstance(reader, MemoryReader)

    @pytest.mark.parametrize("name,reader", ALL_READERS, ids=[n for n, _ in ALL_READERS])
    def test_read_returns_read_response(self, name: str, reader: MemoryReader) -> None:
        query = _make_query("test query")
        response = reader.read(query)
        assert isinstance(response, ReadResponse)

    @pytest.mark.parametrize("name,reader", ALL_READERS, ids=[n for n, _ in ALL_READERS])
    def test_read_empty_store_returns_empty_results(
        self, name: str, reader: MemoryReader
    ) -> None:
        query = _make_query("nonexistent content")
        response = reader.read(query)
        assert isinstance(response.retrieved_memories, list)
        assert response.latency_ms >= 0.0

    @pytest.mark.parametrize("name,reader", ALL_READERS, ids=[n for n, _ in ALL_READERS])
    def test_read_respects_top_k(self, name: str, reader: MemoryReader) -> None:
        query = ReadQuery(
            query="test",
            top_k=2,
            context=ReadQueryContext(dataset_day=0, task_id="test"),
        )
        response = reader.read(query)
        assert len(response.retrieved_memories) <= 2

    @pytest.mark.parametrize("name,reader", ALL_READERS, ids=[n for n, _ in ALL_READERS])
    def test_scores_monotonic_descending(self, name: str, reader: MemoryReader) -> None:
        # Write some events first (cast to writer if applicable)
        if isinstance(reader, MemoryWriter):
            for i in range(5):
                event = _make_event(f"M-{i}", f"Content number {i}")
                reader.write(event)

        query = _make_query("Content")
        response = reader.read(query)
        scores = [mem.score for mem in response.retrieved_memories]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not monotonic descending: {scores}"
            )


# --- LifecyclePolicy Contract ---


ALL_POLICIES: list[tuple[str, LifecyclePolicy]] = [
    ("ExponentialDecay", ExponentialDecayPolicy(decay_lambda=0.05, threshold=0.35)),
    ("LinearDecay", LinearDecayPolicy(decay_rate=0.05, threshold=0.35)),
    ("ScoreThresholdPruning", ScoreThresholdPruningPolicy(threshold=0.35)),
    ("CapacityBasedPruning", CapacityBasedPruningPolicy(max_capacity=5)),
    ("ImportanceBasedPromotion", ImportanceBasedPromotionPolicy(importance_threshold=0.6)),
]


@pytest.mark.contract
class TestLifecyclePolicyContract:
    """Contract tests: any LifecyclePolicy must satisfy these."""

    @pytest.mark.parametrize("name,policy", ALL_POLICIES, ids=[n for n, _ in ALL_POLICIES])
    def test_is_lifecycle_policy(self, name: str, policy: LifecyclePolicy) -> None:
        assert isinstance(policy, LifecyclePolicy)

    @pytest.mark.parametrize("name,policy", ALL_POLICIES, ids=[n for n, _ in ALL_POLICIES])
    def test_apply_returns_list_of_strings(
        self, name: str, policy: LifecyclePolicy
    ) -> None:
        scores = {"M-001": 0.9, "M-002": 0.3, "M-003": 0.1}
        result = policy.apply(day=10, memory_scores=scores)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    @pytest.mark.parametrize("name,policy", ALL_POLICIES, ids=[n for n, _ in ALL_POLICIES])
    def test_apply_empty_scores_returns_empty(
        self, name: str, policy: LifecyclePolicy
    ) -> None:
        result = policy.apply(day=5, memory_scores={})
        assert result == []

    @pytest.mark.parametrize("name,policy", ALL_POLICIES, ids=[n for n, _ in ALL_POLICIES])
    def test_apply_is_deterministic(
        self, name: str, policy: LifecyclePolicy
    ) -> None:
        scores = {"M-001": 0.9, "M-002": 0.3, "M-003": 0.5, "M-004": 0.1}
        result_a = sorted(policy.apply(day=7, memory_scores=scores))
        result_b = sorted(policy.apply(day=7, memory_scores=scores))
        assert result_a == result_b, "Policy must be deterministic"

    @pytest.mark.parametrize("name,policy", ALL_POLICIES, ids=[n for n, _ in ALL_POLICIES])
    def test_apply_returns_subset_of_input_ids(
        self, name: str, policy: LifecyclePolicy
    ) -> None:
        scores = {"M-001": 0.9, "M-002": 0.3}
        result = policy.apply(day=5, memory_scores=scores)
        assert set(result).issubset(set(scores.keys()))


# --- User Isolation Contract ---


def _make_user_event(
    event_id: str, content: str, user_id: str, task_id: str = "test"
) -> MemoryEvent:
    """Create a memory event with a specific user_id."""
    return MemoryEvent(
        id=event_id,
        user_id=user_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=0.8,
        entities=["test"],
        task_id=task_id,
    )


def _make_user_query(
    query_text: str, user_id: str, task_id: str = "test", day: int = 0
) -> ReadQuery:
    """Create a read query with a specific user_id."""
    return ReadQuery(
        query=query_text,
        top_k=5,
        context=ReadQueryContext(dataset_day=day, task_id=task_id, user_id=user_id),
    )


ALL_READWRITERS: list[tuple[str, MemoryReader]] = [
    ("EpisodicBuffer", EpisodicBuffer(capacity=50)),
    ("ContextBuffer", ContextBuffer()),
    ("Scratchpad", Scratchpad()),
    ("EpisodicStore", EpisodicStore()),
    ("PreferenceStore", PreferenceStore()),
    ("SemanticStore", SemanticStore()),
    ("EntityStore", EntityStore()),
]


@pytest.mark.contract
class TestUserIsolationContract:
    """Contract: every memory module must isolate reads by user_id."""

    @pytest.mark.parametrize(
        "name,module", ALL_READWRITERS, ids=[n for n, _ in ALL_READWRITERS]
    )
    def test_user_a_cannot_see_user_b_memories(
        self, name: str, module: MemoryReader
    ) -> None:
        if isinstance(module, MemoryWriter):
            module.write(_make_user_event("M-UA", "User A secret", "user-a"))
            module.write(_make_user_event("M-UB", "User B secret", "user-b"))

        response_a = module.read(_make_user_query("secret", "user-a"))
        ids_a = {m.memory_id for m in response_a.retrieved_memories}
        assert "M-UB" not in ids_a, f"{name} leaked user-b memory to user-a"


@pytest.mark.contract
class TestConfidenceFieldContract:
    """Contract: every retrieved memory must have a bounded confidence field."""

    @pytest.mark.parametrize(
        "name,module", ALL_READWRITERS, ids=[n for n, _ in ALL_READWRITERS]
    )
    def test_confidence_bounded(self, name: str, module: MemoryReader) -> None:
        if isinstance(module, MemoryWriter):
            module.write(_make_user_event("M-CF", "Confidence check", "user-default"))

        response = module.read(_make_user_query("Confidence check", "user-default"))
        for memory in response.retrieved_memories:
            assert 0.0 <= memory.confidence <= 1.0, (
                f"{name} returned confidence out of bounds: {memory.confidence}"
            )
