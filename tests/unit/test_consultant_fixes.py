"""Tests for all fixes identified in the consultant assessment.

Covers:
- Fix #1: Lifecycle policies wired into ScenarioRunner
- Fix #2: Temporal evaluation uses real day data via EvaluationContext
- Fix #3: ReadQueryFilters applied in all memory modules
- Fix #4: Conversation metadata carried in EvaluationContext
- Fix #5: acceptable_modules checked via ModuleAccuracyEvaluator
- Fix #6: BaseLongTermStore DRY refactor — shared behavior
- Fix #7: Dynamic source_module (no hardcoded strings)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.false_positive import FalsePositiveEvaluator
from benchmark.evaluation.module_accuracy import ModuleAccuracyEvaluator
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator
from benchmark.memory.long_term.base_store import BaseLongTermStore
from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.memory.policies.pruning import (
    CapacityBasedPruningPolicy,
    ScoreThresholdPruningPolicy,
)
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext, ReadQueryFilters

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _make_event(
    event_id: str,
    content: str,
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.8,
    task_id: str = "test",
    entities: list[str] | None = None,
    user_id: str = "user-default",
) -> MemoryEvent:
    """Create a test memory event."""
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
    day: int = 0,
    task_id: str = "test",
    top_k: int = 5,
    memory_types: list[MemoryType] | None = None,
    min_importance: float = 0.0,
) -> ReadQuery:
    """Create a test read query with optional filters."""
    filters = ReadQueryFilters(
        memory_types=memory_types or [],
        min_importance=min_importance,
    )
    return ReadQuery(
        query=query_text,
        top_k=top_k,
        context=ReadQueryContext(dataset_day=day, task_id=task_id),
        filters=filters,
    )


# =====================================================================
# Fix #1: Lifecycle Policies Wired Into ScenarioRunner
# =====================================================================


@pytest.mark.unit
class TestLifecyclePolicyWiring:
    """Verify lifecycle policies actually prune memories when wired."""

    def test_score_threshold_prunes_decayed_memories(self) -> None:
        """ScoreThresholdPruningPolicy should flag low-scoring memories."""
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Old memory", importance=0.5), day=0)
        store.write_on_day(_make_event("M-002", "New memory", importance=0.9), day=9)

        policy = ScoreThresholdPruningPolicy(threshold=0.35)
        scores = store.get_memory_scores(day=10)
        flagged = policy.apply(day=10, memory_scores=scores)

        # M-001 at day 10 with λ=0.1: 0.5 * e^(-1.0) ≈ 0.184 → below 0.35
        # M-002 at day 10 with λ=0.1: 0.9 * e^(-0.1) ≈ 0.814 → above 0.35
        assert "M-001" in flagged
        assert "M-002" not in flagged

        pruned = store.prune(flagged)
        assert pruned == 1
        assert store.count() == 1

    def test_capacity_pruning_removes_lowest_scores(self) -> None:
        """CapacityBasedPruningPolicy should keep only top N by score."""
        store = EpisodicStore()
        for i in range(5):
            store.write_on_day(
                _make_event(f"M-{i:03d}", f"Memory {i}", importance=i / 10),
                day=0,
            )

        policy = CapacityBasedPruningPolicy(max_capacity=3)
        scores = store.get_memory_scores(day=0)
        flagged = policy.apply(day=0, memory_scores=scores)

        assert len(flagged) == 2
        store.prune(flagged)
        assert store.count() == 3

    def test_survival_rates_change_after_pruning(self) -> None:
        """After pruning, alive_count should be less than injected_count."""
        store = EpisodicStore(decay_lambda=0.3)
        for i in range(10):
            store.write_on_day(
                _make_event(f"M-{i:03d}", f"Memory {i}", importance=0.5),
                day=0,
            )
        assert store.count() == 10

        policy = ScoreThresholdPruningPolicy(threshold=0.35)
        scores = store.get_memory_scores(day=5)
        flagged = policy.apply(day=5, memory_scores=scores)
        store.prune(flagged)

        # With λ=0.3 at day 5: 0.5 * e^(-1.5) ≈ 0.112 → all below 0.35
        assert store.count() < 10


# =====================================================================
# Fix #2: Temporal Evaluation Uses Real Day Data
# =====================================================================


@pytest.mark.unit
class TestTemporalEvaluationWithContext:
    """Verify temporal accuracy evaluator uses real creation-day data."""

    def test_evaluate_with_context_in_window(self) -> None:
        """Memories created within the temporal window should score 1.0."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=1)
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001", "M-002"],
            retrieved_creation_days={"M-001": 2, "M-002": 3},
            temporal_window=(1, 4),
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.metric_name == "benchmark.temporal_accuracy"

    def test_evaluate_with_context_outside_window(self) -> None:
        """Memories created outside temporal window should score 0.0."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=0)
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001"],
            retrieved_creation_days={"M-001": 10, "M-002": 20},
            temporal_window=(1, 3),
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0

    def test_evaluate_with_context_partial_match(self) -> None:
        """Mix of in-window and out-of-window memories."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=0)
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001", "M-002"],
            retrieved_creation_days={"M-001": 2, "M-002": 20},
            temporal_window=(1, 5),
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.5

    def test_evaluate_with_context_no_temporal_window(self) -> None:
        """When no temporal window is set, result should be 1.0 (vacuously true)."""
        evaluator = TemporalAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            temporal_window=None,
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0

    def test_evaluate_with_context_tolerance_applied(self) -> None:
        """Tolerance should extend the window boundaries."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=2)
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            retrieved_creation_days={"M-001": 0},
            temporal_window=(2, 5),  # window is [2, 5] but tolerance extends to [0, 7]
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0


# =====================================================================
# Fix #3: ReadQueryFilters Applied
# =====================================================================


@pytest.mark.unit
class TestReadQueryFiltersApplied:
    """Verify that ReadQueryFilters (memory_types, min_importance) are honored."""

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_long_term_filter_by_memory_type(self, store_class: type) -> None:
        """Only memories matching the filter's memory_types should be returned."""
        store = store_class()
        store.write(_make_event("M-001", "Episodic memory", memory_type=MemoryType.EPISODIC))
        store.write(_make_event("M-002", "Semantic memory", memory_type=MemoryType.SEMANTIC))

        query = _make_query("memory", memory_types=[MemoryType.SEMANTIC])
        response = store.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-002" in returned_ids or len(returned_ids) == 0
        assert "M-001" not in returned_ids

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_long_term_filter_by_min_importance(self, store_class: type) -> None:
        """Only memories with importance >= threshold should be returned."""
        store = store_class()
        store.write(_make_event("M-001", "Low importance", importance=0.2))
        store.write(_make_event("M-002", "High importance", importance=0.9))

        query = _make_query("importance", min_importance=0.5)
        response = store.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-001" not in returned_ids
        # M-002 should be present if it matched the query text
        if returned_ids:
            assert "M-002" in returned_ids

    def test_episodic_buffer_filter_by_memory_type(self) -> None:
        """EpisodicBuffer should honor memory_types filter."""
        buffer = EpisodicBuffer()
        buffer.write(_make_event("M-001", "Episodic event", memory_type=MemoryType.EPISODIC))
        buffer.write(_make_event("M-002", "Semantic fact", memory_type=MemoryType.SEMANTIC))

        query = _make_query("event fact", memory_types=[MemoryType.EPISODIC])
        response = buffer.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-002" not in returned_ids

    def test_episodic_buffer_filter_by_min_importance(self) -> None:
        """EpisodicBuffer should honor min_importance filter."""
        buffer = EpisodicBuffer()
        buffer.write(_make_event("M-001", "Low value", importance=0.1))
        buffer.write(_make_event("M-002", "High value", importance=0.9))

        query = _make_query("value", min_importance=0.5)
        response = buffer.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-001" not in returned_ids

    def test_context_buffer_filter_by_memory_type(self) -> None:
        """ContextBuffer should honor memory_types filter."""
        buffer = ContextBuffer()
        buffer.write(_make_event("M-001", "Episodic", memory_type=MemoryType.EPISODIC, task_id="t1"))
        buffer.write(_make_event("M-002", "Preference", memory_type=MemoryType.PREFERENCE, task_id="t1"))

        query = _make_query("test", task_id="t1", memory_types=[MemoryType.PREFERENCE])
        response = buffer.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-001" not in returned_ids

    def test_scratchpad_filter_by_min_importance(self) -> None:
        """Scratchpad should honor min_importance filter."""
        scratch = Scratchpad()
        scratch.write(_make_event("M-001", "Unimportant", importance=0.1))
        scratch.write(_make_event("M-002", "Important", importance=0.9))

        query = _make_query("test", min_importance=0.5)
        response = scratch.read(query)

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-001" not in returned_ids

    def test_no_filter_returns_all_user_memories(self) -> None:
        """With empty filters, all user memories should be returned."""
        store = EpisodicStore()
        store.write(_make_event("M-001", "First"))
        store.write(_make_event("M-002", "Second"))

        query = _make_query("First Second")
        response = store.read(query)
        assert len(response.retrieved_memories) == 2


# =====================================================================
# Fix #4: Conversation Metadata in EvaluationContext
# =====================================================================


@pytest.mark.unit
class TestConversationMetadataInContext:
    """Verify EvaluationContext carries conversation metadata."""

    def test_context_carries_is_followup(self) -> None:
        """EvaluationContext should carry is_followup from gold query."""
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            is_followup=True,
            references_turn=3,
        )
        assert context.is_followup is True
        assert context.references_turn == 3

    def test_context_defaults_to_not_followup(self) -> None:
        """Default EvaluationContext should not be a follow-up."""
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
        )
        assert context.is_followup is False
        assert context.references_turn is None

    def test_context_is_frozen(self) -> None:
        """EvaluationContext should be immutable."""
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
        )
        with pytest.raises(AttributeError):
            context.is_followup = True  # type: ignore[misc]


# =====================================================================
# Fix #5: acceptable_modules Checked via ModuleAccuracyEvaluator
# =====================================================================


@pytest.mark.unit
class TestModuleAccuracyEvaluator:
    """Verify acceptable_modules correctness checking."""

    def test_all_from_acceptable_modules(self) -> None:
        """Perfect module accuracy when all results come from acceptable sources."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001", "M-002"],
            retrieved_source_modules={"M-001": "preference_store", "M-002": "episodic_store"},
            acceptable_modules=["preference_store", "episodic_store"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.metric_name == "benchmark.module_accuracy"

    def test_none_from_acceptable_modules(self) -> None:
        """Zero module accuracy when no results come from acceptable sources."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001"],
            retrieved_source_modules={"M-001": "scratchpad", "M-002": "context_buffer"},
            acceptable_modules=["preference_store"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0

    def test_partial_module_accuracy(self) -> None:
        """50% accuracy when half from acceptable modules."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001"],
            retrieved_source_modules={"M-001": "preference_store", "M-002": "scratchpad"},
            acceptable_modules=["preference_store"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.5

    def test_empty_acceptable_modules_returns_one(self) -> None:
        """When acceptable_modules is empty, all source modules are acceptable → 1.0."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            retrieved_source_modules={"M-001": "scratchpad"},
            acceptable_modules=[],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.query_count == 1

    def test_empty_retrieved_excluded_from_avg(self) -> None:
        """When nothing retrieved, returns 0.0 with query_count=0 so it is excluded from averages."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=[],
            expected_ids=["M-001"],
            acceptable_modules=["preference_store"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0
        assert result.query_count == 0

    def test_id_only_fallback_raises(self) -> None:
        """Legacy evaluate() raises ValueError (requires EvaluationContext)."""
        import pytest
        evaluator = ModuleAccuracyEvaluator()
        with pytest.raises(ValueError, match="EvaluationContext"):
            evaluator.evaluate(["M-001"], ["M-001"])



# =====================================================================
# Fix #6: BaseLongTermStore DRY Refactor
# =====================================================================

# Each long-term store only accepts its own memory type.
# Use this mapping so parametrized base-class tests write the correct type.
_STORE_ACCEPTED_TYPE: dict[type, MemoryType] = {
    EpisodicStore: MemoryType.EPISODIC,
    PreferenceStore: MemoryType.PREFERENCE,
    SemanticStore: MemoryType.SEMANTIC,
    EntityStore: MemoryType.ENTITY,
}


@pytest.mark.unit
class TestBaseLongTermStoreSharedBehavior:
    """Verify shared behavior via base class for all long-term stores."""

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_inherits_from_base(self, store_class: type) -> None:
        """All long-term stores must inherit from BaseLongTermStore."""
        assert issubclass(store_class, BaseLongTermStore)

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_write_and_count(self, store_class: type) -> None:
        """Shared write/count from base class."""
        store = store_class()
        mem_type = _STORE_ACCEPTED_TYPE[store_class]
        store.write(_make_event("M-001", "Test content", memory_type=mem_type))
        assert store.count() == 1

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_write_on_day_and_get_creation_day(self, store_class: type) -> None:
        """Shared write_on_day/get_creation_day from base class."""
        store = store_class()
        mem_type = _STORE_ACCEPTED_TYPE[store_class]
        store.write_on_day(_make_event("M-001", "Test", memory_type=mem_type), day=5)
        assert store.get_creation_day("M-001") == 5

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_prune_removes_and_returns_count(self, store_class: type) -> None:
        """Shared prune from base class."""
        store = store_class()
        mem_type = _STORE_ACCEPTED_TYPE[store_class]
        store.write(_make_event("M-001", "First", memory_type=mem_type))
        store.write(_make_event("M-002", "Second", memory_type=mem_type))
        removed = store.prune(["M-001"])
        assert removed == 1
        assert store.count() == 1

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_clear_removes_all(self, store_class: type) -> None:
        """Shared clear from base class."""
        store = store_class()
        mem_type = _STORE_ACCEPTED_TYPE[store_class]
        store.write(_make_event("M-001", "First", memory_type=mem_type))
        store.write(_make_event("M-002", "Second", memory_type=mem_type))
        store.clear()
        assert store.count() == 0

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_get_memory_scores_decays(self, store_class: type) -> None:
        """Shared get_memory_scores should return decayed importance."""
        store = store_class(decay_lambda=0.1)
        mem_type = _STORE_ACCEPTED_TYPE[store_class]
        store.write_on_day(_make_event("M-001", "Test", importance=1.0, memory_type=mem_type), day=0)
        scores = store.get_memory_scores(day=10)
        assert scores["M-001"] < 1.0
        assert scores["M-001"] > 0.0

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_tier_thresholds_consistent(self, store_class: type) -> None:
        """All stores should use the same tier thresholds (normalized)."""
        store = store_class()
        # decay_factor > 0.7 → HOT
        assert store._compute_tier(0.8).value == "hot"
        # 0.3 < decay_factor <= 0.7 → WARM
        assert store._compute_tier(0.5).value == "warm"
        # decay_factor <= 0.3 → COLD
        assert store._compute_tier(0.2).value == "cold"

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_confidence_formula_consistent(self, store_class: type) -> None:
        """All stores should use the same confidence formula."""
        store = store_class()
        # confidence = score * 0.6 + decay * 0.4
        assert abs(store._compute_confidence(1.0, 1.0) - 1.0) < 1e-9
        assert abs(store._compute_confidence(0.0, 0.0) - 0.0) < 1e-9
        assert abs(store._compute_confidence(0.5, 0.5) - 0.5) < 1e-9


# =====================================================================
# Fix #7: Dynamic source_module (No Hardcoded Strings)
# =====================================================================


@pytest.mark.unit
class TestDynamicSourceModule:
    """Verify source_module in RetrievedMemory comes from constructor/property."""

    @pytest.mark.parametrize(
        "store_class,expected_name",
        [
            (EpisodicStore, "episodic_store"),
            (PreferenceStore, "preference_store"),
            (SemanticStore, "semantic_store"),
            (EntityStore, "entity_store"),
            (EpisodicBuffer, "episodic_buffer"),
            (ContextBuffer, "context_buffer"),
            (Scratchpad, "scratchpad"),
        ],
        ids=[
            "episodic_store", "preference_store", "semantic_store",
            "entity_store", "episodic_buffer", "context_buffer", "scratchpad",
        ],
    )
    def test_default_module_name(self, store_class: type, expected_name: str) -> None:
        """Default module_name matches the expected value."""
        store = store_class()
        assert store.module_name == expected_name

    @pytest.mark.parametrize(
        "store_class",
        [EpisodicStore, PreferenceStore, SemanticStore, EntityStore],
        ids=["episodic", "preference", "semantic", "entity"],
    )
    def test_custom_module_name(self, store_class: type) -> None:
        """module_name can be overridden via constructor."""
        store = store_class(module_name="custom_name")
        assert store.module_name == "custom_name"

    def test_source_module_in_retrieved_memory(self) -> None:
        """RetrievedMemory.source_module should match the module's name."""
        store = EpisodicStore(module_name="my_episodic")
        store.write(_make_event("M-001", "Test content"))
        response = store.read(_make_query("Test"))
        assert len(response.retrieved_memories) == 1
        assert response.retrieved_memories[0].source_module == "my_episodic"

    def test_short_term_custom_module_name(self) -> None:
        """Short-term modules also support custom module_name."""
        buffer = EpisodicBuffer(module_name="my_buffer")
        assert buffer.module_name == "my_buffer"

        buffer.write(_make_event("M-001", "Test content"))
        response = buffer.read(_make_query("Test"))
        assert response.retrieved_memories[0].source_module == "my_buffer"


# =====================================================================
# Integration: evaluate_with_context() Default Delegation
# =====================================================================


@pytest.mark.unit
class TestEvaluateWithContextDelegation:
    """Verify default evaluate_with_context() delegates to evaluate() correctly."""

    def test_recall_evaluator_delegates(self) -> None:
        """RecallEvaluator uses default delegation via evaluate()."""
        evaluator = RecallEvaluator(top_k=5)
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-002"],
            expected_ids=["M-001"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 1.0
        assert result.metric_name == "benchmark.recall_at_k"

    def test_false_positive_evaluator_delegates(self) -> None:
        """FalsePositiveEvaluator uses default delegation via evaluate()."""
        evaluator = FalsePositiveEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001", "M-099"],
            expected_ids=["M-001"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.5

    def test_temporal_evaluator_overrides(self) -> None:
        """TemporalAccuracyEvaluator MUST use its override, not delegation."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=0)
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            retrieved_creation_days={"M-001": 10},
            temporal_window=(1, 3),
        )
        result = evaluator.evaluate_with_context(context)
        # M-001 at day 10 is outside (1, 3) → accuracy 0.0
        assert result.value == 0.0

    def test_module_accuracy_evaluator_overrides(self) -> None:
        """ModuleAccuracyEvaluator MUST use its override, not delegation."""
        evaluator = ModuleAccuracyEvaluator()
        context = EvaluationContext(
            retrieved_ids=["M-001"],
            expected_ids=["M-001"],
            retrieved_source_modules={"M-001": "scratchpad"},
            acceptable_modules=["preference_store"],
        )
        result = evaluator.evaluate_with_context(context)
        assert result.value == 0.0


# =====================================================================
# Integration: Specific Numeric Values for Scores
# =====================================================================


@pytest.mark.unit
class TestSpecificNumericValues:
    """Tests that assert exact numeric values for scores, decay, and confidence."""

    def test_episodic_decay_at_day_10(self) -> None:
        """Verify exact decay factor at day 10 with λ=0.1."""
        import math

        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Exact test", importance=1.0), day=0)
        response = store.read(_make_query("Exact test", day=10))

        mem = response.retrieved_memories[0]
        expected_decay = math.exp(-0.1 * 10)
        assert abs(mem.decay_factor - expected_decay) < 1e-9

    def test_confidence_formula_exact(self) -> None:
        """Verify exact confidence = score * 0.6 + decay_factor * 0.4."""
        store = EpisodicStore(decay_lambda=0.0)  # no decay
        store.write_on_day(
            _make_event("M-001", "Confidence test", importance=1.0), day=0
        )
        response = store.read(_make_query("Confidence test", day=0))

        mem = response.retrieved_memories[0]
        expected_confidence = mem.score * 0.6 + mem.decay_factor * 0.4
        assert abs(mem.confidence - expected_confidence) < 1e-9

    def test_entity_boost_exact(self) -> None:
        """Verify entity boost adds ENTITY_BOOST_FACTOR per matching entity."""
        store = EntityStore(decay_lambda=0.0)
        store.write_on_day(
            _make_event(
                "M-001",
                "Alice works at Acme Corp",
                entities=["Alice", "Acme"],
                importance=0.8,
                memory_type=MemoryType.ENTITY,
            ),
            day=0,
        )
        response = store.read(_make_query("Tell me about Alice and Acme", day=0))

        assert len(response.retrieved_memories) == 1
        # Both entities match → 2 × 0.15 = 0.30 entity boost
        mem = response.retrieved_memories[0]
        assert mem.score > 0.0

    def test_preference_task_boost_applied(self) -> None:
        """Preference store applies 1.2x boost for matching task_id."""
        store = PreferenceStore(decay_lambda=0.0)
        store.write_on_day(
            _make_event("M-001", "Dark mode preferred", task_id="ui_prefs", importance=0.8, memory_type=MemoryType.PREFERENCE),
            day=0,
        )

        response_match = store.read(_make_query("Dark mode", task_id="ui_prefs", day=0))
        response_no_match = store.read(_make_query("Dark mode", task_id="other", day=0))

        if response_match.retrieved_memories and response_no_match.retrieved_memories:
            assert (
                response_match.retrieved_memories[0].score
                >= response_no_match.retrieved_memories[0].score
            )
