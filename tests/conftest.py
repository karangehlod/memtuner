"""Shared test fixtures and configuration.

Provides reusable fixtures across unit, contract, and integration tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.config.schema import (
    BenchmarkConfig,
    BenchmarkScopeConfig,
    MemoryConfig,
    MemorySelectionConfig,
)
from benchmark.cost.tracker import InMemoryCostTracker
from benchmark.factory.registry import MemoryModuleRegistry
from benchmark.gold.oracle import GoldOracle
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext
from benchmark.time.simulated_clock import SimulatedClock

FIXED_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture()
def fixed_timestamp() -> datetime:
    """Return a fixed timestamp for deterministic tests."""
    return FIXED_EPOCH


@pytest.fixture()
def simulated_clock() -> SimulatedClock:
    """Return a fresh simulated clock starting at fixed epoch."""
    return SimulatedClock(epoch=FIXED_EPOCH)


@pytest.fixture()
def sample_memory_event(fixed_timestamp: datetime) -> MemoryEvent:
    """Return a sample memory event for testing."""
    return MemoryEvent(
        id="M-001",
        type=MemoryType.EPISODIC,
        content="User prefers Postgres over Pinecone for vector storage",
        timestamp=fixed_timestamp,
        importance=0.85,
        entities=["user", "postgres", "pinecone"],
        task_id="db_selection",
    )


@pytest.fixture()
def sample_preference_event(fixed_timestamp: datetime) -> MemoryEvent:
    """Return a sample preference memory event."""
    return MemoryEvent(
        id="M-002",
        type=MemoryType.PREFERENCE,
        content="User prefers dark mode in IDE",
        timestamp=fixed_timestamp,
        importance=0.7,
        entities=["user", "ide"],
        task_id="ui_preferences",
    )


@pytest.fixture()
def sample_read_query() -> ReadQuery:
    """Return a sample read query."""
    return ReadQuery(
        query="Which database does the user prefer?",
        top_k=5,
        context=ReadQueryContext(simulated_day=3, task_id="db_selection"),
    )


@pytest.fixture()
def populated_registry() -> MemoryModuleRegistry:
    """Return a registry with all default implementations registered."""
    registry = MemoryModuleRegistry()
    registry.register("episodic_buffer", EpisodicBuffer)
    registry.register("context_buffer", ContextBuffer)
    registry.register("scratchpad", Scratchpad)
    registry.register("episodic_store", EpisodicStore)
    registry.register("preference_store", PreferenceStore)
    return registry


@pytest.fixture()
def default_benchmark_config() -> BenchmarkConfig:
    """Return a default benchmark config for testing."""
    return BenchmarkConfig(
        memory=MemoryConfig(
            enabled=MemorySelectionConfig(
                short_term=["episodic_buffer"],
                long_term=["episodic_store"],
            ),
        ),
        benchmark=BenchmarkScopeConfig(
            evaluation_horizon=14,
            seed=42,
            scenarios=["delayed_recall"],
        ),
    )


@pytest.fixture()
def cost_tracker() -> InMemoryCostTracker:
    """Return a fresh in-memory cost tracker."""
    return InMemoryCostTracker()


@pytest.fixture()
def gold_oracle() -> GoldOracle:
    """Return a fresh gold oracle."""
    return GoldOracle()
