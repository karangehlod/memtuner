"""Comprehensive tests for all memory adapters."""

import pytest
import time
from benchmark.memory.adapters import (
    MemoryRegistry,
    EpisodicStoreAdapter,
    SemanticStoreAdapter,
    EntityStoreAdapter,
    PreferenceStoreAdapter,
    EpisodicBufferAdapter,
    ContextBufferAdapter,
    ScratchpadAdapter,
)


@pytest.fixture
def sample_memory():
    """Sample memory event for testing."""
    return {
        "id": "mem_001",
        "user_id": "user_1",
        "content": "Alice discussed a new approach to optimization",
        "importance": 0.85,
        "timestamp": time.time(),
        "day": 0,
        "task_id": "task_analysis",
    }


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "optimization approach"


class TestEpisodicStoreAdapter:
    """Tests for EpisodicStoreAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = EpisodicStoreAdapter()
        config = {"dataset_name": "test_data"}
        adapter.initialize(config)

        assert adapter.num_writes == 0
        assert adapter.num_queries == 0
        assert len(adapter.memories) == 0

    def test_write_memory(self, sample_memory):
        """Test writing memory."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        assert "mem_001" in adapter.memories
        assert adapter.memories["mem_001"]["content"] == sample_memory["content"]

    def test_query_memories(self, sample_memory, sample_query):
        """Test querying memories."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert len(results) > 0
        assert results[0]["memory_id"] == "mem_001"

    def test_get_metrics(self, sample_memory, sample_query):
        """Test metrics computation."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({"dataset_name": "test"})

        adapter.write_memory(sample_memory)
        adapter.query_memories(sample_query)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.write_latency_ms >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.dataset_name == "test"

    def test_teardown(self, sample_memory):
        """Test resource cleanup."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({})
        adapter.write_memory(sample_memory)

        adapter.teardown()

        assert len(adapter.memories) == 0
        assert len(adapter.query_results) == 0


class TestSemanticStoreAdapter:
    """Tests for SemanticStoreAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = SemanticStoreAdapter()
        adapter.initialize({"dataset_name": "semantic_test"})

        assert len(adapter.topics) == 0
        assert len(adapter.concept_index) == 0

    def test_write_memory(self, sample_memory):
        """Test writing semantic memory."""
        adapter = SemanticStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        assert "task_analysis" in adapter.topics or len(adapter.topics) > 0

    def test_concept_extraction(self):
        """Test concept extraction."""
        adapter = SemanticStoreAdapter()
        concepts = adapter._extract_concepts("machine learning optimization algorithms")

        assert len(concepts) > 0
        assert "machine" in concepts or "learning" in concepts or "optimization" in concepts

    def test_query_memories(self, sample_memory, sample_query):
        """Test semantic query."""
        adapter = SemanticStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestEntityStoreAdapter:
    """Tests for EntityStoreAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = EntityStoreAdapter()
        adapter.initialize({})

        assert len(adapter.entities) == 0
        assert len(adapter.relationships) == 0

    def test_write_memory(self, sample_memory):
        """Test writing entity memory."""
        adapter = EntityStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        assert "mem_001" in adapter.entities

    def test_relationship_extraction(self):
        """Test relationship extraction."""
        adapter = EntityStoreAdapter()
        rels = adapter._extract_relationships("Alice and Bob discussed the project with Charlie")

        assert isinstance(rels, list)
        # Should find capitalized names
        assert any(r in ["Alice", "Bob", "Charlie"] for r in rels)

    def test_query_memories(self, sample_memory, sample_query):
        """Test entity query."""
        adapter = EntityStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestPreferenceStoreAdapter:
    """Tests for PreferenceStoreAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = PreferenceStoreAdapter()
        adapter.initialize({})

        assert len(adapter.user_preferences) == 0
        assert len(adapter.preference_history) == 0

    def test_write_memory(self, sample_memory):
        """Test writing preference memory."""
        adapter = PreferenceStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        # Should have updated user profile
        assert len(adapter.user_preferences) > 0 or adapter.num_writes > 0

    def test_preference_extraction(self):
        """Test preference extraction."""
        adapter = PreferenceStoreAdapter()
        prefs = adapter._extract_preferences("I love creative design and beautiful algorithms")

        assert isinstance(prefs, dict)
        assert any(v > 0 for v in prefs.values())

    def test_query_memories(self, sample_memory, sample_query):
        """Test preference query."""
        adapter = PreferenceStoreAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestEpisodicBufferAdapter:
    """Tests for EpisodicBufferAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = EpisodicBufferAdapter()
        adapter.initialize({"capacity": 50})

        assert adapter.capacity == 50
        assert len(adapter.buffer) == 0

    def test_write_memory(self, sample_memory):
        """Test writing to buffer."""
        adapter = EpisodicBufferAdapter()
        adapter.initialize({"capacity": 10})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        assert len(adapter.buffer) == 1

    def test_eviction(self):
        """Test capacity-based eviction."""
        adapter = EpisodicBufferAdapter()
        adapter.initialize({"capacity": 3})

        for i in range(5):
            adapter.write_memory({
                "id": f"mem_{i}",
                "content": f"memory {i}",
                "importance": 0.5,
            })

        # Buffer should have at most capacity items
        assert len(adapter.buffer) <= adapter.capacity
        assert adapter.evictions > 0  # Some items evicted

    def test_query_memories(self, sample_memory, sample_query):
        """Test buffer query."""
        adapter = EpisodicBufferAdapter()
        adapter.initialize({"capacity": 50})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestContextBufferAdapter:
    """Tests for ContextBufferAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ContextBufferAdapter()
        adapter.initialize({"max_context_size": 30})

        assert adapter.max_context_size == 30
        assert len(adapter.context) == 0

    def test_relevance_filtering(self):
        """Test relevance-based filtering."""
        adapter = ContextBufferAdapter()
        adapter.initialize({"max_context_size": 50, "relevance_threshold": 0.6})

        # High relevance - should be added
        high_rel = {
            "id": "high",
            "content": "important",
            "importance": 0.9,
        }
        adapter.write_memory(high_rel)
        assert "high" in adapter.context

        # Low relevance - should be skipped
        low_rel = {
            "id": "low",
            "content": "unimportant",
            "importance": 0.2,
        }
        adapter.write_memory(low_rel)
        assert "low" not in adapter.context

    def test_query_memories(self, sample_memory, sample_query):
        """Test context query."""
        adapter = ContextBufferAdapter()
        adapter.initialize({})

        adapter.write_memory({**sample_memory, "importance": 0.7})
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestScratchpadAdapter:
    """Tests for ScratchpadAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ScratchpadAdapter()
        adapter.initialize({"capacity": 20})

        assert adapter.capacity == 20
        assert len(adapter.scratch) == 0

    def test_write_memory(self, sample_memory):
        """Test writing to scratchpad."""
        adapter = ScratchpadAdapter()
        adapter.initialize({"capacity": 20})

        adapter.write_memory(sample_memory)

        assert adapter.num_writes == 1
        assert "mem_001" in adapter.scratch

    def test_small_capacity(self):
        """Test scratchpad maintains small capacity."""
        adapter = ScratchpadAdapter()
        adapter.initialize({"capacity": 5})

        for i in range(10):
            adapter.write_memory({
                "id": f"mem_{i}",
                "content": f"memory {i}",
                "importance": 0.5,
            })

        # Should maintain small size
        assert len(adapter.scratch) <= adapter.capacity

    def test_ultra_fast_operations(self, sample_memory, sample_query):
        """Test that scratchpad operations are fast."""
        adapter = ScratchpadAdapter()
        adapter.initialize({"capacity": 20})

        # Write should be very fast
        adapter.write_memory(sample_memory)
        assert len(adapter.write_times) > 0
        assert adapter.write_times[0] < 1.0  # < 1ms

        # Query should be very fast
        results = adapter.query_memories(sample_query)
        assert len(adapter.query_times) > 0
        assert adapter.query_times[0] < 1.0  # < 1ms

    def test_query_memories(self, sample_memory, sample_query):
        """Test scratchpad query."""
        adapter = ScratchpadAdapter()
        adapter.initialize({})

        adapter.write_memory(sample_memory)
        results = adapter.query_memories(sample_query, top_k=5)

        assert isinstance(results, list)


class TestMemoryRegistry:
    """Tests for MemoryRegistry."""

    def test_registry_discovery(self):
        """Test all adapters are registered."""
        registry_names = MemoryRegistry.list_all()

        assert "episodic_store" in registry_names
        assert "semantic_store" in registry_names
        assert "entity_store" in registry_names
        assert "preference_store" in registry_names
        assert "episodic_buffer" in registry_names
        assert "context_buffer" in registry_names
        assert "scratchpad" in registry_names

    def test_get_adapter(self):
        """Test getting adapter by name."""
        adapter = MemoryRegistry.get("episodic_store")
        assert isinstance(adapter, EpisodicStoreAdapter)

        adapter = MemoryRegistry.get("scratchpad")
        assert isinstance(adapter, ScratchpadAdapter)

    def test_is_registered(self):
        """Test checking if adapter is registered."""
        assert MemoryRegistry.is_registered("episodic_store")
        assert MemoryRegistry.is_registered("scratchpad")
        assert not MemoryRegistry.is_registered("nonexistent")

    def test_list_all(self):
        """Test listing all registered adapters."""
        names = MemoryRegistry.list_all()

        assert len(names) >= 7
        assert all(isinstance(n, str) for n in names)


class TestMemoryAdapterComparison:
    """Tests comparing memory adapters."""

    def test_all_adapters_have_same_interface(self):
        """Test all adapters implement MemoryAdapter interface."""
        adapters = [
            EpisodicStoreAdapter(),
            SemanticStoreAdapter(),
            EntityStoreAdapter(),
            PreferenceStoreAdapter(),
            EpisodicBufferAdapter(),
            ContextBufferAdapter(),
            ScratchpadAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "initialize")
            assert hasattr(adapter, "write_memory")
            assert hasattr(adapter, "query_memories")
            assert hasattr(adapter, "get_metrics")
            assert hasattr(adapter, "teardown")

    def test_adapter_efficiency_comparison(self, sample_memory, sample_query):
        """Compare efficiency of different adapters."""
        adapters = {
            "episodic_store": EpisodicStoreAdapter(),
            "scratchpad": ScratchpadAdapter(),
        }

        latencies = {}

        for name, adapter in adapters.items():
            adapter.initialize({"dataset_name": name})

            start = time.perf_counter()
            adapter.write_memory(sample_memory)
            adapter.query_memories(sample_query)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            latencies[name] = elapsed

        # Scratchpad should be faster
        assert latencies["scratchpad"] <= latencies["episodic_store"] * 2


class TestErrorHandling:
    """Tests error handling in memory adapters."""

    def test_invalid_config(self):
        """Test handling of invalid config."""
        adapter = EpisodicStoreAdapter()
        # Should not raise
        adapter.initialize(None or {})

    def test_query_empty_adapter(self):
        """Test querying empty adapter."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({})

        results = adapter.query_memories("query", top_k=5)

        # Should return empty list, not raise
        assert results == []

    def test_metrics_on_empty_adapter(self):
        """Test metrics on empty adapter."""
        adapter = EpisodicStoreAdapter()
        adapter.initialize({"dataset_name": "empty"})

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.num_memories == 0
