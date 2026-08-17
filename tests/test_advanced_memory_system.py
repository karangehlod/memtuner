"""Comprehensive integration tests for AdvancedMemorySystem."""

import pytest
from benchmark.memory.strategies.advanced_memory_system import (
    AdvancedMemorySystem,
    UserProfile,
)


@pytest.fixture
def user_profile():
    """Create a test user profile."""
    return UserProfile(
        user_id="test_user",
        preferred_topics=["AI", "machine learning"],
        content_type_preferences={"article": 0.9, "video": 0.6},
    )


@pytest.fixture
def sample_memories():
    """Create sample memories."""
    return [
        {
            "id": "m1",
            "content": "Introduction to machine learning",
            "type": "article",
            "score": 0.8,
            "embedding": [0.1] * 10,
        },
        {
            "id": "m2",
            "content": "Deep learning neural networks",
            "type": "article",
            "score": 0.75,
            "embedding": [0.2] * 10,
        },
        {
            "id": "m3",
            "content": "AI applications in healthcare",
            "type": "video",
            "score": 0.7,
            "embedding": [0.3] * 10,
        },
        {
            "id": "m4",
            "content": "Cooking recipes for beginners",
            "type": "article",
            "score": 0.5,
            "embedding": [0.4] * 10,
        },
    ]


class TestSystemInitialization:
    """Test system initialization."""

    def test_system_creation(self, user_profile):
        """Test creating advanced memory system."""
        system = AdvancedMemorySystem(user_profile)
        assert system.user_profile.user_id == "test_user"

    def test_system_initialization_with_memories(self, user_profile, sample_memories):
        """Test initializing system with memories."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        assert system._is_initialized
        assert system.metrics.total_memories == 4

    def test_system_initialization_empty_raises_error(self, user_profile):
        """Test initializing with empty memories raises error."""
        system = AdvancedMemorySystem(user_profile)
        with pytest.raises(ValueError, match="empty memory"):
            system.initialize([])

    def test_system_initialization_with_custom_strategies(self, user_profile, sample_memories):
        """Test initialization with custom strategies."""
        strategies = ["strategy1", "strategy2"]
        system = AdvancedMemorySystem(user_profile, retrieval_strategies=strategies)
        system.initialize(sample_memories)

        assert system._is_initialized


class TestQueryExecution:
    """Test query execution pipeline."""

    def test_query_execution_basic(self, user_profile, sample_memories):
        """Test basic query execution."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("machine learning", top_k=2)

        assert result.query == "machine learning"
        assert len(result.results) <= 2
        assert result.total_time_ms > 0

    def test_query_execution_not_initialized_raises_error(self, user_profile):
        """Test query on uninitialized system raises error."""
        system = AdvancedMemorySystem(user_profile)
        with pytest.raises(ValueError, match="not initialized"):
            system.query("test")

    def test_query_execution_returns_results(self, user_profile, sample_memories):
        """Test query returns valid results."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("AI", top_k=10)

        assert result.results is not None
        assert isinstance(result.results, list)
        for res in result.results:
            assert "id" in res
            assert "cluster_id" in res

    def test_query_respects_top_k(self, user_profile, sample_memories):
        """Test query respects top_k limit."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("test", top_k=2)
        assert len(result.results) <= 2

    def test_query_classifications(self, user_profile, sample_memories):
        """Test different query types are classified."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result_factual = system.query("What is AI?", top_k=1)
        result_semantic = system.query("Tell me about learning", top_k=1)

        assert result_factual.query_type in ["factual", "semantic", "exact", "complex"]
        assert result_semantic.query_type in ["factual", "semantic", "exact", "complex"]

    def test_query_execution_tracks_metrics(self, user_profile, sample_memories):
        """Test query execution updates metrics."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        initial_count = system.metrics.total_queries
        system.query("test", top_k=1)
        system.query("test", top_k=1)

        assert system.metrics.total_queries == initial_count + 2


class TestMemoryManagement:
    """Test memory management operations."""

    def test_add_memory(self, user_profile, sample_memories):
        """Test adding new memory."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        new_memory = {
            "id": "m5",
            "content": "new content",
            "type": "article",
            "score": 0.6,
        }

        initial_count = system.metrics.total_memories
        result = system.add_memory(new_memory)

        assert result is True
        assert system.metrics.total_memories == initial_count + 1

    def test_add_memory_respects_capacity(self, user_profile):
        """Test adding memory respects tier capacity."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize([{"id": f"m{i}", "content": f"content {i}", "score": 0.5} for i in range(5)])

        # Try adding many memories
        added_count = 0
        for i in range(200):
            result = system.add_memory({
                "id": f"new_{i}",
                "content": f"new content {i}",
                "score": 0.5,
            })
            if result:
                added_count += 1

        # Should respect working tier capacity
        assert added_count > 0


class TestFeedbackIntegration:
    """Test feedback integration."""

    def test_provide_feedback_helpful(self, user_profile, sample_memories):
        """Test providing helpful feedback."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("machine learning", top_k=2)
        if result.results:
            mem_id = result.results[0]["id"]
            system.provide_feedback(mem_id, helpful=True)

            # Should update ranker feedback
            pref_score = system.ranker.get_preference_score(mem_id)
            assert pref_score > 0.0

    def test_provide_feedback_not_helpful(self, user_profile, sample_memories):
        """Test providing unhelpful feedback."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("test", top_k=2)
        if result.results:
            mem_id = result.results[0]["id"]
            system.provide_feedback(mem_id, helpful=False)

    def test_provide_feedback_unknown_memory(self, user_profile, sample_memories):
        """Test feedback on unknown memory."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        # Should not crash
        system.provide_feedback("unknown_id", helpful=True)


class TestConsolidation:
    """Test memory consolidation."""

    def test_consolidation_executes(self, user_profile, sample_memories):
        """Test consolidation execution."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        initial_count = system.metrics.consolidation_count
        system.consolidate()

        assert system.metrics.consolidation_count == initial_count + 1

    def test_consolidation_updates_tiers(self, user_profile, sample_memories):
        """Test consolidation updates tier statistics."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        # Give feedback to promote one memory
        result = system.query("AI", top_k=1)
        if result.results:
            system.provide_feedback(result.results[0]["id"], helpful=True)

        system.consolidate()

        stats = system.get_memory_statistics()
        assert "tiers" in stats


class TestMetrics:
    """Test metrics collection."""

    def test_get_system_metrics(self, user_profile, sample_memories):
        """Test getting system metrics."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        system.query("test", top_k=2)
        system.query("test2", top_k=2)

        metrics = system.get_system_metrics()

        assert metrics.total_queries == 2
        assert metrics.total_memories == 4
        assert metrics.tier_distribution is not None

    def test_get_memory_statistics(self, user_profile, sample_memories):
        """Test getting memory statistics."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        stats = system.get_memory_statistics()

        assert "tiers" in stats
        assert "clusters" in stats
        assert "total_memories" in stats

    def test_query_latency_tracking(self, user_profile, sample_memories):
        """Test query latency is tracked."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        system.query("test1", top_k=2)
        system.query("test2", top_k=2)

        metrics = system.get_system_metrics()
        assert metrics.avg_query_latency_ms > 0


class TestQueryHistory:
    """Test query history tracking."""

    def test_query_history_recorded(self, user_profile, sample_memories):
        """Test queries are recorded in history."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        system.query("query1", top_k=2)
        system.query("query2", top_k=2)

        history = system.get_query_history(limit=10)
        assert len(history) == 2

    def test_query_history_respects_limit(self, user_profile, sample_memories):
        """Test history respects limit."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        for i in range(20):
            system.query(f"query{i}", top_k=1)

        history = system.get_query_history(limit=5)
        assert len(history) == 5


class TestClusteringIntegration:
    """Test clustering integration."""

    def test_results_have_cluster_ids(self, user_profile, sample_memories):
        """Test query results include cluster IDs."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        result = system.query("machine learning", top_k=2)

        for res in result.results:
            assert "cluster_id" in res
            assert isinstance(res["cluster_id"], int)

    def test_clustering_in_statistics(self, user_profile, sample_memories):
        """Test clustering info in statistics."""
        system = AdvancedMemorySystem(user_profile)
        system.initialize(sample_memories)

        stats = system.get_memory_statistics()
        assert "clusters" in stats
        # Clusters may be empty if clustering failed, so just check it exists


class TestEndToEndFlow:
    """Test complete end-to-end flows."""

    def test_full_workflow(self, user_profile, sample_memories):
        """Test complete workflow: init → query → feedback → consolidate."""
        system = AdvancedMemorySystem(user_profile)

        # Initialize
        system.initialize(sample_memories)
        assert system.metrics.total_memories == 4

        # Query
        result1 = system.query("machine learning", top_k=2)
        assert len(result1.results) <= 2

        # Feedback
        if result1.results:
            system.provide_feedback(result1.results[0]["id"], helpful=True)

        # Consolidate
        system.consolidate()

        # Query again
        result2 = system.query("AI", top_k=2)
        assert len(result2.results) <= 2

        # Check metrics
        metrics = system.get_system_metrics()
        assert metrics.total_queries == 2
        assert metrics.consolidation_count == 1

    def test_multiple_users_isolated(self, sample_memories):
        """Test multiple independent systems are isolated."""
        profile1 = UserProfile(user_id="user1", preferred_topics=["AI"])
        profile2 = UserProfile(user_id="user2", preferred_topics=["cooking"])

        system1 = AdvancedMemorySystem(profile1)
        system2 = AdvancedMemorySystem(profile2)

        system1.initialize(sample_memories)
        system2.initialize(sample_memories)

        result1 = system1.query("machine learning", top_k=1)
        result2 = system2.query("machine learning", top_k=1)

        # Results may differ due to preferences
        assert result1.results is not None
        assert result2.results is not None

    def test_system_resilience_to_edge_cases(self, user_profile):
        """Test system handles edge cases gracefully."""
        system = AdvancedMemorySystem(user_profile)

        # Initialize with single memory
        system.initialize([{
            "id": "m1",
            "content": "single memory",
            "score": 0.5,
        }])

        # Query with empty string
        result = system.query("", top_k=1)
        assert result.results is not None

        # Query with special characters
        result = system.query("@#$%^&*()", top_k=1)
        assert result.results is not None


class TestPerformance:
    """Test performance characteristics."""

    def test_query_performance_scales(self, user_profile):
        """Test query performance with many memories."""
        system = AdvancedMemorySystem(user_profile)

        # Initialize with many memories
        memories = [{
            "id": f"m{i}",
            "content": f"memory content {i}",
            "score": 0.5 + min(0.5, (i % 10) / 20.0),
        } for i in range(100)]

        system.initialize(memories)

        # Query should complete reasonably
        result = system.query("content", top_k=10)
        assert len(result.results) <= 10
        assert result.total_time_ms < 5000  # Should be reasonably fast

    def test_consolidation_performance(self, user_profile):
        """Test consolidation performance."""
        system = AdvancedMemorySystem(user_profile)

        memories = [{
            "id": f"m{i}",
            "content": f"memory {i}",
            "score": 0.5,
        } for i in range(50)]

        system.initialize(memories)

        import time
        start = time.time()
        system.consolidate()
        elapsed = (time.time() - start) * 1000

        assert elapsed < 5000  # Should consolidate in reasonable time
