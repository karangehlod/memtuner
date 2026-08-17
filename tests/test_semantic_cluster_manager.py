"""Comprehensive tests for SemanticClusterManager."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any

from benchmark.memory.strategies.semantic_cluster_manager import SemanticClusterManager


@dataclass
class MockMemory:
    """Mock Memory object for testing."""
    id: str
    content: str
    embedding: Optional[list[float]] = None
    score: float = 0.5


class TestSemanticClusterManagerInitialization:
    """Test cluster manager initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        manager = SemanticClusterManager()
        assert manager.num_clusters == 5
        assert manager.algorithm == 'kmeans'
        assert manager.random_state == 42
        assert manager.max_terms_per_cluster == 10

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        manager = SemanticClusterManager(
            num_clusters=8,
            algorithm='hierarchical',
            random_state=123,
            max_terms_per_cluster=5,
        )
        assert manager.num_clusters == 8
        assert manager.algorithm == 'hierarchical'
        assert manager.random_state == 123
        assert manager.max_terms_per_cluster == 5


class TestSemanticClusterManagerFit:
    """Test clustering fit operation."""

    def test_fit_with_valid_memories(self):
        """Test fitting with valid memories."""
        manager = SemanticClusterManager(num_clusters=3)
        memories = [
            MockMemory(f"m{i}", f"content for memory {i}", embedding=[float(i)] * 10)
            for i in range(10)
        ]

        manager.fit(memories)
        assert manager._fitted
        assert len(manager._clusters) == 3

    def test_fit_with_empty_memories_raises_error(self):
        """Test that fitting with empty list raises ValueError."""
        manager = SemanticClusterManager()
        with pytest.raises(ValueError, match="empty memories"):
            manager.fit([])

    def test_fit_organizes_memories_into_clusters(self):
        """Test that fit properly organizes memories."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [
            MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5)
            for i in range(6)
        ]

        manager.fit(memories)

        # All memories should be assigned to a cluster
        total_assigned = sum(len(cluster) for cluster in manager._clusters.values())
        assert total_assigned == len(memories)

    def test_fit_creates_centroids(self):
        """Test that fit creates centroids for each cluster."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [
            MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 3)
            for i in range(4)
        ]

        manager.fit(memories)

        assert len(manager._centroids) == 2
        for centroid in manager._centroids.values():
            assert isinstance(centroid, np.ndarray)


class TestSemanticClusterManagerPredict:
    """Test cluster prediction for new memories."""

    def test_predict_returns_valid_cluster_id(self):
        """Test that predict returns a valid cluster ID."""
        manager = SemanticClusterManager(num_clusters=3)
        memories = [
            MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5)
            for i in range(10)
        ]
        manager.fit(memories)

        new_memory = MockMemory("m_new", "new content", embedding=[3.5] * 5)
        cluster_id = manager.predict(new_memory)

        assert 0 <= cluster_id < 3

    def test_predict_without_fit_raises_error(self):
        """Test that predict without fit raises RuntimeError."""
        manager = SemanticClusterManager()
        memory = MockMemory("m0", "content", embedding=[0.5] * 5)

        with pytest.raises(RuntimeError, match="not fitted"):
            manager.predict(memory)

    def test_predict_handles_memory_without_embedding(self):
        """Test that predict handles memory without embedding gracefully."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [
            MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5)
            for i in range(4)
        ]
        manager.fit(memories)

        # Memory without embedding should get assigned to largest cluster
        new_memory = MockMemory("m_new", "new content", embedding=None)
        cluster_id = manager.predict(new_memory)

        assert 0 <= cluster_id < 2


class TestSemanticClusterManagerClusterOperations:
    """Test cluster query and management operations."""

    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [
            MockMemory(f"m{i}", f"important content here {i}", embedding=[float(i)] * 5)
            for i in range(4)
        ]
        manager.fit(memories)

        summary = manager.get_cluster_summary(0)
        assert "Cluster 0" in summary
        assert "size=" in summary

    def test_get_cluster_summary_invalid_cluster_raises_error(self):
        """Test that invalid cluster ID raises KeyError."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(2)]
        manager.fit(memories)

        with pytest.raises(KeyError):
            manager.get_cluster_summary(999)

    def test_get_cluster_members(self):
        """Test retrieving cluster members."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(4)]
        manager.fit(memories)

        members = manager.get_cluster_members(0)
        assert isinstance(members, list)
        assert len(members) > 0
        assert all(isinstance(m, MockMemory) for m in members)

    def test_get_cluster_members_invalid_cluster_raises_error(self):
        """Test that invalid cluster ID raises KeyError."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(2)]
        manager.fit(memories)

        with pytest.raises(KeyError):
            manager.get_cluster_members(999)


class TestSemanticClusterManagerReorganization:
    """Test cluster reorganization operations."""

    def test_reorganize_updates_clusters(self):
        """Test that reorganize recomputes clusters."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(4)]
        manager.fit(memories)

        original_assignment = dict(manager._memory_to_cluster)

        # Reorganize
        manager.reorganize()

        assert manager._fitted
        total_assigned = sum(len(cluster) for cluster in manager._clusters.values())
        assert total_assigned == len(memories)

    def test_reorganize_without_memories_raises_error(self):
        """Test that reorganize without prior fit raises RuntimeError."""
        manager = SemanticClusterManager()
        with pytest.raises(RuntimeError, match="No memories"):
            manager.reorganize()


class TestSemanticClusterManagerClusterInfo:
    """Test cluster information retrieval."""

    def test_get_cluster_info_returns_complete_info(self):
        """Test that get_cluster_info returns all fields."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5, score=0.7) for i in range(4)]
        manager.fit(memories)

        info = manager.get_cluster_info(0)
        assert info.cluster_id == 0
        assert info.size > 0
        assert isinstance(info.centroid, (list, type(None)))
        assert isinstance(info.top_terms, list)
        assert isinstance(info.summary, str)
        assert isinstance(info.average_score, float)

    def test_get_all_clusters(self):
        """Test retrieving all clusters info."""
        manager = SemanticClusterManager(num_clusters=3)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(6)]
        manager.fit(memories)

        all_clusters = manager.get_all_clusters()
        assert len(all_clusters) == 3
        assert all(hasattr(c, 'cluster_id') for c in all_clusters)


class TestSemanticClusterManagerAlgorithms:
    """Test different clustering algorithms."""

    def test_kmeans_algorithm(self):
        """Test K-means clustering."""
        manager = SemanticClusterManager(num_clusters=2, algorithm='kmeans')
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(6)]
        manager.fit(memories)

        assert manager._fitted
        assert len(manager._clusters) == 2

    def test_hierarchical_algorithm(self):
        """Test hierarchical clustering."""
        manager = SemanticClusterManager(num_clusters=2, algorithm='hierarchical')
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(6)]
        manager.fit(memories)

        assert manager._fitted
        assert len(manager._clusters) == 2

    def test_dbscan_algorithm(self):
        """Test DBSCAN clustering."""
        manager = SemanticClusterManager(algorithm='dbscan')
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i) + np.random.randn() * 0.1 for i in range(5)]) for i in range(10)]
        manager.fit(memories)

        assert manager._fitted
        # DBSCAN may create variable number of clusters
        assert len(manager._clusters) > 0

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises RuntimeError."""
        manager = SemanticClusterManager(algorithm='invalid')
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(4)]

        with pytest.raises(RuntimeError, match="Clustering fit failed"):
            manager.fit(memories)


class TestSemanticClusterManagerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_handles_single_cluster(self):
        """Test clustering with single cluster."""
        manager = SemanticClusterManager(num_clusters=1)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(3)]
        manager.fit(memories)

        assert len(manager._clusters) == 1
        assert len(manager.get_cluster_members(0)) == 3

    def test_handles_more_clusters_than_memories(self):
        """Test when num_clusters exceeds number of memories."""
        manager = SemanticClusterManager(num_clusters=10)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(3)]
        manager.fit(memories)

        # Should handle gracefully
        assert manager._fitted
        total_assigned = sum(len(cluster) for cluster in manager._clusters.values())
        assert total_assigned == 3

    def test_handles_memories_without_embeddings(self):
        """Test handling of memories without explicit embeddings."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [
            MockMemory(f"m{i}", f"content for memory {i}", embedding=None)
            for i in range(4)
        ]
        manager.fit(memories)

        # Should use fallback embedding from content hash
        assert manager._fitted
        assert len(manager._clusters) == 2


class TestSemanticClusterManagerConsistency:
    """Test consistency and reproducibility."""

    def test_same_random_state_produces_same_clusters(self):
        """Test that same random state produces same clustering."""
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(10)]

        manager1 = SemanticClusterManager(num_clusters=3, random_state=42)
        manager1.fit(memories)
        assignment1 = dict(manager1._memory_to_cluster)

        manager2 = SemanticClusterManager(num_clusters=3, random_state=42)
        manager2.fit(memories)
        assignment2 = dict(manager2._memory_to_cluster)

        assert assignment1 == assignment2

    def test_memory_assignment_tracked(self):
        """Test that memory-to-cluster assignments are tracked correctly."""
        manager = SemanticClusterManager(num_clusters=2)
        memories = [MockMemory(f"m{i}", f"content {i}", embedding=[float(i)] * 5) for i in range(4)]
        manager.fit(memories)

        # All memories should be in the mapping
        for memory in memories:
            assert memory.id in manager._memory_to_cluster
            cluster_id = manager._memory_to_cluster[memory.id]
            assert 0 <= cluster_id < 2
