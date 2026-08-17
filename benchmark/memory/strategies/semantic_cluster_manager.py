"""Semantic clustering manager for organizing memories by theme."""

import numpy as np
from typing import Any, Optional, Protocol
from dataclasses import dataclass, field

from benchmark.models.memory_event import MemoryEvent


@dataclass
class ClusterInfo:
    """Information about a memory cluster."""
    cluster_id: int
    size: int
    centroid: Optional[list[float]]
    top_terms: list[str]
    summary: str
    average_score: float


class SemanticClusterManager:
    """Manages semantic clustering of memories for theme-based organization.

    Uses K-means clustering to group memories by semantic similarity,
    automatically summarizes cluster themes, and provides utilities for
    cluster-based operations.

    Supported algorithms:
    - K-means: Fast, scalable clustering
    - Hierarchical: Preserves cluster hierarchy
    - DBSCAN: Density-based, no preset cluster count
    """

    def __init__(
        self,
        num_clusters: int = 5,
        algorithm: str = 'kmeans',
        random_state: int = 42,
        max_terms_per_cluster: int = 10,
    ):
        """Initialize the clustering manager.

        Args:
            num_clusters: Number of clusters to create (for kmeans/hierarchical)
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan')
            random_state: Random seed for reproducibility
            max_terms_per_cluster: Max terms to extract per cluster summary
        """
        self.num_clusters = num_clusters
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_terms_per_cluster = max_terms_per_cluster

        # Cluster state
        self._clusters: dict[int, list[dict[str, Any]]] = {}
        self._centroids: dict[int, np.ndarray] = {}
        self._fitted = False
        self._memory_to_cluster: dict[str, int] = {}
        self._cluster_scores: dict[int, list[float]] = {}

    def fit(self, memories: list[dict[str, Any]]) -> None:
        """Fit clustering model on memories.

        Args:
            memories: List of Memory objects to cluster

        Raises:
            ValueError: If memories list is empty
            RuntimeError: If clustering fails
        """
        if not memories:
            raise ValueError("Cannot fit with empty memories list")

        try:
            # Extract embeddings
            embeddings = self._extract_embeddings(memories)

            if embeddings.shape[0] == 0:
                raise RuntimeError("No valid embeddings extracted from memories")

            # Run clustering algorithm
            if self.algorithm == 'kmeans':
                labels = self._fit_kmeans(embeddings)
            elif self.algorithm == 'hierarchical':
                labels = self._fit_hierarchical(embeddings)
            elif self.algorithm == 'dbscan':
                labels = self._fit_dbscan(embeddings)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Organize memories by cluster
            self._organize_clusters(memories, labels, embeddings)
            self._fitted = True

        except Exception as e:
            raise RuntimeError(f"Clustering fit failed: {e}")

    def predict(self, memory: dict[str, Any]) -> int:
        """Predict cluster for a new memory.

        Args:
            memory: Memory to classify

        Returns:
            Cluster ID (0 to num_clusters-1)

        Raises:
            RuntimeError: If model not fitted or prediction fails
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        try:
            embedding = self._extract_single_embedding(memory)

            if embedding is None:
                # Fallback: assign to largest cluster
                return max(self._clusters.keys(),
                          key=lambda k: len(self._clusters[k]))

            # Find nearest centroid
            min_distance = float('inf')
            best_cluster = 0

            for cluster_id, centroid in self._centroids.items():
                # Handle dimension mismatch by using largest cluster as fallback
                try:
                    if embedding.shape != centroid.shape:
                        continue
                    distance = np.linalg.norm(embedding - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
                except (ValueError, TypeError):
                    continue

            return best_cluster

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def get_cluster_summary(self, cluster_id: int) -> str:
        """Get human-readable summary of cluster theme.

        Args:
            cluster_id: ID of cluster to summarize

        Returns:
            String summary of cluster theme

        Raises:
            KeyError: If cluster_id doesn't exist
        """
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster {cluster_id} not found")

        cluster_memories = self._clusters[cluster_id]

        if not cluster_memories:
            return f"Empty cluster {cluster_id}"

        # Extract top terms from memory content
        terms = self._extract_top_terms(cluster_memories)

        if not terms:
            return f"Cluster {cluster_id} ({len(cluster_memories)} items)"

        # Build summary
        term_str = ", ".join(terms[:self.max_terms_per_cluster])
        avg_score = np.mean(self._cluster_scores.get(cluster_id, [0.0]))

        return f"Cluster {cluster_id}: {term_str} (size={len(cluster_memories)}, score={avg_score:.2f})"

    def get_cluster_members(self, cluster_id: int) -> list[dict[str, Any]]:
        """Get all memories in a cluster.

        Args:
            cluster_id: ID of cluster

        Returns:
            List of Memory objects in cluster

        Raises:
            KeyError: If cluster_id doesn't exist
        """
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster {cluster_id} not found")

        return list(self._clusters[cluster_id])

    def reorganize(self) -> None:
        """Re-run clustering and update all clusters.

        Useful after many new memories added or if clustering parameters changed.

        Raises:
            RuntimeError: If no memories have been clustered yet
        """
        if not self._memory_to_cluster:
            raise RuntimeError("No memories to reorganize")

        # Collect all memories
        all_memories = []
        for cluster_mems in self._clusters.values():
            all_memories.extend(cluster_mems)

        # Clear state and refit
        self._clusters.clear()
        self._centroids.clear()
        self._memory_to_cluster.clear()
        self._cluster_scores.clear()
        self._fitted = False

        # Refit
        self.fit(all_memories)

    def get_cluster_info(self, cluster_id: int) -> ClusterInfo:
        """Get detailed information about a cluster.

        Args:
            cluster_id: ID of cluster

        Returns:
            ClusterInfo dataclass with full details

        Raises:
            KeyError: If cluster_id doesn't exist
        """
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster {cluster_id} not found")

        cluster_mems = self._clusters[cluster_id]
        top_terms = self._extract_top_terms(cluster_mems)
        centroid = self._centroids[cluster_id]
        avg_score = np.mean(self._cluster_scores.get(cluster_id, [0.0]))

        return ClusterInfo(
            cluster_id=cluster_id,
            size=len(cluster_mems),
            centroid=centroid.tolist() if centroid is not None else None,
            top_terms=top_terms,
            summary=self.get_cluster_summary(cluster_id),
            average_score=avg_score,
        )

    def get_all_clusters(self) -> list[ClusterInfo]:
        """Get information about all clusters.

        Returns:
            List of ClusterInfo objects for all clusters
        """
        return [self.get_cluster_info(cid) for cid in sorted(self._clusters.keys())]

    # Private helper methods

    def _extract_embeddings(self, memories: list[dict[str, Any]]) -> np.ndarray:
        """Extract embeddings from memories."""
        embeddings = []

        for memory in memories:
            emb = self._extract_single_embedding(memory)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return np.array([])

        return np.array(embeddings)

    def _extract_single_embedding(self, memory: dict[str, Any]) -> Optional[np.ndarray]:
        """Extract embedding from single memory."""
        if hasattr(memory, 'embedding') and memory.embedding is not None:
            if isinstance(memory.embedding, (list, tuple)):
                return np.array(memory.embedding, dtype=np.float32)
            elif isinstance(memory.embedding, np.ndarray):
                return memory.embedding.astype(np.float32)

        # Fallback: create simple embedding from content hash
        if hasattr(memory, 'content') and memory.content:
            import hashlib
            content_hash = hashlib.md5(str(memory.content).encode()).digest()
            return np.array([float(b) / 256.0 for b in content_hash[:16]], dtype=np.float32)

        return None

    def _organize_clusters(
        self,
        memories: list[dict[str, Any]],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        """Organize memories into clusters based on labels."""
        # Initialize cluster structures
        for cluster_id in range(self.num_clusters):
            self._clusters[cluster_id] = []
            self._cluster_scores[cluster_id] = []

        # Assign memories to clusters
        for memory, label, embedding in zip(memories, labels, embeddings):
            cluster_id = int(label)
            self._clusters[cluster_id].append(memory)
            self._memory_to_cluster[memory.id] = cluster_id

            # Store memory score if available
            if hasattr(memory, 'score'):
                self._cluster_scores[cluster_id].append(float(memory.score))
            else:
                self._cluster_scores[cluster_id].append(0.5)

        # Compute centroids
        for cluster_id in range(self.num_clusters):
            cluster_embeddings = embeddings[labels == cluster_id]
            if len(cluster_embeddings) > 0:
                self._centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
            else:
                self._centroids[cluster_id] = np.zeros_like(embeddings[0])

    def _fit_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit K-means clustering."""
        from sklearn.cluster import KMeans

        n_clusters = min(self.num_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        return labels

    def _fit_hierarchical(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit hierarchical clustering."""
        from scipy.cluster.hierarchy import linkage, fcluster

        linkage_matrix = linkage(embeddings, method='ward')
        n_clusters = min(self.num_clusters, len(embeddings))
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        return labels

    def _fit_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        # Estimate eps from distances
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(scaled)
        distances = neighbors.kneighbors(scaled)[0]
        eps = np.mean(distances) + np.std(distances)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(scaled)

        # Remap labels: DBSCAN uses -1 for noise, map to valid clusters
        unique_labels = set(labels)
        if -1 in unique_labels:
            # Assign noise points to nearest cluster
            noise_mask = labels == -1
            if np.any(noise_mask):
                cluster_centers = {
                    label: np.mean(scaled[labels == label], axis=0)
                    for label in unique_labels if label != -1
                }

                for i in np.where(noise_mask)[0]:
                    nearest_cluster = min(
                        cluster_centers.keys(),
                        key=lambda c: np.linalg.norm(scaled[i] - cluster_centers[c])
                    )
                    labels[i] = nearest_cluster

        return labels

    def _extract_top_terms(self, memories: list[dict[str, Any]], top_n: int = 10) -> list[str]:
        """Extract top terms from memory content."""
        from collections import Counter

        terms = []

        for memory in memories:
            if hasattr(memory, 'content') and memory.content:
                # Simple word tokenization
                words = str(memory.content).lower().split()
                # Filter short words
                words = [w for w in words if len(w) > 3]
                terms.extend(words[:5])  # Take first 5 words per memory

        if not terms:
            return []

        # Get most common terms
        counter = Counter(terms)
        top_terms = [term for term, _ in counter.most_common(top_n)]

        return top_terms
