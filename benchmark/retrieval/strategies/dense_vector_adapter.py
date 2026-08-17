"""Adapter for Dense Vector - semantic retrieval with pre-trained embeddings."""

import hashlib
import math
import time
from typing import Any

import numpy as np

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)

# Try to import sentence-transformers at module level
try:
    from sentence_transformers import SentenceTransformer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import os as _os
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
    if _CUDA_AVAILABLE:
        _cpu_cap = max(1, (_os.cpu_count() or 4) // 2)
        try:
            _torch.set_num_threads(_cpu_cap)
        except RuntimeError:
            pass
except ImportError:
    _CUDA_AVAILABLE = False

_EMBEDDING_DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"


class DenseVectorAdapter(RetrievalStrategy):
    """Benchmarks dense vector retrieval using pre-trained embeddings.

    Uses semantic embeddings (sentence-transformers) for dense retrieval.
    Ranks documents by cosine similarity in embedding space.
    """

    name = "dense_vector"
    _MAX_STORED_RESULTS = 10000

    def __init__(self):
        self.documents: dict[str, str] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.config: dict[str, Any] = {}
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.model = None
        self.use_transformers = _TRANSFORMERS_AVAILABLE

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize dense embeddings for documents.

        Args:
            documents: List of dicts with 'id' and 'content' fields.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            start = time.time()
            self.start_time = start

            # Try to load model if transformers available
            if _TRANSFORMERS_AVAILABLE:
                try:
                    self.model = SentenceTransformer("all-MiniLM-L6-v2", device=_EMBEDDING_DEVICE)
                    self.use_transformers = True
                except Exception:
                    # Network error or model not available, use fallback
                    self.use_transformers = False
            else:
                self.use_transformers = False

            # Store documents
            self.documents = {}
            self.embeddings = {}
            doc_ids = []
            texts = []

            for doc in documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                self.documents[doc_id] = content
                doc_ids.append(doc_id)
                texts.append(content)

            # Generate embeddings
            if self.use_transformers and self.model:
                embeddings_array = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=512 if _CUDA_AVAILABLE else 128,
                )

                for doc_id, embedding in zip(doc_ids, embeddings_array):
                    self.embeddings[doc_id] = embedding.tolist()
            else:
                # Fallback: simple hash-based "embeddings"
                for doc_id, text in zip(doc_ids, texts):
                    # Create a simple vector from text
                    self.embeddings[doc_id] = self._simple_embedding(text)

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize dense vectors: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using dense vectors.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of ranked documents with scores.

        Raises:
            RuntimeError: If search fails.
        """
        try:
            start = time.time()

            results = []

            if self.use_transformers and self.model:
                # Encode query (module-level import, not repeated per query)
                query_embedding = self.model.encode([query], show_progress_bar=False)[0]

                # Compute cosine similarity
                scores = {}
                for doc_id, doc_embedding in self.embeddings.items():
                    doc_vec = np.array(doc_embedding)
                    query_vec = np.array(query_embedding)

                    # Cosine similarity
                    similarity = np.dot(query_vec, doc_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
                    )
                    scores[doc_id] = float(similarity)
            else:
                # Fallback: simple embedding similarity
                query_emb = self._simple_embedding(query)
                scores = {}

                for doc_id, doc_emb in self.embeddings.items():
                    similarity = self._cosine_similarity(query_emb, doc_emb)
                    scores[doc_id] = similarity

            # Rank by score
            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Get top-k
            for doc_id, score in ranked[:top_k]:
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "content": self.documents.get(doc_id, ""),
                })

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1

            # Bounded memory accumulation (prevent unbounded growth)
            self.search_results.extend(results)
            if len(self.search_results) > self._MAX_STORED_RESULTS:
                self.search_results = self.search_results[-self._MAX_STORED_RESULTS:]

            return results

        except Exception as e:
            self.errors += 1
            raise RuntimeError(f"Failed to search with dense vectors: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Compute performance metrics for dense vector retrieval."""
        try:
            # Compute real metrics using score-based relevance estimation
            metric_summary = compute_metric_summary(
                all_results=[self.search_results] if self.search_results else [],
                use_score_estimation=True,
            )

            # Efficiency metrics
            avg_query_latency = (
                sum(self.query_times) / len(self.query_times)
                if self.query_times
                else 0.0
            )

            # Index size (embeddings)
            index_size = sum(
                len(str(emb).encode())
                for emb in self.embeddings.values()
            )

            # Success rate
            success_rate = 1.0 - (
                self.errors / max(1, self.num_queries + self.errors)
            )

            return RetrievalMetrics(
                recall_at_10=metric_summary["recall_at_10"],
                recall_at_100=metric_summary["recall_at_100"],
                mrr=metric_summary["mrr"],
                ndcg=metric_summary["ndcg"],
                precision_at_10=metric_summary["precision_at_10"],
                query_latency_ms=avg_query_latency * 1000,
                index_build_time_sec=self.build_time,
                index_size_bytes=float(index_size),
                success_rate=success_rate,
                error_count=self.errors,
                strategy_name="dense_vector",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute dense vector metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.documents.clear()
        self.embeddings.clear()
        self.query_times.clear()
        self.search_results.clear()
        self.model = None

    @staticmethod
    def _simple_embedding(text: str) -> list[float]:
        """Create simple embedding from text for fallback."""
        # Hash-based simple embedding (module-level import)
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        # Convert hex to floats
        return [float(int(text_hash[i:i+2], 16)) / 256.0 for i in range(0, 32, 2)]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# Auto-register on import
RetrievalStrategyRegistry.register("dense_vector", DenseVectorAdapter)
