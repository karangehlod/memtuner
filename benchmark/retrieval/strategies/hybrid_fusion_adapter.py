"""Adapter for Hybrid Fusion - combining sparse and dense retrieval."""

import time
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)
from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter
from benchmark.retrieval.strategies.learned_dense_adapter import LearnedDenseAdapter


class HybridFusionAdapter(RetrievalStrategy):
    """Benchmarks hybrid fusion retrieval combining sparse and dense strategies.

    Uses Reciprocal Rank Fusion (RRF) to combine scores from BM25 and
    learned dense retrieval. Provides best-of-both-worlds accuracy.
    """

    name = "hybrid_fusion"

    def __init__(self):
        self.sparse_adapter = BM25Adapter()
        self.dense_adapter = LearnedDenseAdapter()
        self.documents: dict[str, str] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.build_time: float = 0.0

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize both sparse and dense adapters."""
        try:
            start = time.time()
            self.start_time = start

            # Store documents
            self.documents = {doc.get("id", ""): doc.get("content", "") for doc in documents}

            # Initialize both strategies
            self.sparse_adapter.initialize(documents)
            self.dense_adapter.initialize(documents)

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize hybrid fusion: {e}")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search using hybrid fusion with RRF."""
        try:
            start = time.time()

            # Get results from both strategies
            sparse_results = self.sparse_adapter.search(query, top_k=100)
            dense_results = self.dense_adapter.search(query, top_k=100)

            # Build RRF scores: 1/(k + rank)
            rrf_scores: dict[str, float] = {}

            # Add sparse contributions
            for rank, result in enumerate(sparse_results, 1):
                doc_id = result["doc_id"]
                sparse_rrf = 1.0 / (60 + rank)  # k=60 for sparse
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_rrf

            # Add dense contributions
            for rank, result in enumerate(dense_results, 1):
                doc_id = result["doc_id"]
                dense_rrf = 1.0 / (60 + rank)  # k=60 for dense
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_rrf

            # Rank by fused score
            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for doc_id, score in ranked[:top_k]:
                results.append({
                    "doc_id": doc_id,
                    "score": float(score),
                    "content": self.documents.get(doc_id, ""),
                })

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self.search_results.extend(results)
            return results

        except Exception as e:
            self.errors += 1
            raise RuntimeError(f"Hybrid fusion search failed: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Get performance metrics."""
        try:
            # Compute real metrics using score-based relevance estimation
            metric_summary = compute_metric_summary(
                all_results=[self.search_results] if self.search_results else [],
                use_score_estimation=True,
            )

            avg_query_latency = (
                sum(self.query_times) / len(self.query_times)
                if self.query_times else 0.0
            )

            # Combined index size (rough estimate)
            sparse_metrics = self.sparse_adapter.get_metrics()
            dense_metrics = self.dense_adapter.get_metrics()
            combined_index_size = sparse_metrics.index_size_bytes + dense_metrics.index_size_bytes

            success_rate = 1.0 - (self.errors / max(1, self.num_queries + self.errors))

            return RetrievalMetrics(
                recall_at_10=metric_summary["recall_at_10"],
                recall_at_100=metric_summary["recall_at_100"],
                mrr=metric_summary["mrr"],
                ndcg=metric_summary["ndcg"],
                precision_at_10=metric_summary["precision_at_10"],
                query_latency_ms=avg_query_latency * 1000,
                index_build_time_sec=self.build_time,
                index_size_bytes=combined_index_size,
                success_rate=success_rate,
                error_count=self.errors,
                strategy_name="hybrid_fusion",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute hybrid fusion metrics: {e}")

    def teardown(self) -> None:
        """Clean up."""
        self.sparse_adapter.teardown()
        self.dense_adapter.teardown()
        self.documents.clear()
        self.query_times.clear()
        self.search_results.clear()


RetrievalStrategyRegistry.register("hybrid_fusion", HybridFusionAdapter)
