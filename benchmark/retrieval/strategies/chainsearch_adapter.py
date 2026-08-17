"""Adapter for ChainSearch - advanced multi-chain retrieval ranking."""

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
from benchmark.retrieval.strategies.ann_adapter import ANNAdapter


class ChainSearchAdapter(RetrievalStrategy):
    """Benchmarks ChainSearch - advanced multi-chain ranking strategy.

    Combines three independent retrieval chains:
    1. BM25 (keyword matching)
    2. Dense vector (semantic similarity)
    3. ANN (fast approximate matching)

    Then fuses results using weighted scoring for optimal recall and precision.
    """

    name = "chainsearch"

    def __init__(self):
        self.bm25_adapter = BM25Adapter()
        self.dense_adapter = LearnedDenseAdapter()
        self.ann_adapter = ANNAdapter()
        self.documents: dict[str, str] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.build_time: float = 0.0

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize all three retrieval chains."""
        try:
            start = time.time()
            self.start_time = start

            # Store documents
            self.documents = {doc.get("id", ""): doc.get("content", "") for doc in documents}

            # Initialize all three chains
            self.bm25_adapter.initialize(documents)
            self.dense_adapter.initialize(documents)
            self.ann_adapter.initialize(documents)

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChainSearch: {e}")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search using ChainSearch multi-chain fusion."""
        try:
            start = time.time()

            # Get results from all three chains
            bm25_results = self.bm25_adapter.search(query, top_k=100)
            dense_results = self.dense_adapter.search(query, top_k=100)
            ann_results = self.ann_adapter.search(query, top_k=100)

            # Build weighted fusion scores
            chain_scores: dict[str, float] = {}

            # Chain 1: BM25 (weight=0.3)
            for rank, result in enumerate(bm25_results, 1):
                doc_id = result["doc_id"]
                score = 0.3 * (1.0 / (60 + rank))
                chain_scores[doc_id] = chain_scores.get(doc_id, 0.0) + score

            # Chain 2: Dense (weight=0.5, highest weight for semantic quality)
            for rank, result in enumerate(dense_results, 1):
                doc_id = result["doc_id"]
                score = 0.5 * (1.0 / (60 + rank))
                chain_scores[doc_id] = chain_scores.get(doc_id, 0.0) + score

            # Chain 3: ANN (weight=0.2, fast approximation)
            for rank, result in enumerate(ann_results, 1):
                doc_id = result["doc_id"]
                score = 0.2 * (1.0 / (60 + rank))
                chain_scores[doc_id] = chain_scores.get(doc_id, 0.0) + score

            # Rank by fused score
            ranked = sorted(chain_scores.items(), key=lambda x: x[1], reverse=True)

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
            raise RuntimeError(f"ChainSearch search failed: {e}")

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

            # Combined index size (all three chains)
            bm25_metrics = self.bm25_adapter.get_metrics()
            dense_metrics = self.dense_adapter.get_metrics()
            ann_metrics = self.ann_adapter.get_metrics()
            combined_index_size = (
                bm25_metrics.index_size_bytes +
                dense_metrics.index_size_bytes +
                ann_metrics.index_size_bytes
            )

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
                strategy_name="chainsearch",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute ChainSearch metrics: {e}")

    def teardown(self) -> None:
        """Clean up."""
        self.bm25_adapter.teardown()
        self.dense_adapter.teardown()
        self.ann_adapter.teardown()
        self.documents.clear()
        self.query_times.clear()
        self.search_results.clear()


RetrievalStrategyRegistry.register("chainsearch", ChainSearchAdapter)
