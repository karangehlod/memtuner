"""Adapter for ChainSearch - advanced multi-chain retrieval ranking."""

import time
from collections import defaultdict
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.ann_adapter import ANNAdapter
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)
from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter
from benchmark.retrieval.strategies.learned_dense_adapter import LearnedDenseAdapter


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
        # Per-query result lists — compute_metric_summary expects list[list[dict]],
        # not a flat list. The previous list.extend() was a correctness bug: it
        # passed all Q×top_k results as a single "query" to the aggregator.
        self._per_query_results: list[list[dict]] = []
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.build_time: float = 0.0

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize all three retrieval chains."""
        try:
            start = time.time()
            self.start_time = start

            self.documents = {doc.get("id", ""): doc.get("content", "") for doc in documents}

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

            bm25_results = self.bm25_adapter.search(query, top_k=100)
            dense_results = self.dense_adapter.search(query, top_k=100)
            ann_results = self.ann_adapter.search(query, top_k=100)

            # defaultdict(float) eliminates the double-lookup dict.get() + __setitem__
            # pattern — each update is a single hash lookup with in-place += .
            chain_scores: dict[str, float] = defaultdict(float)

            for rank, result in enumerate(bm25_results, 1):
                chain_scores[result["doc_id"]] += 0.3 * (1.0 / (60 + rank))

            for rank, result in enumerate(dense_results, 1):
                chain_scores[result["doc_id"]] += 0.5 * (1.0 / (60 + rank))

            for rank, result in enumerate(ann_results, 1):
                chain_scores[result["doc_id"]] += 0.2 * (1.0 / (60 + rank))

            import heapq as _hq
            results = [
                {"doc_id": doc_id, "score": float(score),
                 "content": self.documents.get(doc_id, "")}
                for doc_id, score in _hq.nlargest(top_k, chain_scores.items(), key=lambda x: x[1])
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)  # one list per query, not flat extend
            return results

        except Exception as e:
            self.errors += 1
            raise RuntimeError(f"ChainSearch search failed: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Get performance metrics."""
        try:
            metric_summary = compute_metric_summary(
                all_results=self._per_query_results if self._per_query_results else [],
                use_score_estimation=True,
            )

            avg_query_latency = (
                sum(self.query_times) / len(self.query_times)
                if self.query_times else 0.0
            )

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
        self._per_query_results.clear()


RetrievalStrategyRegistry.register("chainsearch", ChainSearchAdapter)
