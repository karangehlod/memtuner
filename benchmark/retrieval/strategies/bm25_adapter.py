"""Adapter for BM25 - probabilistic sparse retrieval."""

import time
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)


class BM25Adapter(RetrievalStrategy):
    """Benchmarks BM25 probabilistic ranking algorithm.

    BM25 is a probabilistic sparse retrieval model that ranks documents
    based on term frequency and inverse document frequency.
    """

    name = "bm25"
    _MAX_STORED_RESULTS = 10000  # Prevent unbounded memory growth

    def __init__(self):
        self.documents: dict[str, str] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.config: dict[str, Any] = {}
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.cumulative_score = 0.0  # For online metric computation

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize BM25 index with documents.

        Args:
            documents: List of dicts with 'id' and 'content' fields.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            start = time.time()
            self.start_time = start

            # Try to import rank_bm25, fallback to simple implementation
            try:
                from rank_bm25 import BM25Okapi

                self.use_rank_bm25 = True
                self.bm25 = None
            except ImportError:
                self.use_rank_bm25 = False

            # Store documents
            self.documents = {}
            tokenized_corpus = []

            for doc in documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                self.documents[doc_id] = content

                # Simple tokenization
                tokens = content.lower().split()
                tokenized_corpus.append(tokens)

            # Initialize BM25 if available
            if self.use_rank_bm25 and tokenized_corpus:
                from rank_bm25 import BM25Okapi

                self.bm25 = BM25Okapi(tokenized_corpus)
                self.doc_ids = [d.get("id", "") for d in documents]

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize BM25: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using BM25.

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

            if self.use_rank_bm25 and self.bm25:
                # Use rank-bm25 library
                query_tokens = query.lower().split()
                scores = self.bm25.get_scores(query_tokens)

                # Rank by score
                ranked = sorted(
                    enumerate(scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Get top-k
                for idx, score in ranked[:top_k]:
                    if idx < len(self.doc_ids):
                        results.append({
                            "doc_id": self.doc_ids[idx],
                            "score": float(score),
                            "content": self.documents.get(self.doc_ids[idx], ""),
                        })
            else:
                # Fallback: simple term matching
                query_terms = set(query.lower().split())

                scored_docs = []
                for doc_id, content in self.documents.items():
                    doc_terms = set(content.lower().split())
                    overlap = len(query_terms & doc_terms)
                    score = overlap / max(len(query_terms), 1)
                    scored_docs.append((doc_id, score, content))

                # Sort by score
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Get top-k
                for doc_id, score, content in scored_docs[:top_k]:
                    results.append({
                        "doc_id": doc_id,
                        "score": score,
                        "content": content,
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
            raise RuntimeError(f"Failed to search with BM25: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Compute performance metrics for BM25.

        Returns:
            RetrievalMetrics with all computed metrics.
        """
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

            # Index size (exact bytes)
            index_size = sum(
                len(content.encode()) for content in self.documents.values()
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
                strategy_name="bm25",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute BM25 metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.documents.clear()
        self.query_times.clear()
        self.search_results.clear()
        if hasattr(self, "bm25"):
            self.bm25 = None


# Auto-register on import
RetrievalStrategyRegistry.register("bm25", BM25Adapter)
