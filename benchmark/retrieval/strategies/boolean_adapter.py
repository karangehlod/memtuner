"""Adapter for Boolean - exact keyword matching retrieval."""

import time
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)


class BooleanAdapter(RetrievalStrategy):
    """Benchmarks Boolean exact keyword matching retrieval.

    Boolean search matches documents containing exact keywords using AND/OR logic.
    Fastest retrieval but lower recall due to exact matching requirements.
    """

    name = "boolean"

    def __init__(self):
        self.documents: dict[str, str] = {}
        self.inverted_index: dict[str, set[str]] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.config: dict[str, Any] = {}
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize Boolean search with inverted index.

        Args:
            documents: List of dicts with 'id' and 'content' fields.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            start = time.time()
            self.start_time = start

            # Store documents and build inverted index
            self.documents = {}
            self.inverted_index = {}

            for doc in documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                self.documents[doc_id] = content

                # Tokenize and build inverted index
                tokens = content.lower().split()
                unique_tokens = set(tokens)

                for token in unique_tokens:
                    if token not in self.inverted_index:
                        self.inverted_index[token] = set()
                    self.inverted_index[token].add(doc_id)

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Boolean search: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using Boolean AND logic.

        Args:
            query: Query string with terms separated by spaces.
            top_k: Number of results to return.

        Returns:
            List of documents matching all query terms.

        Raises:
            RuntimeError: If search fails.
        """
        try:
            start = time.time()

            results = []

            # Parse query terms
            query_terms = query.lower().split()

            if not query_terms:
                elapsed = time.time() - start
                self.query_times.append(elapsed)
                self.num_queries += 1
                return []

            # Boolean AND - documents must contain ALL terms
            matching_docs = None

            for term in query_terms:
                term_docs = self.inverted_index.get(term, set())

                if matching_docs is None:
                    matching_docs = term_docs.copy()
                else:
                    matching_docs &= term_docs

            if matching_docs is None:
                matching_docs = set()

            # Score by number of matching terms
            scored_docs = []
            for doc_id in matching_docs:
                content = self.documents.get(doc_id, "")

                # Score: how many terms appear in document
                match_count = sum(
                    1 for term in query_terms
                    if term in content.lower()
                )
                score = match_count / max(len(query_terms), 1)

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
            self.search_results.extend(results)

            return results

        except Exception as e:
            self.errors += 1
            raise RuntimeError(f"Failed to search with Boolean: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Compute performance metrics for Boolean search.

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

            # Index size (inverted index)
            index_size = sum(
                len(term.encode()) * len(doc_ids)
                for term, doc_ids in self.inverted_index.items()
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
                strategy_name="boolean",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute Boolean metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.documents.clear()
        self.inverted_index.clear()
        self.query_times.clear()
        self.search_results.clear()


# Auto-register on import
RetrievalStrategyRegistry.register("boolean", BooleanAdapter)
