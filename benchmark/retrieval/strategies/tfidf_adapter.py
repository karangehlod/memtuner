"""Adapter for TF-IDF - sparse vector retrieval."""

import time
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)


class TFIDFAdapter(RetrievalStrategy):
    """Benchmarks TF-IDF vector space retrieval.

    TF-IDF (Term Frequency-Inverse Document Frequency) represents documents
    as sparse vectors and ranks by cosine similarity.
    """

    name = "tfidf"

    def __init__(self):
        self.documents: dict[str, str] = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.config: dict[str, Any] = {}
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.doc_ids: list[str] = []

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize TF-IDF vectorizer with documents.

        Args:
            documents: List of dicts with 'id' and 'content' fields.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            start = time.time()
            self.start_time = start

            # Try to import sklearn
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self.use_sklearn = True
            except ImportError:
                self.use_sklearn = False

            # Store documents
            self.documents = {}
            self.doc_ids = []
            texts = []

            for doc in documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                self.documents[doc_id] = content
                self.doc_ids.append(doc_id)
                texts.append(content)

            # Initialize vectorizer if available
            if self.use_sklearn and texts:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    lowercase=True,
                    token_pattern=r"\b\w+\b",
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize TF-IDF: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using TF-IDF cosine similarity.

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

            if self.use_sklearn and self.vectorizer and self.tfidf_matrix is not None:
                # Transform query
                query_vector = self.vectorizer.transform([query])

                # Compute cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity

                similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

                # Rank by similarity
                ranked = sorted(
                    enumerate(similarities),
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
                # Fallback: simple term overlap
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
            self.search_results.extend(results)

            return results

        except Exception as e:
            self.errors += 1
            raise RuntimeError(f"Failed to search with TF-IDF: {e}")

    def get_metrics(self) -> RetrievalMetrics:
        """Compute performance metrics for TF-IDF.

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
                strategy_name="tfidf",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute TF-IDF metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.documents.clear()
        self.query_times.clear()
        self.search_results.clear()
        self.vectorizer = None
        self.tfidf_matrix = None


# Auto-register on import
RetrievalStrategyRegistry.register("tfidf", TFIDFAdapter)
