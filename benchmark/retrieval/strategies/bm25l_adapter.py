"""Adapter for BM25L - variant optimized for longer documents."""

import time
from typing import Any

from benchmark.retrieval.metrics_utils import compute_metric_summary
from benchmark.retrieval.strategies.base import (
    RetrievalMetrics,
    RetrievalStrategy,
    RetrievalStrategyRegistry,
)


class BM25LAdapter(RetrievalStrategy):
    """Benchmarks BM25L retrieval - improved for longer documents.

    BM25L is a variant of BM25 designed to handle document length
    normalization better, especially for longer documents.
    Uses dynamic k1 parameter that adapts to document length.
    """

    name = "bm25l"

    def __init__(self):
        self.documents: dict[str, str] = {}
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.tokenized_docs = []
        self.idf_scores: dict[str, float] = {}
        self.query_times: list[float] = []
        self.search_results: list[tuple[str, float]] = []
        self.num_queries = 0
        self.errors = 0
        self.start_time: float = 0.0
        self.build_time: float = 0.0

        # BM25L parameters
        self.b = 0.75  # Length normalization parameter
        self.k1 = 1.2  # Term frequency saturation parameter (dynamic)

    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize BM25L index."""
        try:
            start = time.time()
            self.start_time = start

            self.documents = {}
            self.tokenized_docs = []
            self.doc_lengths = []
            doc_ids = []

            # Tokenize and store documents
            for doc in documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                self.documents[doc_id] = content
                doc_ids.append(doc_id)

                # Tokenize
                tokens = self._tokenize(content)
                self.tokenized_docs.append(tokens)
                self.doc_lengths.append(len(tokens))

            # Calculate average document length
            if self.doc_lengths:
                self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
            else:
                self.avg_doc_length = 0.0

            # Calculate IDF scores
            self._compute_idf_scores()

            self.build_time = time.time() - start

        except Exception as e:
            raise RuntimeError(f"Failed to initialize BM25L: {e}")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search using BM25L scoring."""
        try:
            start = time.time()

            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # Score documents
            scores = {}
            for doc_id, tokens in zip(self.documents.keys(), self.tokenized_docs):
                score = self._score_document(query_tokens, tokens)
                scores[doc_id] = score

            # Rank by score
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Build results
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
            raise RuntimeError(f"BM25L search failed: {e}")

    def _score_document(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """Score document using BM25L formula."""
        score = 0.0
        doc_length = len(doc_tokens)

        # Dynamic k1 based on document length
        k1_dynamic = self._get_dynamic_k1(doc_length)

        for token in query_tokens:
            if token not in self.idf_scores:
                continue

            # Term frequency in document
            tf = doc_tokens.count(token)

            # BM25L length normalization (improved for longer docs)
            # BM25L = BM25 + (1 - b) term at denominator
            idf = self.idf_scores[token]

            # Calculate normalized TF
            norm_factor = 1.0 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1.0))
            norm_tf = (k1_dynamic + 1.0) * tf / (k1_dynamic * norm_factor + tf)

            score += idf * norm_tf

        return score

    def _get_dynamic_k1(self, doc_length: int) -> float:
        """Get dynamic k1 based on document length."""
        # For longer documents, use lower k1 to reduce impact of term frequency
        if doc_length > self.avg_doc_length * 1.5:
            return max(0.5, self.k1 - 0.3)  # Lower k1 for long docs
        elif doc_length < self.avg_doc_length * 0.5:
            return min(2.0, self.k1 + 0.3)  # Higher k1 for short docs
        else:
            return self.k1

    def _compute_idf_scores(self) -> None:
        """Compute IDF scores for all terms."""
        self.idf_scores = {}
        num_docs = len(self.tokenized_docs)

        # Collect all unique terms
        all_terms = set()
        for tokens in self.tokenized_docs:
            all_terms.update(tokens)

        # Calculate IDF for each term
        for term in all_terms:
            # Count documents containing term
            doc_freq = sum(1 for tokens in self.tokenized_docs if term in tokens)

            # IDF = log((N - n + 0.5) / (n + 0.5))
            idf = max(0.0, (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            self.idf_scores[term] = idf

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        return text.lower().split()

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

            index_size = sum(len(t) * 8 for t in self.tokenized_docs)  # Rough estimate

            success_rate = 1.0 - (self.errors / max(1, self.num_queries + self.errors))

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
                strategy_name="bm25l",
                num_queries=self.num_queries,
                num_documents=len(self.documents),
                elapsed_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute BM25L metrics: {e}")

    def teardown(self) -> None:
        """Clean up."""
        self.documents.clear()
        self.tokenized_docs.clear()
        self.doc_lengths.clear()
        self.idf_scores.clear()
        self.query_times.clear()
        self.search_results.clear()


RetrievalStrategyRegistry.register("bm25l", BM25LAdapter)
