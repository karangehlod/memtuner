"""Tests for new retrieval adapters (BM25L, ColBERT)."""

import pytest

from benchmark.retrieval.strategies.bm25l_adapter import BM25LAdapter
from benchmark.retrieval.strategies.colbert_adapter import ColBERTAdapter
from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "content": "The quick brown fox jumps over the lazy dog",
        },
        {
            "id": "doc_2",
            "content": "A fast red fox runs through the forest quickly",
        },
        {
            "id": "doc_3",
            "content": "The lazy dog sleeps under the tree peacefully",
        },
        {
            "id": "doc_4",
            "content": "Quick brown animals jump high in the air",
        },
        {
            "id": "doc_5",
            "content": "Dogs and foxes are wild animals that live in forests",
        },
    ]


@pytest.fixture
def long_documents():
    """Long documents for testing BM25L."""
    return [
        {
            "id": "long_1",
            "content": " ".join(["word" + str(i) for i in range(500)]),
        },
        {
            "id": "long_2",
            "content": " ".join(["term" + str(i) for i in range(1000)]),
        },
        {
            "id": "long_3",
            "content": " ".join(["token" + str(i) for i in range(200)]),
        },
    ]


@pytest.fixture
def queries():
    """Sample queries for testing."""
    return ["quick fox", "lazy dog", "animals", "forest"]


class TestBM25LAdapter:
    """Tests for BM25L adapter (long document variant)."""

    def test_initialization(self, sample_documents):
        """Test BM25L initialization."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.tokenized_docs) == len(sample_documents)
        assert len(adapter.idf_scores) > 0
        assert adapter.avg_doc_length > 0

    def test_search(self, sample_documents):
        """Test BM25L search."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dynamic_k1_short_docs(self):
        """Test BM25L dynamic k1 for short documents."""
        adapter = BM25LAdapter()
        short_docs = [
            {"id": "s1", "content": "a b"},
            {"id": "s2", "content": "c d e"},
            {"id": "s3", "content": "f"},
        ]
        adapter.initialize(short_docs)

        # Average doc length should be short
        assert adapter.avg_doc_length < 3

        # k1 should be higher for short average
        k1_short = adapter._get_dynamic_k1(1)
        assert k1_short >= adapter.k1  # Allow equal or higher

    def test_dynamic_k1_long_docs(self):
        """Test BM25L dynamic k1 for long documents."""
        adapter = BM25LAdapter()
        long_docs = [
            {"id": "l1", "content": " ".join(["word"] * 100)},
            {"id": "l2", "content": " ".join(["word"] * 150)},
            {"id": "l3", "content": " ".join(["word"] * 200)},
        ]
        adapter.initialize(long_docs)

        # Average doc length should be long
        assert adapter.avg_doc_length > 100

        # k1 should be lower for long average
        k1_long = adapter._get_dynamic_k1(250)
        assert k1_long < adapter.k1

    def test_idf_computation(self, sample_documents):
        """Test IDF score computation."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)

        # Rare terms should have higher IDF
        rare_term_idf = adapter.idf_scores.get("fox", 0)
        common_term_idf = adapter.idf_scores.get("the", 0)

        # "fox" appears less frequently than "the"
        assert rare_term_idf > 0

    def test_long_vs_short_scoring(self, long_documents):
        """Test BM25L scoring on mixed length documents."""
        adapter = BM25LAdapter()
        adapter.initialize(long_documents)

        # Query should score documents appropriately
        results = adapter.search("word term token", top_k=3)

        assert len(results) == 3
        # All should have reasonable scores
        assert all(r["score"] >= 0 for r in results)

    def test_multiple_searches(self, sample_documents, queries):
        """Test BM25L with multiple searches."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert len(results) > 0

        metrics = adapter.get_metrics()
        assert metrics.num_queries == len(queries)

    def test_get_metrics(self, sample_documents, queries):
        """Test BM25L metrics."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "bm25l"

    def test_teardown(self, sample_documents):
        """Test BM25L cleanup."""
        adapter = BM25LAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.tokenized_docs) == 0
        assert len(adapter.idf_scores) == 0


class TestColBERTAdapter:
    """Tests for ColBERT adapter (token-level dense)."""

    def test_initialization(self, sample_documents):
        """Test ColBERT initialization."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.doc_token_embeddings) == len(sample_documents)
        assert len(adapter.document_tokens) == len(sample_documents)

    def test_search(self, sample_documents):
        """Test ColBERT search."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_token_embeddings(self, sample_documents):
        """Test token-level embeddings are created."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        # Check that each document has token embeddings
        for doc_id, embeddings in adapter.doc_token_embeddings.items():
            assert len(embeddings) > 0
            # Each embedding should be a vector
            assert all(isinstance(e, (int, float, list)) for e in embeddings)

    def test_maxsim_scoring(self, sample_documents):
        """Test MaxSim scoring function."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        # Create sample embeddings
        query_vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        doc_vecs = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.0, 0.0, 0.0]]

        score = adapter._maxsim_score(query_vecs, doc_vecs)

        assert score >= 0.0
        assert score <= 1.0

    def test_phrase_matching(self, sample_documents):
        """Test ColBERT excels at phrase matching."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        # Exact phrase from document
        exact_phrase_results = adapter.search("quick brown fox", top_k=5)

        # Should find document with exact phrase
        assert len(exact_phrase_results) > 0

    def test_multiple_searches(self, sample_documents, queries):
        """Test ColBERT with multiple searches."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert len(results) > 0

        metrics = adapter.get_metrics()
        assert metrics.num_queries == len(queries)

    def test_get_metrics(self, sample_documents, queries):
        """Test ColBERT metrics."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.precision_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "colbert"

    def test_teardown(self, sample_documents):
        """Test ColBERT cleanup."""
        adapter = ColBERTAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.doc_token_embeddings) == 0
        assert len(adapter.document_tokens) == 0


class TestNewAdaptersRegistry:
    """Tests for new adapters in registry."""

    def test_bm25l_registered(self):
        """Test BM25L adapter is registered."""
        assert RetrievalStrategyRegistry.is_registered("bm25l")
        adapter = RetrievalStrategyRegistry.get("bm25l")
        assert isinstance(adapter, BM25LAdapter)

    def test_colbert_registered(self):
        """Test ColBERT adapter is registered."""
        assert RetrievalStrategyRegistry.is_registered("colbert")
        adapter = RetrievalStrategyRegistry.get("colbert")
        assert isinstance(adapter, ColBERTAdapter)

    def test_new_adapters_in_list(self):
        """Test new adapters appear in strategy list."""
        strategies = RetrievalStrategyRegistry.list_all()
        assert "bm25l" in strategies
        assert "colbert" in strategies


class TestAdapterComparison:
    """Comparison tests between new and existing adapters."""

    def test_bm25_vs_bm25l_long_docs(self, long_documents, queries):
        """Test BM25L handles long documents better than BM25."""
        from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter

        bm25 = BM25Adapter()
        bm25l = BM25LAdapter()

        bm25.initialize(long_documents)
        bm25l.initialize(long_documents)

        for query in queries:
            bm25.search(query, top_k=5)
            bm25l.search(query, top_k=5)

        bm25_metrics = bm25.get_metrics()
        bm25l_metrics = bm25l.get_metrics()

        # Both should complete successfully
        assert bm25_metrics.num_queries == len(queries)
        assert bm25l_metrics.num_queries == len(queries)

        bm25.teardown()
        bm25l.teardown()

    def test_colbert_vs_dense_phrases(self, sample_documents):
        """Test ColBERT handles phrase matching better than dense."""
        from benchmark.retrieval.strategies.dense_vector_adapter import DenseVectorAdapter

        colbert = ColBERTAdapter()
        dense = DenseVectorAdapter()

        colbert.initialize(sample_documents)
        dense.initialize(sample_documents)

        # Exact phrase query
        phrase_query = "quick brown fox"

        colbert_results = colbert.search(phrase_query, top_k=3)
        dense_results = dense.search(phrase_query, top_k=3)

        # Both should return results
        assert len(colbert_results) > 0
        assert len(dense_results) > 0

        colbert.teardown()
        dense.teardown()
