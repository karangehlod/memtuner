"""Tests for sparse retrieval strategies."""

import pytest
from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter
from benchmark.retrieval.strategies.tfidf_adapter import TFIDFAdapter
from benchmark.retrieval.strategies.boolean_adapter import BooleanAdapter
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
            "content": "A fast red fox runs through the forest",
        },
        {
            "id": "doc_3",
            "content": "The lazy dog sleeps under the tree",
        },
        {
            "id": "doc_4",
            "content": "Quick brown animals jump high",
        },
        {
            "id": "doc_5",
            "content": "Dogs and foxes are animals",
        },
    ]


@pytest.fixture
def queries():
    """Sample queries for testing."""
    return [
        "quick fox",
        "lazy dog",
        "brown fox jumps",
        "forest animals",
        "tree sleep",
    ]


class TestBM25Adapter:
    """Tests for BM25 adapter."""

    def test_initialization(self, sample_documents):
        """Test BM25 initialization."""
        adapter = BM25Adapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test BM25 search."""
        adapter = BM25Adapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, sample_documents):
        """Test BM25 with empty query."""
        adapter = BM25Adapter()
        adapter.initialize(sample_documents)

        results = adapter.search("", top_k=5)
        assert isinstance(results, list)

    def test_get_metrics(self, sample_documents, queries):
        """Test BM25 metrics."""
        adapter = BM25Adapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.num_documents == len(sample_documents)

    def test_teardown(self, sample_documents):
        """Test BM25 cleanup."""
        adapter = BM25Adapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.query_times) == 0


class TestTFIDFAdapter:
    """Tests for TF-IDF adapter."""

    def test_initialization(self, sample_documents):
        """Test TF-IDF initialization."""
        adapter = TFIDFAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test TF-IDF search."""
        adapter = TFIDFAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_searches(self, sample_documents, queries):
        """Test TF-IDF with multiple searches."""
        adapter = TFIDFAdapter()
        adapter.initialize(sample_documents)

        all_results = []
        for query in queries:
            results = adapter.search(query, top_k=5)
            all_results.extend(results)
            assert len(results) > 0

        assert len(all_results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test TF-IDF metrics."""
        adapter = TFIDFAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)

    def test_teardown(self, sample_documents):
        """Test TF-IDF cleanup."""
        adapter = TFIDFAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0


class TestBooleanAdapter:
    """Tests for Boolean adapter."""

    def test_initialization(self, sample_documents):
        """Test Boolean initialization."""
        adapter = BooleanAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.inverted_index) > 0

    def test_search(self, sample_documents):
        """Test Boolean search."""
        adapter = BooleanAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        # Boolean search with AND logic may return fewer results
        assert isinstance(results, list)
        if results:
            assert all("doc_id" in r and "score" in r for r in results)

    def test_search_no_matches(self, sample_documents):
        """Test Boolean search with no matches."""
        adapter = BooleanAdapter()
        adapter.initialize(sample_documents)

        # Query with terms that don't both appear
        results = adapter.search("xyzabc nonexistent", top_k=5)

        assert len(results) == 0

    def test_get_metrics(self, sample_documents, queries):
        """Test Boolean metrics."""
        adapter = BooleanAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "boolean"

    def test_teardown(self, sample_documents):
        """Test Boolean cleanup."""
        adapter = BooleanAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.inverted_index) == 0


class TestRetrievalRegistry:
    """Tests for retrieval strategy registry."""

    def test_registry_has_sparse_strategies(self):
        """Test that sparse strategies are registered."""
        assert RetrievalStrategyRegistry.is_registered("bm25")
        assert RetrievalStrategyRegistry.is_registered("tfidf")
        assert RetrievalStrategyRegistry.is_registered("boolean")

    def test_get_bm25(self):
        """Test getting BM25 adapter."""
        adapter = RetrievalStrategyRegistry.get("bm25")
        assert isinstance(adapter, BM25Adapter)

    def test_get_tfidf(self):
        """Test getting TF-IDF adapter."""
        adapter = RetrievalStrategyRegistry.get("tfidf")
        assert isinstance(adapter, TFIDFAdapter)

    def test_get_boolean(self):
        """Test getting Boolean adapter."""
        adapter = RetrievalStrategyRegistry.get("boolean")
        assert isinstance(adapter, BooleanAdapter)

    def test_list_all(self):
        """Test listing all strategies."""
        strategies = RetrievalStrategyRegistry.list_all()
        assert "bm25" in strategies
        assert "tfidf" in strategies
        assert "boolean" in strategies
