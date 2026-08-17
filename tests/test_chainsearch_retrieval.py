"""Tests for ChainSearch retrieval strategy."""

import pytest
from benchmark.retrieval.strategies.chainsearch_adapter import ChainSearchAdapter
from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "doc_1", "content": "The quick brown fox jumps over the lazy dog"},
        {"id": "doc_2", "content": "A fast red fox runs through the forest"},
        {"id": "doc_3", "content": "The lazy dog sleeps under the tree"},
        {"id": "doc_4", "content": "Quick brown animals jump high"},
        {"id": "doc_5", "content": "Dogs and foxes are animals"},
    ]


@pytest.fixture
def queries():
    """Sample queries for testing."""
    return ["quick fox", "lazy dog", "brown fox jumps", "forest animals", "tree sleep"]


class TestChainSearchAdapter:
    """Tests for ChainSearch adapter."""

    def test_initialization(self, sample_documents):
        """Test ChainSearch initialization."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0
        assert adapter.bm25_adapter.documents
        assert adapter.dense_adapter.documents
        assert adapter.ann_adapter.documents

    def test_search(self, sample_documents):
        """Test ChainSearch search."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multi_chain_fusion(self, sample_documents):
        """Test that ChainSearch combines all three chains."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        # Search and get results
        results = adapter.search("fox", top_k=10)
        result_ids = {r["doc_id"] for r in results}

        # Should get documents (from fusion of all chains)
        assert len(result_ids) > 0

    def test_multiple_searches(self, sample_documents, queries):
        """Test ChainSearch with multiple searches."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert len(results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test ChainSearch metrics."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.precision_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "chainsearch"
        # ChainSearch should have largest index (all three chains)
        assert metrics.index_size_bytes > 0

    def test_teardown(self, sample_documents):
        """Test ChainSearch cleanup."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.query_times) == 0


class TestChainSearchRegistry:
    """Tests for ChainSearch registry."""

    def test_registry_has_chainsearch(self):
        """Test that ChainSearch is registered."""
        assert RetrievalStrategyRegistry.is_registered("chainsearch")

    def test_get_chainsearch(self):
        """Test getting ChainSearch adapter."""
        adapter = RetrievalStrategyRegistry.get("chainsearch")
        assert isinstance(adapter, ChainSearchAdapter)

    def test_list_all_includes_chainsearch(self):
        """Test listing all strategies includes ChainSearch."""
        strategies = RetrievalStrategyRegistry.list_all()
        assert "chainsearch" in strategies


class TestChainSearchComparison:
    """Comparison tests for ChainSearch."""

    def test_all_adapters_produce_valid_metrics(self, sample_documents, queries):
        """Test ChainSearch produces valid metrics."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        # Basic validation
        assert 0.0 <= metrics.recall_at_10 <= 1.0
        assert 0.0 <= metrics.precision_at_10 <= 1.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.index_build_time_sec > 0
        assert metrics.num_queries == len(queries)

        adapter.teardown()

    def test_chainsearch_handles_empty_results(self, sample_documents):
        """Test that ChainSearch gracefully handles queries with no results."""
        adapter = ChainSearchAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("xyzabc123nonexistent", top_k=5)
        assert isinstance(results, list)

        adapter.teardown()
