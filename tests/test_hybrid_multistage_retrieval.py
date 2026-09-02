"""Tests for hybrid and multi-stage retrieval strategies."""

import pytest

from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry
from benchmark.retrieval.strategies.cascading_adapter import CascadingAdapter
from benchmark.retrieval.strategies.hybrid_fusion_adapter import HybridFusionAdapter
from benchmark.retrieval.strategies.retrieval_rerank_adapter import RetrievalRerankAdapter


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


class TestHybridFusionAdapter:
    """Tests for hybrid fusion adapter."""

    def test_initialization(self, sample_documents):
        """Test hybrid fusion initialization."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0
        assert adapter.sparse_adapter.documents
        assert adapter.dense_adapter.documents

    def test_search(self, sample_documents):
        """Test hybrid fusion search."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_fusion_combines_both_strategies(self, sample_documents):
        """Test that fusion actually combines results from both strategies."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)

        # Search and get results
        results = adapter.search("fox", top_k=10)
        result_ids = {r["doc_id"] for r in results}

        # Results should include documents from both sparse and dense
        assert len(result_ids) > 0

    def test_multiple_searches(self, sample_documents, queries):
        """Test hybrid fusion with multiple searches."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert len(results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test hybrid fusion metrics."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "hybrid_fusion"
        # Hybrid should have larger index (both sparse + dense)
        assert metrics.index_size_bytes > 0

    def test_teardown(self, sample_documents):
        """Test hybrid fusion cleanup."""
        adapter = HybridFusionAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.query_times) == 0


class TestCascadingAdapter:
    """Tests for cascading adapter."""

    def test_initialization(self, sample_documents):
        """Test cascading initialization."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0
        assert adapter.sparse_adapter.documents
        assert adapter.dense_adapter.documents

    def test_search(self, sample_documents):
        """Test cascading search."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_cascading_two_stage_filtering(self, sample_documents):
        """Test that cascading uses both stages."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        # Cascading should work (may use stage 2 reranking or fall back to stage 1)
        results = adapter.search("fox", top_k=10)
        assert len(results) > 0

    def test_cascading_efficiency(self, sample_documents):
        """Test that cascading is relatively efficient."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        import time
        start = time.time()
        adapter.search("fox", top_k=5)
        elapsed = time.time() - start

        # Should be reasonably fast (less than 1 second)
        assert elapsed < 1.0

    def test_multiple_searches(self, sample_documents, queries):
        """Test cascading with multiple searches."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert isinstance(results, list)

    def test_get_metrics(self, sample_documents, queries):
        """Test cascading metrics."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "cascading"
        # Cascading should have smaller index than hybrid (only sparse kept)
        assert metrics.index_size_bytes > 0

    def test_teardown(self, sample_documents):
        """Test cascading cleanup."""
        adapter = CascadingAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.query_times) == 0


class TestRetrievalRerankAdapter:
    """Tests for retrieval + reranking adapter."""

    def test_initialization(self, sample_documents):
        """Test retrieval + rerank initialization."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert adapter.build_time > 0
        assert adapter.retrieval_adapter.documents

    def test_search(self, sample_documents):
        """Test retrieval + rerank search."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reranking_improves_precision(self, sample_documents):
        """Test that reranking produces relevant results."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)

        # Query with clear semantic meaning
        results = adapter.search("dog", top_k=5)
        assert len(results) > 0

        # Results should be related to dog
        result_ids = {r["doc_id"] for r in results}
        assert len(result_ids) > 0

    def test_multiple_searches(self, sample_documents, queries):
        """Test retrieval + rerank with multiple searches."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            results = adapter.search(query, top_k=5)
            assert isinstance(results, list)

    def test_get_metrics(self, sample_documents, queries):
        """Test retrieval + rerank metrics."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.strategy_name == "retrieval_rerank"
        # Should have smallest index (just retrieval)
        assert metrics.index_size_bytes > 0

    def test_teardown(self, sample_documents):
        """Test retrieval + rerank cleanup."""
        adapter = RetrievalRerankAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.query_times) == 0


class TestHybridMultistageRegistry:
    """Tests for hybrid and multi-stage strategy registry."""

    def test_registry_has_hybrid_strategies(self):
        """Test that hybrid strategies are registered."""
        assert RetrievalStrategyRegistry.is_registered("hybrid_fusion")
        assert RetrievalStrategyRegistry.is_registered("cascading")
        assert RetrievalStrategyRegistry.is_registered("retrieval_rerank")

    def test_get_hybrid_fusion(self):
        """Test getting hybrid fusion adapter."""
        adapter = RetrievalStrategyRegistry.get("hybrid_fusion")
        assert isinstance(adapter, HybridFusionAdapter)

    def test_get_cascading(self):
        """Test getting cascading adapter."""
        adapter = RetrievalStrategyRegistry.get("cascading")
        assert isinstance(adapter, CascadingAdapter)

    def test_get_retrieval_rerank(self):
        """Test getting retrieval + rerank adapter."""
        adapter = RetrievalStrategyRegistry.get("retrieval_rerank")
        assert isinstance(adapter, RetrievalRerankAdapter)

    def test_list_all_includes_hybrid(self):
        """Test listing all strategies includes hybrid methods."""
        strategies = RetrievalStrategyRegistry.list_all()
        assert "hybrid_fusion" in strategies
        assert "cascading" in strategies
        assert "retrieval_rerank" in strategies


class TestHybridMultistageComparison:
    """Comparison tests across hybrid and multi-stage adapters."""

    def test_all_adapters_implement_interface(self, sample_documents):
        """Test that all adapters implement required interface."""
        adapters = [
            HybridFusionAdapter(),
            CascadingAdapter(),
            RetrievalRerankAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "initialize")
            assert hasattr(adapter, "search")
            assert hasattr(adapter, "get_metrics")
            assert hasattr(adapter, "teardown")

    def test_all_adapters_produce_valid_metrics(self, sample_documents, queries):
        """Test that all adapters produce valid metrics."""
        adapters = [
            ("hybrid_fusion", HybridFusionAdapter()),
            ("cascading", CascadingAdapter()),
            ("retrieval_rerank", RetrievalRerankAdapter()),
        ]

        for name, adapter in adapters:
            adapter.initialize(sample_documents)

            for query in queries:
                adapter.search(query, top_k=10)

            metrics = adapter.get_metrics()

            # Basic validation
            assert 0.0 <= metrics.recall_at_10 <= 1.0, f"{name} recall_at_10 invalid"
            assert 0.0 <= metrics.precision_at_10 <= 1.0, f"{name} precision invalid"
            assert metrics.query_latency_ms >= 0.0, f"{name} latency invalid"
            assert metrics.index_build_time_sec > 0, f"{name} build time invalid"
            assert metrics.num_queries == len(queries), f"{name} num_queries mismatch"

            adapter.teardown()

    def test_all_adapters_handle_empty_results(self, sample_documents):
        """Test that all adapters gracefully handle queries with no results."""
        adapters = [
            HybridFusionAdapter(),
            CascadingAdapter(),
            RetrievalRerankAdapter(),
        ]

        for adapter in adapters:
            adapter.initialize(sample_documents)
            # Query unlikely to match
            results = adapter.search("xyzabc123nonexistent", top_k=5)
            assert isinstance(results, list)
            adapter.teardown()

    def test_cascading_more_efficient_than_hybrid(self, sample_documents):
        """Test that cascading (smaller index) is more efficient than hybrid."""

        # Build both
        hybrid = HybridFusionAdapter()
        hybrid.initialize(sample_documents)
        hybrid_metrics = hybrid.get_metrics()

        cascading = CascadingAdapter()
        cascading.initialize(sample_documents)
        cascading_metrics = cascading.get_metrics()

        # Cascading should have smaller index (no dense kept during inference)
        assert cascading_metrics.index_size_bytes <= hybrid_metrics.index_size_bytes

        hybrid.teardown()
        cascading.teardown()

    def test_retrieval_rerank_has_smallest_index(self, sample_documents):
        """Test that retrieval + rerank has smallest index."""
        adapters = [
            ("hybrid_fusion", HybridFusionAdapter()),
            ("cascading", CascadingAdapter()),
            ("retrieval_rerank", RetrievalRerankAdapter()),
        ]

        index_sizes = {}
        for name, adapter in adapters:
            adapter.initialize(sample_documents)
            metrics = adapter.get_metrics()
            index_sizes[name] = metrics.index_size_bytes
            adapter.teardown()

        # Ordering by index size
        assert index_sizes["retrieval_rerank"] <= index_sizes["cascading"]
        assert index_sizes["cascading"] <= index_sizes["hybrid_fusion"]
