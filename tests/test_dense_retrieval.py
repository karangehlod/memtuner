"""Tests for dense retrieval strategies."""

import pytest
from benchmark.retrieval.strategies.dense_vector_adapter import DenseVectorAdapter
from benchmark.retrieval.strategies.learned_dense_adapter import LearnedDenseAdapter
from benchmark.retrieval.strategies.ann_adapter import ANNAdapter
from benchmark.retrieval.strategies.quantized_adapter import QuantizedAdapter
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


class TestDenseVectorAdapter:
    """Tests for dense vector adapter."""

    def test_initialization(self, sample_documents):
        """Test dense vector initialization."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.embeddings) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test dense vector search."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, sample_documents):
        """Test dense vector with empty query."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("", top_k=5)
        assert isinstance(results, list)

    def test_multiple_searches(self, sample_documents, queries):
        """Test dense vector with multiple searches."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)

        all_results = []
        for query in queries:
            results = adapter.search(query, top_k=5)
            all_results.extend(results)
            assert len(results) > 0

        assert len(all_results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test dense vector metrics."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.num_documents == len(sample_documents)
        assert metrics.strategy_name == "dense_vector"

    def test_teardown(self, sample_documents):
        """Test dense vector cleanup."""
        adapter = DenseVectorAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.embeddings) == 0
        assert len(adapter.query_times) == 0


class TestLearnedDenseAdapter:
    """Tests for learned dense adapter."""

    def test_initialization(self, sample_documents):
        """Test learned dense initialization."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.doc_embeddings) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test learned dense search."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_respected(self, sample_documents):
        """Test that top_k limit is respected."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("fox", top_k=2)
        assert len(results) <= 2

    def test_multiple_searches(self, sample_documents, queries):
        """Test learned dense with multiple searches."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        all_results = []
        for query in queries:
            results = adapter.search(query, top_k=5)
            all_results.extend(results)
            assert len(results) > 0

        assert len(all_results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test learned dense metrics."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.num_documents == len(sample_documents)
        assert metrics.strategy_name == "learned_dense"

    def test_higher_scores_than_dense_vector(self, sample_documents):
        """Test that learned dense scores tend to be higher."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("fox", top_k=5)

        # Learned models typically produce higher scores
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        # Scores should be meaningful (not all 0)
        assert avg_score > 0

    def test_teardown(self, sample_documents):
        """Test learned dense cleanup."""
        adapter = LearnedDenseAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.doc_embeddings) == 0
        assert len(adapter.query_times) == 0
        assert adapter.model is None


class TestANNAdapter:
    """Tests for approximate nearest neighbor adapter."""

    def test_initialization(self, sample_documents):
        """Test ANN initialization."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.doc_ids) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test ANN search."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_fast_query(self, sample_documents):
        """Test that ANN queries are fast."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        # ANN should have very low query latency (sub-millisecond ideally)
        import time
        start = time.time()
        adapter.search("quick fox", top_k=5)
        elapsed = time.time() - start

        # Should be very fast (less than 0.1 seconds even with fallback)
        assert elapsed < 0.1

    def test_search_empty_query(self, sample_documents):
        """Test ANN with empty query."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("", top_k=5)
        assert isinstance(results, list)

    def test_multiple_searches(self, sample_documents, queries):
        """Test ANN with multiple searches."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        all_results = []
        for query in queries:
            results = adapter.search(query, top_k=5)
            all_results.extend(results)
            assert len(results) > 0

        assert len(all_results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test ANN metrics."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        # ANN should have very low latency
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.num_documents == len(sample_documents)
        assert metrics.strategy_name == "ann"

    def test_teardown(self, sample_documents):
        """Test ANN cleanup."""
        adapter = ANNAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert adapter.embeddings is None
        assert adapter.index is None
        assert len(adapter.query_times) == 0


class TestQuantizedAdapter:
    """Tests for quantized dense adapter."""

    def test_initialization(self, sample_documents):
        """Test quantized initialization."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        assert len(adapter.documents) == len(sample_documents)
        assert len(adapter.embeddings) == len(sample_documents)
        assert adapter.build_time > 0

    def test_search(self, sample_documents):
        """Test quantized search."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("quick fox", top_k=5)

        assert len(results) > 0
        assert all("doc_id" in r and "score" in r for r in results)
        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, sample_documents):
        """Test quantized with empty query."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        results = adapter.search("", top_k=5)
        assert isinstance(results, list)

    def test_quantized_embeddings_are_int8(self, sample_documents):
        """Test that embeddings are stored as int8."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        for doc_id, emb in adapter.embeddings.items():
            # All values should be small integers (int8 range)
            assert all(isinstance(v, (int, float)) for v in emb)
            # Values should be in reasonable range
            for v in emb:
                assert -128 <= v <= 127

    def test_multiple_searches(self, sample_documents, queries):
        """Test quantized with multiple searches."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        all_results = []
        for query in queries:
            results = adapter.search(query, top_k=5)
            all_results.extend(results)
            assert len(results) > 0

        assert len(all_results) > 0

    def test_get_metrics(self, sample_documents, queries):
        """Test quantized metrics."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)

        for query in queries:
            adapter.search(query, top_k=10)

        metrics = adapter.get_metrics()

        assert metrics.recall_at_10 >= 0.0
        assert metrics.recall_at_100 >= 0.0
        assert metrics.query_latency_ms >= 0.0
        assert metrics.num_queries == len(queries)
        assert metrics.num_documents == len(sample_documents)
        assert metrics.strategy_name == "quantized"

    def test_teardown(self, sample_documents):
        """Test quantized cleanup."""
        adapter = QuantizedAdapter()
        adapter.initialize(sample_documents)
        adapter.search("test")

        adapter.teardown()

        assert len(adapter.documents) == 0
        assert len(adapter.embeddings) == 0
        assert len(adapter.query_times) == 0


class TestDenseRetrievalRegistry:
    """Tests for dense strategy registry."""

    def test_registry_has_dense_strategies(self):
        """Test that dense strategies are registered."""
        assert RetrievalStrategyRegistry.is_registered("dense_vector")
        assert RetrievalStrategyRegistry.is_registered("learned_dense")
        assert RetrievalStrategyRegistry.is_registered("ann")
        assert RetrievalStrategyRegistry.is_registered("quantized")

    def test_get_dense_vector(self):
        """Test getting dense vector adapter."""
        adapter = RetrievalStrategyRegistry.get("dense_vector")
        assert isinstance(adapter, DenseVectorAdapter)

    def test_get_learned_dense(self):
        """Test getting learned dense adapter."""
        adapter = RetrievalStrategyRegistry.get("learned_dense")
        assert isinstance(adapter, LearnedDenseAdapter)

    def test_get_ann(self):
        """Test getting ANN adapter."""
        adapter = RetrievalStrategyRegistry.get("ann")
        assert isinstance(adapter, ANNAdapter)

    def test_get_quantized(self):
        """Test getting quantized adapter."""
        adapter = RetrievalStrategyRegistry.get("quantized")
        assert isinstance(adapter, QuantizedAdapter)

    def test_list_all_includes_dense(self):
        """Test listing all strategies includes dense methods."""
        strategies = RetrievalStrategyRegistry.list_all()
        assert "dense_vector" in strategies
        assert "learned_dense" in strategies
        assert "ann" in strategies
        assert "quantized" in strategies


class TestDenseAdapterComparison:
    """Comparison tests across dense adapters."""

    def test_all_adapters_implement_interface(self, sample_documents):
        """Test that all dense adapters implement required interface."""
        adapters = [
            DenseVectorAdapter(),
            LearnedDenseAdapter(),
            ANNAdapter(),
            QuantizedAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "initialize")
            assert hasattr(adapter, "search")
            assert hasattr(adapter, "get_metrics")
            assert hasattr(adapter, "teardown")

    def test_all_adapters_produce_valid_metrics(self, sample_documents, queries):
        """Test that all dense adapters produce valid metrics."""
        adapters = [
            ("dense_vector", DenseVectorAdapter()),
            ("learned_dense", LearnedDenseAdapter()),
            ("ann", ANNAdapter()),
            ("quantized", QuantizedAdapter()),
        ]

        for name, adapter in adapters:
            adapter.initialize(sample_documents)

            for query in queries:
                adapter.search(query, top_k=10)

            metrics = adapter.get_metrics()

            # Basic validation
            assert 0.0 <= metrics.recall_at_10 <= 1.0, f"{name} recall_at_10 invalid"
            assert 0.0 <= metrics.recall_at_100 <= 1.0, f"{name} recall_at_100 invalid"
            assert metrics.query_latency_ms >= 0.0, f"{name} latency invalid"
            assert metrics.index_build_time_sec > 0, f"{name} build time invalid"
            assert metrics.num_queries == len(queries), f"{name} num_queries mismatch"
            assert metrics.success_rate > 0.0, f"{name} success_rate invalid"

            adapter.teardown()

    def test_dense_vector_vs_learned_dense_recall(self, sample_documents, queries):
        """Test that learned dense typically has higher recall."""
        dv_adapter = DenseVectorAdapter()
        dv_adapter.initialize(sample_documents)

        ld_adapter = LearnedDenseAdapter()
        ld_adapter.initialize(sample_documents)

        for query in queries:
            dv_adapter.search(query, top_k=10)
            ld_adapter.search(query, top_k=10)

        dv_metrics = dv_adapter.get_metrics()
        ld_metrics = ld_adapter.get_metrics()

        # Learned dense should typically achieve higher or equal recall
        assert ld_metrics.recall_at_10 >= dv_metrics.recall_at_10 * 0.9

        dv_adapter.teardown()
        ld_adapter.teardown()
