"""Unit tests for built-in retrieval strategies.

Tests BM25, Embeddings, Hybrid strategies (LLM and Database require
external services so they are not tested here without mocking).
"""

from datetime import datetime

import numpy as np
import pytest

from benchmark.models.memory_event import MemoryEvent, MemoryType


def _make_memory(mem_id: str, content: str, user_id: str = "u1") -> MemoryEvent:
    return MemoryEvent(
        id=mem_id,
        user_id=user_id,
        type=MemoryType.PREFERENCE,
        content=content,
        timestamp=datetime.now(),
        importance=0.7,
        task_id="T-001",
    )


@pytest.fixture
def sample_memories():
    return [
        _make_memory("M-001", "User loves pizza and Italian food"),
        _make_memory("M-002", "User prefers dark mode in their IDE"),
        _make_memory("M-003", "Meeting with John scheduled for Tuesday"),
        _make_memory("M-004", "User likes hiking and outdoor activities", user_id="u2"),
    ]


@pytest.fixture(autouse=True)
def stub_sentence_transformer(monkeypatch):
    from benchmark.memory.strategies import embeddings_strategy

    # The real strategy caches loaded models at module scope keyed by
    # "model_name:device" so it doesn't reload weights per cell. That cache
    # persists across tests in this file — clear it so each test's stub
    # class actually gets constructed instead of silently reusing a
    # previous test's cached (stub) instance.
    monkeypatch.setattr(embeddings_strategy, "_MODEL_CACHE", {})
    monkeypatch.setattr(embeddings_strategy, "_MODEL_CACHE_ORDER", [])
    monkeypatch.setattr(embeddings_strategy, "_INDEX_CACHE", {})
    monkeypatch.setattr(embeddings_strategy, "_INDEX_CACHE_ORDER", [])

    class StubSentenceTransformer:
        init_calls: list[tuple[str, dict[str, object]]] = []

        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name
            self.kwargs = kwargs
            StubSentenceTransformer.init_calls.append((model_name, kwargs))

        def encode(
            self,
            texts,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            show_progress_bar: bool = False,
            batch_size: int = 128,
        ):
            del convert_to_tensor, show_progress_bar, batch_size

            def _encode_one(text: str) -> np.ndarray:
                lower_text = text.lower()
                features = np.array(
                    [
                        1.0 if "pizza" in lower_text or "italian" in lower_text or "food" in lower_text else 0.0,
                        1.0 if "dark mode" in lower_text or "ide" in lower_text else 0.0,
                        1.0 if "meeting" in lower_text or "tuesday" in lower_text or "john" in lower_text else 0.0,
                        1.0 if "hiking" in lower_text or "outdoor" in lower_text else 0.0,
                    ],
                    dtype=np.float32,
                )
                if not features.any():
                    features[0] = 0.1
                if normalize_embeddings:
                    norm = np.linalg.norm(features)
                    if norm > 0:
                        features = features / norm
                return features

            if isinstance(texts, str):
                return _encode_one(texts)
            return [_encode_one(text) for text in texts]

    monkeypatch.setattr(embeddings_strategy, "SentenceTransformer", StubSentenceTransformer)
    return StubSentenceTransformer


# ---------------------------------------------------------------------------
# BM25 Strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBM25Strategy:
    def test_name(self):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        assert BM25Strategy().name() == "bm25"

    def test_is_available(self):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        assert BM25Strategy.is_available() is True

    def test_retrieve_returns_scored_list(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("food preferences", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_retrieve_top_k_respected(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("user", top_k=2)
        assert len(results) <= 2

    def test_retrieve_user_filter(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("user", top_k=10, user_id="u2")
        ids = [r[0] for r in results]
        assert all(
            sample_memories[i].user_id == "u2"
            for i, _ in enumerate(sample_memories)
            if sample_memories[i].id in ids
        )

    def test_best_match_ranked_first(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("pizza Italian food", top_k=2)
        assert results[0][0] == "M-001"

    def test_retrieve_empty_index_returns_empty(self):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index([])
        results = strategy.retrieve("anything", top_k=5)
        assert results == []

    def test_clear_resets_index(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        strategy.clear()
        results = strategy.retrieve("pizza", top_k=5)
        assert results == []

    def test_scores_are_floats(self, sample_memories):
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        strategy = BM25Strategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("user likes", top_k=3)
        assert all(isinstance(score, float) for _, score in results)


# ---------------------------------------------------------------------------
# Embeddings Strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingsStrategy:
    def test_name(self):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        assert EmbeddingsStrategy().name() == "embeddings"

    def test_is_available_reflects_sentence_transformers_import(self, monkeypatch: pytest.MonkeyPatch):
        from benchmark.memory.strategies import embeddings_strategy
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy

        assert EmbeddingsStrategy.is_available() is True

        monkeypatch.setattr(embeddings_strategy, "SentenceTransformer", None)
        assert EmbeddingsStrategy.is_available() is False

    def test_passes_cache_folder_to_sentence_transformer(self, stub_sentence_transformer):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy

        strategy = EmbeddingsStrategy(
            model_name="all-MiniLM-L6-v2",
            cache_dir="/tmp/embed-cache",
        )

        assert strategy.name() == "embeddings"
        model_name, kwargs = stub_sentence_transformer.init_calls[-1]
        assert model_name == "all-MiniLM-L6-v2"
        assert kwargs["cache_folder"] == "/tmp/embed-cache"

    def test_retrieve_returns_scored_list(self, sample_memories):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("what food does the user eat", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_retrieve_top_k_respected(self, sample_memories):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("user preference", top_k=2)
        assert len(results) <= 2

    def test_best_semantic_match_ranked_first(self, sample_memories):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("Italian food pizza", top_k=3)
        assert results[0][0] == "M-001"

    def test_scores_between_minus1_and_1(self, sample_memories):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index(sample_memories)
        results = strategy.retrieve("food", top_k=3)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_retrieve_empty_index_returns_empty(self):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index([])
        results = strategy.retrieve("anything", top_k=5)
        assert results == []

    def test_clear_resets_index(self, sample_memories):
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        strategy = EmbeddingsStrategy()
        strategy.index(sample_memories)
        strategy.clear()
        results = strategy.retrieve("pizza", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Hybrid Strategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHybridStrategy:
    @staticmethod
    def _build_hybrid(strategies: list[str], confidence_threshold: float = 0.5):
        from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        from benchmark.memory.strategies.hybrid_strategy import HybridStrategy

        class StubEmbeddingsStrategy(RetrievalStrategy):
            def __init__(self) -> None:
                self._indexed = False

            def index(self, memories):
                self._indexed = bool(memories)

            def retrieve(self, query: str, top_k: int = 5, user_id: str | None = None):
                if not self._indexed:
                    return []
                return [("stub-embeddings-memory", 0.8)][:top_k]

            def name(self) -> str:
                return "stub-embeddings"

            def clear(self) -> None:
                self._indexed = False

        kwargs = {
            "strategies": strategies,
            "confidence_threshold": confidence_threshold,
        }
        if "bm25" in strategies:
            kwargs["bm25_strategy"] = BM25Strategy()
        if "embeddings" in strategies:
            kwargs["embeddings_strategy"] = StubEmbeddingsStrategy()
        return HybridStrategy(**kwargs)

    def test_name_reflects_strategies(self):
        hybrid = self._build_hybrid(["bm25", "embeddings"])
        assert "hybrid" in hybrid.name()
        assert "bm25" in hybrid.name()

    def test_is_available(self):
        from benchmark.memory.strategies.hybrid_strategy import HybridStrategy
        assert HybridStrategy.is_available() is True

    def test_retrieve_returns_results(self, sample_memories):
        hybrid = self._build_hybrid(["bm25"])
        hybrid.index(sample_memories)
        results = hybrid.retrieve("food", top_k=2)
        assert isinstance(results, list)

    def test_falls_back_when_low_confidence(self, sample_memories):
        # Use very high threshold so BM25 always falls back to embeddings
        hybrid = self._build_hybrid(["bm25", "embeddings"], confidence_threshold=100.0)
        hybrid.index(sample_memories)
        results = hybrid.retrieve("pizza", top_k=2)
        # Should still return results from embeddings fallback
        assert isinstance(results, list)

    def test_clear_clears_all_sub_strategies(self, sample_memories):
        hybrid = self._build_hybrid(["bm25"])
        hybrid.index(sample_memories)
        hybrid.clear()
        results = hybrid.retrieve("pizza", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Bootstrap + Registry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStrategyBootstrap:
    def test_bootstrap_registers_available_strategies(self):
        from benchmark.factory.registry import RetrievalStrategyRegistry
        from benchmark.factory.bootstrap import bootstrap_retrieval_strategies

        registry = RetrievalStrategyRegistry()
        bootstrap_retrieval_strategies(registry)

        # All installed strategies should be registered
        names = registry.registered_names()
        assert "bm25" in names
        assert "embeddings" in names
        assert "hybrid" in names

    def test_registry_resolve_bm25(self):
        from benchmark.factory.registry import RetrievalStrategyRegistry
        from benchmark.factory.bootstrap import bootstrap_retrieval_strategies

        registry = RetrievalStrategyRegistry()
        bootstrap_retrieval_strategies(registry)

        strategy = registry.resolve("bm25")
        assert strategy.name() == "bm25"

    def test_registry_resolve_embeddings(self):
        from benchmark.factory.registry import RetrievalStrategyRegistry
        from benchmark.factory.bootstrap import bootstrap_retrieval_strategies

        registry = RetrievalStrategyRegistry()
        bootstrap_retrieval_strategies(registry)

        strategy = registry.resolve("embeddings")
        assert strategy.name() == "embeddings"


@pytest.mark.unit
class TestRegistry:
    def test_registry_unknown_strategy_raises(self):
        from benchmark.factory.registry import RetrievalStrategyRegistry
        from benchmark.exceptions.memory_errors import RegistryResolutionError

        registry = RetrievalStrategyRegistry()
        with pytest.raises(RegistryResolutionError):
            registry.resolve("nonexistent_strategy")


@pytest.mark.unit
class TestApiEmbeddingsStrategy:
    """ollama_embeddings/hf_inference_embeddings were consolidated into this
    single OpenAI-compatible-endpoint strategy — see api_embeddings_strategy.py.
    """

    @staticmethod
    def _make_strategy(monkeypatch: pytest.MonkeyPatch, embed_fn):
        from benchmark.memory.strategies.api_embeddings_strategy import ApiEmbeddingsStrategy

        monkeypatch.setenv("BENCHMARK_OPENAI_BASE_URL", "http://localhost:11434/v1")
        strategy = ApiEmbeddingsStrategy(model_name="fake-model")
        monkeypatch.setattr(strategy, "_embed_texts", embed_fn)
        return strategy

    @staticmethod
    def _one_hot_embed(texts: list[str]) -> list[np.ndarray]:
        results = []
        for text in texts:
            vec = np.zeros(8, dtype=np.float32)
            vec[hash(text) % 8] = 1.0
            results.append(vec)
        return results

    def test_name(self, monkeypatch: pytest.MonkeyPatch):
        strategy = self._make_strategy(monkeypatch, self._one_hot_embed)
        assert strategy.name() == "api_embeddings"

    def test_requires_base_url(self, monkeypatch: pytest.MonkeyPatch):
        from benchmark.memory.strategies.api_embeddings_strategy import ApiEmbeddingsStrategy

        monkeypatch.delenv("BENCHMARK_OPENAI_BASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="requires a base_url"):
            ApiEmbeddingsStrategy(model_name="fake-model")

    def test_retrieve_respects_top_k(self, monkeypatch: pytest.MonkeyPatch, sample_memories):
        strategy = self._make_strategy(monkeypatch, self._one_hot_embed)
        strategy.index(sample_memories)
        results = strategy.retrieve("food", top_k=2)
        assert len(results) <= 2

    def test_user_isolation_no_cross_user_leakage(self, monkeypatch: pytest.MonkeyPatch, sample_memories):
        # Regression test: retrieve() previously never populated the user
        # mask cache, so user_id filtering silently never applied and
        # other users' memories could leak into a user-scoped result set.
        strategy = self._make_strategy(monkeypatch, self._one_hot_embed)
        strategy.index(sample_memories)

        u1_ids = {m.id for m in sample_memories if m.user_id == "u1"}
        results = strategy.retrieve("user", top_k=10, user_id="u1")

        assert results
        assert all(mem_id in u1_ids for mem_id, _ in results)

    def test_clear_resets_index(self, monkeypatch: pytest.MonkeyPatch, sample_memories):
        strategy = self._make_strategy(monkeypatch, self._one_hot_embed)
        strategy.index(sample_memories)
        strategy.clear()
        assert strategy.retrieve("pizza", top_k=5) == []


# ---------------------------------------------------------------------------
# Interface Contract Tests
# ---------------------------------------------------------------------------

@pytest.mark.contract
class TestRetrievalStrategyContract:
    """Contract tests: all strategies MUST satisfy the RetrievalStrategy interface."""

    @pytest.fixture(params=["bm25", "embeddings"])
    def strategy(self, request):
        from benchmark.factory.registry import RetrievalStrategyRegistry
        from benchmark.factory.bootstrap import bootstrap_retrieval_strategies

        registry = RetrievalStrategyRegistry()
        bootstrap_retrieval_strategies(registry)

        if not registry.is_registered(request.param):
            pytest.skip(f"{request.param} not available")

        return registry.resolve(request.param)

    def test_has_name(self, strategy):
        assert isinstance(strategy.name(), str)
        assert len(strategy.name()) > 0

    def test_index_accepts_empty_list(self, strategy):
        strategy.index([])  # Must not raise

    def test_retrieve_on_empty_returns_list(self, strategy):
        strategy.index([])
        result = strategy.retrieve("query", top_k=5)
        assert isinstance(result, list)

    def test_retrieve_respects_top_k(self, strategy, sample_memories):
        strategy.index(sample_memories)
        results = strategy.retrieve("user", top_k=1)
        assert len(results) <= 1

    def test_retrieve_returns_tuples(self, strategy, sample_memories):
        strategy.index(sample_memories)
        results = strategy.retrieve("food", top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            mem_id, score = item
            assert isinstance(mem_id, str)
            assert isinstance(score, float)

    def test_clear_then_retrieve_returns_empty(self, strategy, sample_memories):
        strategy.index(sample_memories)
        strategy.clear()
        results = strategy.retrieve("pizza", top_k=5)
        assert results == []
