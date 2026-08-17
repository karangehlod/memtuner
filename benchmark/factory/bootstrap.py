"""Bootstrap module to register built-in retrieval strategies.

Called at startup to populate the retrieval strategy registry with all
available strategies. Strategies with missing optional dependencies are
silently skipped at registration time — they fail fast at resolution time
if explicitly requested.
"""

from __future__ import annotations

import importlib

from benchmark.factory.registry import RetrievalStrategyRegistry


def bootstrap_retrieval_strategies(
    registry: RetrievalStrategyRegistry,
) -> None:
    """Register all built-in retrieval strategies.

    Strategies are registered only if their dependencies are installed.
    Missing optional dependencies cause the strategy to be skipped —
    requesting an unregistered strategy later raises StrategyResolutionError
    with an install hint.

    Args:
        registry: The strategy registry to populate.
    """
    strategies = [
        ("recency", "benchmark.memory.strategies.recency_strategy", "RecencyStrategy"),
        ("bm25", "benchmark.memory.strategies.bm25_strategy", "BM25Strategy"),
        ("embeddings", "benchmark.memory.strategies.embeddings_strategy", "EmbeddingsStrategy"),
        (
            "api_embeddings",
            "benchmark.memory.strategies.api_embeddings_strategy",
            "ApiEmbeddingsStrategy",
        ),
        (
            "session_embeddings",
            "benchmark.memory.strategies.session_embeddings_strategy",
            "SessionEmbeddingsStrategy",
        ),
        ("hybrid", "benchmark.memory.strategies.hybrid_strategy", "HybridStrategy"),
        ("pgvector", "benchmark.memory.strategies.pgvector_strategy", "PgVectorStrategy"),
        ("llm_rerank", "benchmark.memory.strategies.llm_rerank_strategy", "LLMRerankStrategy"),
        ("llm", "benchmark.memory.strategies.llm_strategy", "LLMStrategy"),
        ("database", "benchmark.memory.strategies.database_strategy", "DatabaseStrategy"),
        # BM25L — BM25 variant with lower-bounded TF for better long-document recall
        ("bm25l", "benchmark.memory.strategies.bm25l_strategy", "BM25LStrategy"),
        # ColBERT-style — token-level MaxSim scoring (better for exact phrase queries)
        ("colbert", "benchmark.memory.strategies.colbert_strategy", "ColBERTStrategy"),
        # Adaptive — per-query routing to BM25/embeddings/hybrid based on query type
        ("adaptive", "benchmark.memory.strategies.adaptive_retrieval_strategy", "AdaptiveRetrievalStrategy"),
    ]

    for strategy_name, module_path, class_name in strategies:
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)

            registry.register(strategy_name, strategy_class)
        except ImportError:
            # Optional dependency not installed — skip silently.
            # Fail-fast happens at resolution time if this strategy is requested.
            pass
