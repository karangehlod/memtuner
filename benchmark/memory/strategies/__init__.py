"""Built-in retrieval strategies for memory systems.

Strategies:
- BM25: Fast keyword matching
- Embeddings: Semantic similarity (local sentence-transformers)
- ApiEmbeddings: Semantic similarity via OpenAI-compatible API endpoint
- LLM: LLM-powered ranking
- Database: PostgreSQL backend
- Hybrid: Multi-strategy fallback

Each strategy is lazily imported so missing optional dependencies
don't break the entire package.
"""

__all__ = [
    "ApiEmbeddingsStrategy",
    "BM25Strategy",
    "DatabaseStrategy",
    "EmbeddingsStrategy",
    "HybridStrategy",
    "LLMStrategy",
]


def __getattr__(name: str):
    if name == "BM25Strategy":
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy

        return BM25Strategy
    if name == "EmbeddingsStrategy":
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy

        return EmbeddingsStrategy
    if name == "ApiEmbeddingsStrategy":
        from benchmark.memory.strategies.api_embeddings_strategy import ApiEmbeddingsStrategy

        return ApiEmbeddingsStrategy
    if name == "LLMStrategy":
        from benchmark.memory.strategies.llm_strategy import LLMStrategy

        return LLMStrategy
    if name == "DatabaseStrategy":
        from benchmark.memory.strategies.database_strategy import DatabaseStrategy

        return DatabaseStrategy
    if name == "HybridStrategy":
        from benchmark.memory.strategies.hybrid_strategy import HybridStrategy

        return HybridStrategy
    raise AttributeError(f"module 'benchmark.memory.strategies' has no attribute {name!r}")
