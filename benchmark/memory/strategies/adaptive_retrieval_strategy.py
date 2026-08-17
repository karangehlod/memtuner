"""Adaptive retrieval strategy — per-query dispatch to BM25 or embeddings.

Wraps AdaptiveStrategySelector: classifies each query into one of
{factual, semantic, exact, complex} and routes to the most appropriate
underlying strategy. This measures the upper-bound benefit of knowing
the query type at retrieval time — a natural ablation for paper §5.

Routing logic:
    factual  → BM25   (who/what/when/where — keyword-heavy)
    exact    → BM25   (quoted phrases, exact terms)
    semantic → embeddings (paraphrase, concept, theme)
    complex  → hybrid (multi-hop — both keyword and semantic signals)

The strategy learns from recall feedback via learn_from_feedback(), which
lets it shift routing over the course of a benchmark run. Feedback is
off by default so Phase 1 baseline results are reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.memory.strategies.adaptive_strategy_selector import AdaptiveStrategySelector

if TYPE_CHECKING:
    from benchmark.models.memory_event import MemoryEvent

# Lazy imports — only loaded if the adaptive strategy is used
_bm25_cls = None
_embed_cls = None
_hybrid_cls = None


def _get_bm25():
    global _bm25_cls
    if _bm25_cls is None:
        from benchmark.memory.strategies.bm25_strategy import BM25Strategy
        _bm25_cls = BM25Strategy
    return _bm25_cls()


def _get_embed():
    global _embed_cls
    if _embed_cls is None:
        from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy
        _embed_cls = EmbeddingsStrategy
    return _embed_cls()


def _get_hybrid(bm25_inst=None, embed_inst=None):
    global _hybrid_cls
    if _hybrid_cls is None:
        from benchmark.memory.strategies.hybrid_strategy import HybridStrategy
        _hybrid_cls = HybridStrategy
    return _hybrid_cls(bm25_strategy=bm25_inst, embeddings_strategy=embed_inst)


# Map classifier output → strategy key
_ROUTE: dict[str, str] = {
    "factual":  "bm25",
    "exact":    "bm25",
    "semantic": "embeddings",
    "complex":  "hybrid",
}


class AdaptiveRetrievalStrategy(RetrievalStrategy):
    """Per-query strategy routing based on query-type classification.

    Benchmarking this strategy answers: "How much recall do we gain if we
    always use the right strategy for each query?" — an oracle upper bound
    for adaptive memory system design.

    All three underlying strategies (BM25, Embeddings, Hybrid) are indexed
    at startup so routing is zero-cost at query time.
    """

    def __init__(self) -> None:
        self._selector = AdaptiveStrategySelector(
            strategies=["bm25", "embeddings", "hybrid"]
        )
        # Override type→strategy routing to match benchmark strategy names
        self._selector._type_to_strategy = {
            "factual":  "bm25",
            "exact":    "bm25",
            "semantic": "embeddings",
            "complex":  "hybrid",
        }

        self._strategies: dict[str, RetrievalStrategy] = {}
        self._indexed = False
        # routing stats for introspection / paper analysis
        self.routing_counts: dict[str, int] = {
            "bm25": 0, "embeddings": 0, "hybrid": 0
        }

    @property
    def name(self) -> str:
        return "adaptive"

    def clear(self) -> None:
        for strat in self._strategies.values():
            strat.clear()
        self._indexed = False
        self.routing_counts = {"bm25": 0, "embeddings": 0, "hybrid": 0}

    def index(self, memories: list["MemoryEvent"]) -> None:
        bm25_inst = _get_bm25()
        embed_inst = _get_embed()
        hybrid_inst = _get_hybrid(bm25_inst=bm25_inst, embed_inst=embed_inst)

        bm25_inst.index(memories)
        embed_inst.index(memories)
        # Hybrid reuses the already-indexed sub-strategies — no double indexing.
        hybrid_inst.index(memories)

        self._strategies = {
            "bm25":       bm25_inst,
            "embeddings": embed_inst,
            "hybrid":     hybrid_inst,
        }
        self._indexed = True
        self.routing_counts = {"bm25": 0, "embeddings": 0, "hybrid": 0}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if not self._indexed:
            return []

        query_type = self._selector.classify_query(query)
        strategy_name = _ROUTE.get(query_type, "bm25")
        self.routing_counts[strategy_name] = (
            self.routing_counts.get(strategy_name, 0) + 1
        )

        strat = self._strategies[strategy_name]
        return strat.retrieve(query, top_k=top_k, user_id=user_id)

    def get_routing_summary(self) -> dict[str, float]:
        """Return fraction of queries routed to each strategy — for paper tables."""
        total = sum(self.routing_counts.values()) or 1
        return {k: v / total for k, v in self.routing_counts.items()}
