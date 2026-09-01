"""Hybrid retrieval strategy combining multiple approaches.

Tries strategies in order: BM25 → Embeddings → LLM
Falls back to next strategy if current one has low confidence.

Latency: 10-2000ms (adaptive) | Cost: Low-High | Accuracy: Best | Setup: 4 hours
"""

from __future__ import annotations

import heapq

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent


class HybridStrategy(RetrievalStrategy):
    """Hybrid retrieval combining BM25 keyword matching with semantic embeddings.

    Uses score-weighted fusion with configurable alpha:
        final_score = alpha × BM25_normalized + (1 - alpha) × embeddings_normalized

    Higher alpha favors keyword precision (good for name/entity queries).
    Lower alpha favors semantic recall (good for paraphrased queries).
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        confidence_threshold: float = 0.5,
        bm25_weight: float = 0.5,
        bm25_strategy: RetrievalStrategy | None = None,
        embeddings_strategy: RetrievalStrategy | None = None,
        llm_strategy: RetrievalStrategy | None = None,
    ) -> None:
        """Initialize hybrid strategy.

        Args:
            strategies: List of strategy names to use.
                       Default: ['bm25', 'embeddings']
            confidence_threshold: Min score to stop trying strategies.
            bm25_weight: Weight for BM25 in fusion (0.0-1.0).
                        embeddings_weight = 1.0 - bm25_weight.
                        Default 0.5 (equal weighting).
        """
        self.strategies_config = strategies or ["bm25", "embeddings"]
        self.confidence_threshold = confidence_threshold
        self._bm25_weight = max(0.0, min(1.0, bm25_weight))
        self._strategy_instances: dict[str, RetrievalStrategy] = {}
        if bm25_strategy is not None:
            self._strategy_instances["bm25"] = bm25_strategy
        if embeddings_strategy is not None:
            self._strategy_instances["embeddings"] = embeddings_strategy
        if llm_strategy is not None:
            self._strategy_instances["llm"] = llm_strategy

    def index(self, memories: list[MemoryEvent]) -> None:
        """Index memories in all active strategies.

        Args:
            memories: List of memories to index.
        """
        for strategy in self._strategy_instances.values():
            strategy.index(memories)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve using Reciprocal Rank Fusion (RRF).

        RRF combines ranked lists from multiple strategies without score
        normalization. Each document gets a fused score based on its rank
        position in each strategy's output:

            rrf_score(doc) = Σ weight_i / (k + rank_i)

        where k=60 is the standard RRF constant and weight_i is the
        configurable weight for each strategy (bm25_weight for BM25,
        1 - bm25_weight for embeddings).

        RRF is proven to outperform score-based fusion because:
        1. No score normalization needed (avoids distortion)
        2. Rank-based fusion is robust to different score scales
        3. Documents that appear in BOTH lists get boosted naturally

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, rrf_score) tuples, sorted descending.

        Raises:
            RuntimeError: If primary strategy (BM25) fails.
        """
        missing = [
            strategy_name
            for strategy_name in self.strategies_config
            if strategy_name not in self._strategy_instances
        ]
        if missing:
            raise RuntimeError(
                "Hybrid strategy requires injected sub-strategies for: "
                + ", ".join(missing)
            )

        RRF_K = 60  # Standard RRF constant (from Cormack et al. 2009)

        # Over-fetch from each strategy for better fusion coverage
        sub_k = max(top_k * 4, 40)

        # Collect ranked lists from each strategy
        all_ranked: dict[str, list[tuple[str, float]]] = {}

        # BM25 is primary — must not fail
        if "bm25" in self._strategy_instances:
            try:
                results = self._strategy_instances["bm25"].retrieve(query, sub_k, user_id)
                if results:
                    all_ranked["bm25"] = results
            except Exception as e:
                raise RuntimeError(f"Primary strategy (BM25) failed: {e}") from e

        # Secondary strategies can fail gracefully
        for strategy_name in self.strategies_config:
            if strategy_name == "bm25":
                continue
            if strategy_name not in self._strategy_instances:
                continue
            try:
                results = self._strategy_instances[strategy_name].retrieve(query, sub_k, user_id)
                if results:
                    all_ranked[strategy_name] = results
            except Exception:
                pass  # Secondary failure is non-fatal

        if not all_ranked:
            return []

        # If only one strategy produced results, return directly
        if len(all_ranked) == 1:
            return list(all_ranked.values())[0][:top_k]

        # Reciprocal Rank Fusion with configurable weights
        strategy_weights = {
            "bm25": self._bm25_weight,
            "embeddings": 1.0 - self._bm25_weight,
        }

        rrf_scores: dict[str, float] = {}
        for strat_name, ranked in all_ranked.items():
            weight = strategy_weights.get(strat_name, 0.5)
            for rank, (mem_id, _score) in enumerate(ranked):
                rrf_contribution = weight / (RRF_K + rank + 1)
                rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + rrf_contribution

        # heapq.nlargest is O(M log K) vs O(M log M) full sort; M≈80, K≤10 → ~2× faster
        return heapq.nlargest(top_k, rrf_scores.items(), key=lambda x: x[1])

    def name(self) -> str:
        """Return strategy name."""
        strategies = "+".join(self.strategies_config)
        return f"hybrid({strategies})"

    def clear(self) -> None:
        """Clear all strategies."""
        for strategy in self._strategy_instances.values():
            strategy.clear()

    @classmethod
    def is_available(cls) -> bool:
        """Hybrid is always available (uses available sub-strategies)."""
        return True
