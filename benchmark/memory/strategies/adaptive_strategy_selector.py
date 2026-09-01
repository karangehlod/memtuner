"""Query-adaptive retrieval strategy selection engine."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

QueryType = Literal["factual", "semantic", "exact", "complex"]


@dataclass
class StrategyRecommendation:
    """Recommendation for a retrieval strategy."""
    strategy_name: str
    confidence: float  # 0-1 score
    reason: str
    success_rate: float = 0.5  # Based on feedback history


class AdaptiveStrategySelector:
    """Automatically select best retrieval strategy per query."""

    def __init__(self, strategies: list[str] | None = None):
        """Initialize strategy selector.

        Args:
            strategies: List of available strategy names
                Default: Common retrieval strategies
        """
        self.strategies = strategies or [
            "bm25",
            "dense_vector",
            "colbert",
            "cascading",
            "hybrid_fusion",
        ]

        # Default strategy for each query type
        self._type_to_strategy: dict[QueryType, str] = {
            "factual": "bm25",
            "semantic": "dense_vector",
            "exact": "colbert",
            "complex": "cascading",
        }

        # Learning from feedback: strategy success per query type
        self._feedback_history: dict[QueryType, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Strategy success tracking
        self._success_counts: dict[str, int] = defaultdict(int)
        self._total_counts: dict[str, int] = defaultdict(int)

    def classify_query(self, query: str) -> QueryType:
        """Classify query into type.

        Args:
            query: Query string

        Returns:
            Query type (factual, semantic, exact, complex)
        """
        query_lower = query.lower().strip()

        # Factual indicators
        factual_keywords = ["what", "who", "when", "where", "how many", "which"]
        if any(query_lower.startswith(kw) for kw in factual_keywords):
            return "factual"

        # Exact phrase indicators
        if '"' in query or "exact" in query_lower:
            return "exact"

        # Complex indicators
        complex_keywords = ["compare", "contrast", "relate", "synthesize", "how", "why"]
        if any(kw in query_lower for kw in complex_keywords):
            return "complex"

        # Default: semantic
        return "semantic"

    def select_strategy(self, query: str) -> str:
        """Select best strategy for query.

        Args:
            query: Query string

        Returns:
            Recommended strategy name

        Raises:
            ValueError: If no strategies available
        """
        recs = self.get_recommendations(query)
        if not recs:
            raise ValueError("No strategies available")
        return recs[0].strategy_name

    def get_recommendations(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[StrategyRecommendation]:
        """Get ranked strategy recommendations.

        Args:
            query: Query string
            top_k: Number of recommendations to return

        Returns:
            List of recommendations ranked by confidence

        Raises:
            ValueError: If top_k invalid or no strategies available
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if not self.strategies:
            raise ValueError("No strategies available")

        query_type = self.classify_query(query)

        recommendations = []

        # Get primary recommendation
        primary = self._type_to_strategy.get(query_type)
        if primary and primary in self.strategies:
            rec = self._build_recommendation(primary, query_type)
            recommendations.append(rec)

        # Get other strategies ranked by success rate
        other_strategies = [s for s in self.strategies if s != primary]
        sorted_others = sorted(
            other_strategies,
            key=lambda s: self._get_success_rate(s),
            reverse=True,
        )

        for strategy in sorted_others[:top_k - 1]:
            rec = self._build_recommendation(strategy, query_type)
            recommendations.append(rec)

        return recommendations[:top_k]

    def learn_from_feedback(
        self,
        query: str,
        strategy: str,
        successful: bool,
    ) -> None:
        """Learn from feedback about strategy performance.

        Args:
            query: Query that was executed
            strategy: Strategy that was used
            successful: Whether it succeeded

        Raises:
            ValueError: If strategy unknown
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        query_type = self.classify_query(query)
        self._feedback_history[query_type][strategy].append(successful)

        # Track overall success
        if strategy not in self._total_counts:
            self._total_counts[strategy] = 0
            self._success_counts[strategy] = 0

        self._total_counts[strategy] += 1
        if successful:
            self._success_counts[strategy] += 1

    def get_strategy_stats(self) -> dict[str, dict[str, float]]:
        """Get performance statistics per strategy.

        Returns:
            Dict mapping strategy → {success_rate, attempts, successes}
        """
        stats = {}

        for strategy in self.strategies:
            total = self._total_counts.get(strategy, 0)
            successes = self._success_counts.get(strategy, 0)

            stats[strategy] = {
                "success_rate": (successes / total) if total > 0 else 0.5,
                "attempts": total,
                "successes": successes,
            }

        return stats

    def get_recommendations_for_type(
        self,
        query_type: QueryType,
    ) -> list[StrategyRecommendation]:
        """Get recommendations for a specific query type.

        Args:
            query_type: Type of query

        Returns:
            Recommendations ranked by success

        Raises:
            ValueError: If query_type unknown
        """
        if query_type not in self._type_to_strategy:
            raise ValueError(f"Unknown query type: {query_type}")

        primary = self._type_to_strategy[query_type]
        recommendations = []

        if primary in self.strategies:
            rec = self._build_recommendation(primary, query_type)
            recommendations.append(rec)

        other_strategies = [s for s in self.strategies if s != primary]
        sorted_others = sorted(
            other_strategies,
            key=lambda s: self._get_success_rate(s),
            reverse=True,
        )

        for strategy in sorted_others:
            rec = self._build_recommendation(strategy, query_type)
            recommendations.append(rec)

        return recommendations

    def reset_feedback(self) -> None:
        """Reset all learning feedback."""
        self._feedback_history.clear()
        self._success_counts.clear()
        self._total_counts.clear()

    # Private helper methods

    def _build_recommendation(
        self,
        strategy: str,
        query_type: QueryType,
    ) -> StrategyRecommendation:
        """Build a recommendation object."""
        is_primary = strategy == self._type_to_strategy.get(query_type)
        success_rate = self._get_success_rate(strategy)

        # Confidence based on whether it's primary + success rate
        confidence = 0.8 if is_primary else 0.5
        confidence = confidence * (0.5 + success_rate * 0.5)  # Boost by success rate

        # Reason
        if is_primary:
            reason = f"Primary strategy for {query_type} queries"
        else:
            reason = f"Alternative strategy (success rate: {success_rate:.1%})"

        return StrategyRecommendation(
            strategy_name=strategy,
            confidence=confidence,
            reason=reason,
            success_rate=success_rate,
        )

    def _get_success_rate(self, strategy: str) -> float:
        """Get success rate for a strategy."""
        total = self._total_counts.get(strategy, 0)
        if total == 0:
            return 0.5  # Neutral prior

        successes = self._success_counts.get(strategy, 0)
        return successes / total
