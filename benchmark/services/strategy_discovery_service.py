"""Strategy discovery service using registry pattern.

Replaces hardcoded strategy discovery with registry-based discovery,
enabling plugin-like system where new strategies work automatically.

This service queries the RetrievalStrategyRegistry to determine which
strategies are available based on installed dependencies.
"""

from dataclasses import dataclass
from typing import Optional

from benchmark.factory.registry import RetrievalStrategyRegistry
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategyInfo:
    """Information about a strategy's availability."""

    name: str
    """Strategy identifier (e.g., 'bm25', 'embeddings')."""

    available: bool
    """Whether this strategy can be used."""

    reason: Optional[str] = None
    """Human-readable reason why unavailable (if applicable)."""


class StrategyAvailabilityService:
    """Discover available retrieval strategies using registry.

    This service replaces 52 lines of hardcoded try/except blocks
    with a clean registry-based API that enables extensibility.

    Usage:
        service = StrategyAvailabilityService(strategy_registry)
        available = service.available_names(allowlist=['bm25', 'embeddings'])
        for strategy_name in available:
            # Use strategy...
    """

    def __init__(self, strategy_registry: RetrievalStrategyRegistry) -> None:
        """Initialize discovery service.

        Args:
            strategy_registry: The registry containing strategy definitions.
        """
        self.registry = strategy_registry

    def discover(
        self,
        allowlist: Optional[list[str]] = None,
        embedding_candidates: Optional[list[tuple[str, str]]] = None,
        reranker_enabled: bool = True,
    ) -> dict[str, StrategyInfo]:
        """Discover all available strategies.

        Returns dict mapping strategy name to StrategyInfo indicating
        whether each strategy is available and why (if not).

        Args:
            allowlist: Restrict to only these strategy names (optional).
            embedding_candidates: List of available embedding models.
            reranker_enabled: Whether reranker models are available.

        Returns:
            Mapping of strategy_name → StrategyInfo.
        """
        results = {}

        # Query registry for what was successfully imported
        for strategy_name in self.registry.registered_names():
            # Apply allowlist filter if provided
            if allowlist and strategy_name not in allowlist:
                results[strategy_name] = StrategyInfo(
                    name=strategy_name,
                    available=False,
                    reason="Not in allowlist",
                )
                continue

            # Special handling for strategies requiring external data
            if strategy_name == "embeddings" and not embedding_candidates:
                results[strategy_name] = StrategyInfo(
                    name=strategy_name,
                    available=False,
                    reason="No embedding models available",
                )
                continue

            if strategy_name in ("llm_rerank", "llm") and not reranker_enabled:
                results[strategy_name] = StrategyInfo(
                    name=strategy_name,
                    available=False,
                    reason="Reranker not enabled",
                )
                continue

            # Strategies are available if registered and passed checks
            results[strategy_name] = StrategyInfo(
                name=strategy_name,
                available=True,
                reason=None,
            )

        return results

    def available_names(
        self,
        allowlist: Optional[list[str]] = None,
        embedding_candidates: Optional[list[tuple[str, str]]] = None,
    ) -> list[str]:
        """Get list of available strategy names.

        Args:
            allowlist: Restrict to only these strategy names (optional).
            embedding_candidates: List of available embedding models.

        Returns:
            List of available strategy names.
        """
        results = self.discover(
            allowlist=allowlist,
            embedding_candidates=embedding_candidates,
        )
        return [name for name, info in results.items() if info.available]

    def get_info(self, strategy_name: str) -> StrategyInfo:
        """Get information about a specific strategy.

        Args:
            strategy_name: The strategy to check.

        Returns:
            StrategyInfo with availability and reason (if unavailable).
        """
        all_info = self.discover()
        return all_info.get(
            strategy_name,
            StrategyInfo(
                name=strategy_name,
                available=False,
                reason="Strategy not registered",
            ),
        )

    def log_availability(self) -> None:
        """Log availability status of all strategies."""
        all_info = self.discover()
        available = [name for name, info in all_info.items() if info.available]
        unavailable = [
            (name, info.reason)
            for name, info in all_info.items()
            if not info.available
        ]

        logger.info(f"Available strategies: {', '.join(available)}")
        if unavailable:
            reasons = ", ".join(
                [f"{name} ({reason})" for name, reason in unavailable]
            )
            logger.debug(f"Unavailable strategies: {reasons}")
