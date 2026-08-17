"""Cost tracker interface and composite tracker.

Defines the contract for cost tracking and provides a composite
that aggregates across multiple cost sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.cost.models import CostEntry
from benchmark.cost.token_cost import TokenCostCalculator
from benchmark.models.answer import TokenUsage
from benchmark.tokenizer.interface import Tokenizer


class CostTracker(ABC):
    """Abstract interface for tracking costs during benchmark execution.

    Each implementation tracks one category of cost (tokens, storage, etc.).
    The CompositeCostTracker aggregates across all sources.
    """

    @abstractmethod
    def record(self, entry: CostEntry) -> None:
        """Record a single cost entry.

        Args:
            entry: The cost entry to record.
        """

    @abstractmethod
    def total_cost_usd(self) -> float:
        """Return the total accumulated cost in USD.

        Returns:
            Total cost in US dollars.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker for a new benchmark run."""


class InMemoryCostTracker(CostTracker):
    """In-memory cost tracker that accumulates cost entries.

    Thread-safe is NOT required — benchmarks are single-threaded.
    """

    def __init__(self) -> None:
        """Initialize with empty cost entries."""
        self._entries: list[CostEntry] = []

    def record(self, entry: CostEntry) -> None:
        """Record a cost entry.

        Args:
            entry: The cost entry to record.
        """
        self._entries.append(entry)

    def total_cost_usd(self) -> float:
        """Return total accumulated cost.

        Returns:
            Sum of all entry amounts in USD.
        """
        return sum(entry.amount_usd for entry in self._entries)

    def entries_by_source(self) -> dict[str, float]:
        """Return costs grouped by source.

        Returns:
            Dictionary mapping source → total cost for that source.
        """
        grouped: dict[str, float] = {}
        for entry in self._entries:
            grouped[entry.source] = grouped.get(entry.source, 0.0) + entry.amount_usd
        return grouped

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()


class CompositeCostTracker(CostTracker):
    """Composite cost tracker that aggregates token + storage costs.

    It can be configured with a Tokenizer instance which will be used by
    the TokenCostCalculator to deterministically compute token counts
    when TokenUsage is not provided.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._entries: list[CostEntry] = []
        self._token_cost_calculator = TokenCostCalculator(tokenizer=tokenizer)

    def record(self, entry: CostEntry) -> None:
        self._entries.append(entry)

    def total_cost_usd(self) -> float:
        return sum(entry.amount_usd for entry in self._entries)

    def entries_by_source(self) -> dict[str, float]:
        grouped: dict[str, float] = {}
        for entry in self._entries:
            grouped[entry.source] = grouped.get(entry.source, 0.0) + entry.amount_usd
        return grouped

    def reset(self) -> None:
        self._entries.clear()

    def record_llm_cost(
        self,
        token_usage: TokenUsage | None,
        model: str,
        prompt_text: str | None = None,
        completion_text: str | None = None,
    ) -> None:
        entry = self._token_cost_calculator.compute_cost(
            token_usage, model, prompt_text, completion_text
        )
        self.record(entry)
