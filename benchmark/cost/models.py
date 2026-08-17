"""Cost data models.

Defines cost tracking data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostEntry:
    """A single cost entry from a benchmark operation.

    Attributes:
        source: What generated this cost (e.g., "llm_tokens", "db_read").
        amount_usd: Cost in US dollars.
        details: Optional breakdown details.
    """

    source: str
    amount_usd: float
    details: dict[str, float] = field(default_factory=dict)
