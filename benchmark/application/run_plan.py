"""Run plan — immutable record of how a benchmark run was composed.

Every benchmark run produces a RunPlan that documents exactly what
was wired, resolved, and configured. This makes runs inspectable
and ensures CLI and matrix produce equivalent setups.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

METRIC_SEMANTICS_VERSION: str = "2.0"


@dataclass(frozen=True)
class RunPlan:
    """Immutable record of a composed benchmark run.

    Contains everything needed to understand what was configured,
    what was actually resolved, and what parameters govern execution.
    """

    # Strategy resolution
    requested_strategy: str
    effective_strategy: str
    resolved_strategy_class: str

    # Memory modules
    memory_modules: tuple[str, ...]
    lifecycle_policies: tuple[str, ...]

    # Dataset identity
    dataset_fingerprint: str
    dataset_query_count: int
    dataset_memory_count: int
    dataset_user_count: int
    dataset_event_day_count: int

    # Evaluation parameters
    recall_k: int
    metric_semantics_version: str = METRIC_SEMANTICS_VERSION

    # Horizon
    requested_horizon: int | None = None
    effective_horizon: int = 0

    # Normalization
    normalization_applied: bool = False
    normalization_delta_days: int = 0

    # Configuration identity
    config_hash: str = ""
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric_semantics_version": self.metric_semantics_version,
            "strategy": {
                "requested": self.requested_strategy,
                "effective": self.effective_strategy,
                "resolved_class": self.resolved_strategy_class,
            },
            "memory_modules": list(self.memory_modules),
            "lifecycle_policies": list(self.lifecycle_policies),
            "dataset": {
                "fingerprint": self.dataset_fingerprint,
                "query_count": self.dataset_query_count,
                "memory_count": self.dataset_memory_count,
                "user_count": self.dataset_user_count,
                "event_day_count": self.dataset_event_day_count,
            },
            "evaluation": {
                "recall_k": self.recall_k,
            },
            "horizon": {
                "requested": self.requested_horizon,
                "effective": self.effective_horizon,
            },
            "normalization": {
                "applied": self.normalization_applied,
                "delta_days": self.normalization_delta_days,
            },
            "config_hash": self.config_hash,
            "seed": self.seed,
        }


def compute_dataset_fingerprint(
    scenario: str,
    query_count: int,
    memory_count: int,
    user_ids: list[str],
) -> str:
    """Compute a deterministic fingerprint for a dataset.

    Args:
        scenario: The scenario name.
        query_count: Number of queries.
        memory_count: Total conversation turns / memory events.
        user_ids: Sorted user IDs.

    Returns:
        A short hex hash identifying this dataset configuration.
    """
    key = f"{scenario}:{query_count}:{memory_count}:{sorted(user_ids)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
