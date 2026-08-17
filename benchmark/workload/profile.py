"""Workload profile definitions.

A workload profile defines the scale and shape of a benchmark run:
- evaluation_horizon: Number of dataset days to include in evaluation
- queries per day target
- users

These are orthogonal to memory type, retrieval strategy, and decay policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadProfile:
    """Immutable workload profile for a benchmark run."""

    name: str
    evaluation_horizon: int
    seed: int = 42

    # Descriptive label shown in reports
    label: str = ""

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", self.name)


# Built-in profiles
LOW_QPD = WorkloadProfile(
    name="low_qpd",
    evaluation_horizon=14,
    label="Low (14 days)",
)

MEDIUM_QPD = WorkloadProfile(
    name="medium_qpd",
    evaluation_horizon=50,
    label="Medium (50 days)",
)

HIGH_QPD = WorkloadProfile(
    name="high_qpd",
    evaluation_horizon=90,
    label="High (90 days)",
)

PRODUCTION = WorkloadProfile(
    name="production",
    evaluation_horizon=90,
    label="Production (90 days)",
)

_REGISTRY: dict[str, WorkloadProfile] = {
    p.name: p for p in [LOW_QPD, MEDIUM_QPD, HIGH_QPD, PRODUCTION]
}


def get_profile(name: str) -> WorkloadProfile:
    """Resolve a workload profile by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown workload profile '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_profiles() -> list[WorkloadProfile]:
    """Return all registered workload profiles."""
    return list(_REGISTRY.values())
