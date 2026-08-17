"""Matrix expander — Cartesian product of benchmark dimensions.

Expands the 3D grid:
  Memory Type × Retrieval Strategy × Decay Policy (with λ steps)

Each cell becomes one MatrixCell — a self-contained, reproducible run spec.

Grid dimensions:
  - Memory types:         [episodic, semantic, preference, entity]
  - Retrieval strategies: [bm25, embeddings, hybrid, pgvector, llm_rerank]
  - Decay policies:       [none, exponential, logarithmic, linear, periodic]
  - Lambda steps:         [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] (6 steps, Δ=0.05)

A 4×5×5×6 full grid = 600 combinations (none/periodic policies get 1 cell each).
Filters reduce this to practical subsets.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

# ─── Axis definitions ──────────────────────────────────────────────────────────

MEMORY_TYPES = ["episodic", "semantic", "preference", "entity"]

RETRIEVAL_STRATEGIES = [
    "bm25",
    "embeddings",
    "api_embeddings",
    "hybrid",
    "pgvector",
    "llm_rerank",
]

DECAY_POLICIES = ["none", "exponential", "logarithmic", "linear", "periodic", "tiered"]

# Lambda steps — logarithmic scale covering the full half-life range.
#
# Half-lives (exponential): ln(2) / λ
#   λ=0.0005 → 1386 days   (near-permanent memory)
#   λ=0.001  →  693 days
#   λ=0.002  →  347 days
#   λ=0.005  →  139 days   ← typical episodic memory horizon
#   λ=0.01   →   69 days
#   λ=0.02   →   35 days
#   λ=0.05   →   14 days
#   λ=0.10   →    7 days   (aggressive recency bias)
#
# Use this broad 8-point sweep for Phase 4 Stage 1 (landscape discovery).
# The corpus spans ~720 days with gold-memory ages up to 294 days; the
# interesting region is usually λ ∈ [0.002, 0.05].
LAMBDA_STEPS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]

# Fine zoom steps — generated dynamically around the best broad-sweep region.
# Exposed here so callers can reference the concept without importing study_matrix.
# The actual fine grid is computed by fine_lambda_steps() in study_matrix.py.
LAMBDA_STEPS_BROAD = LAMBDA_STEPS  # alias for clarity in study_runner.py

# Pruning threshold steps — scales with lambda (higher decay → higher threshold)
# At gentler lambdas, even old memories retain significant weight, so the
# pruning threshold can be lower without retaining stale noise.
#
# DESIGN NOTE (June 1):
# This mapping is used for lambda sweeps (varying λ alone), not for decay policy
# comparisons. When comparing decay policies (none vs exponential vs logarithmic),
# MatrixExpander.expand_full() uses a CONSTANT_PRUNING_THRESHOLD (0.15) across all
# policies to ensure fair comparison. This prevents confounding the decay effect
# with different pruning aggressiveness.
#
PRUNING_THRESHOLD_STEPS = {
    0.001: 0.05,
    0.005: 0.10,
    0.01: 0.15,
    0.02: 0.20,
    0.05: 0.25,
    0.10: 0.30,
}

# Archival floor sweep — test how strongly the floor affects old-memory retrieval.
# 0.0 = no floor (pure decay to zero), 0.65 = current default, 1.0 = no decay ever.
ARCHIVAL_FLOORS: list[float | None] = [0.0, 0.25, 0.50, 0.65, 0.75, 0.90]

# Periodic refresh interval (days) — used for periodic decay
PERIODIC_REFRESH_DAYS = 7


@dataclass(frozen=True)
class DecaySpec:
    """Decay policy specification for one matrix cell."""

    policy: str  # none | exponential | logarithmic | linear | tiered | periodic
    lambda_value: float = 0.0  # 0.0 for none / periodic
    pruning_threshold: float = 0.30
    # How strongly decay influences ranking (0=post-hoc only, 1=full recency bias).
    # Default 0.0 keeps backward compatibility for BM25/embedding phases where
    # decay is only applied as a post-ranking score multiplier.
    # Phase 4 (decay sweep) sets this to 0.5 so different λ values produce
    # measurably different recall — without this, all λ give identical top-K.
    ranking_alpha: float = 0.0
    # Archival floor: minimum decay factor for memories older than archival_day_threshold.
    # None = no floor (pure decay to zero). 0.65 = default.
    # Phase 4b sweeps this across [0.0, 0.25, 0.50, 0.65, 0.75, 0.90].
    archival_floor: float | None = 0.65
    # Age boundary (days) at which the archival floor kicks in (all policies except tiered).
    archival_day_threshold: int = 90
    # Tiered policy: working memory window (no decay). Only used when policy=tiered.
    tiered_working_days: int = 7

    @property
    def label(self) -> str:
        if self.policy == "none":
            return "none"
        if self.policy == "periodic":
            return f"periodic(d={PERIODIC_REFRESH_DAYS})"
        base = f"{self.policy}(λ={self.lambda_value:.2f})"
        if self.archival_floor != 0.65:
            floor_str = f"{self.archival_floor:.2f}" if self.archival_floor is not None else "none"
            base += f"[floor={floor_str}]"
        return base

    def to_config_dict(self) -> dict:
        """Emit policy block for BenchmarkConfig construction."""
        _decay_extras = {
            "archival_floor": self.archival_floor,
            "archival_day_threshold": self.archival_day_threshold,
            "tiered_working_days": self.tiered_working_days,
        }
        if self.policy == "none":
            return {
                "decay": {
                    "type": "exponential", "lambda": 0.0,
                    "ranking_alpha": self.ranking_alpha,
                    **_decay_extras,
                },
                "pruning": {"strategy": "score_threshold", "threshold": self.pruning_threshold},
            }
        if self.policy == "periodic":
            return {
                "decay": {
                    "type": "exponential", "lambda": self.lambda_value,
                    "ranking_alpha": self.ranking_alpha,
                    **_decay_extras,
                },
                "pruning": {"strategy": "age_based", "threshold": self.pruning_threshold},
            }
        # Exponential, logarithmic, linear, tiered map to their own distinct formula types
        decay_type_map = {
            "exponential": "exponential",
            "logarithmic": "logarithmic",  # 1/(1 + λt) — distinct from exponential
            "linear": "linear",
            "tiered": "tiered",  # class-based: 0-7d full, 7-90d decay, 90+ full
        }
        return {
            "decay": {
                "type": decay_type_map.get(self.policy, "exponential"),
                "lambda": self.lambda_value,
                "ranking_alpha": self.ranking_alpha,
                **_decay_extras,
            },
            "pruning": {
                "strategy": "score_threshold",
                "threshold": self.pruning_threshold,
            },
        }


@dataclass(frozen=True)
class MatrixCell:
    """A single cell in the benchmark matrix — one complete run specification."""

    memory_type: str  # episodic | semantic | preference | entity
    retrieval_strategy: str  # bm25 | embeddings | api_embeddings | hybrid | pgvector | llm_rerank
    decay: DecaySpec
    workload_profile: str  # low_qpd | medium_qpd | high_qpd
    seed: int = 42

    @property
    def cell_id(self) -> str:
        """Deterministic unique ID for this cell."""
        key = f"{self.memory_type}:{self.retrieval_strategy}:{self.decay.label}:{self.workload_profile}:{self.seed}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        return f"{self.memory_type} × {self.retrieval_strategy} × {self.decay.label}"

    def to_config_dict(self, evaluation_horizon: int) -> dict:
        """Build a BenchmarkConfig-compatible dict for this cell."""
        policy_block = self.decay.to_config_dict()
        memory_module = f"{self.memory_type}_store"

        return {
            "memory": {
                "enabled": {
                    "short_term": [],
                    "long_term": [memory_module],
                }
            },
            "policies": {
                "module_policies": {
                    memory_module: policy_block,
                }
            },
            "benchmark": {
                "evaluation_horizon": evaluation_horizon,
                "seed": self.seed,
                "scenarios": ["delayed_recall"],
                "retrieval_strategy": self.retrieval_strategy,
            },
            "observability": {
                "exporter": "none",
                "endpoint": "http://localhost:4317",
                "log_level": "WARNING",
            },
            "answering": {"enabled": False, "model": ""},
        }

    def to_summary_dict(self) -> dict:
        """Serializable summary for result reporting."""
        return {
            "cell_id": self.cell_id,
            "memory_type": self.memory_type,
            "retrieval_strategy": self.retrieval_strategy,
            "decay_policy": self.decay.policy,
            "lambda": self.decay.lambda_value,
            "pruning_threshold": self.decay.pruning_threshold,
            "workload_profile": self.workload_profile,
            "seed": self.seed,
            "label": self.label,
        }


class MatrixExpander:
    """Expands benchmark dimensions into a flat list of MatrixCells.

    Supports:
    - Full grid (all combinations)
    - Filtered grid (subset of strategies/types)
    - Quick grid (3×3 core)
    """

    def expand_full(
        self,
        workload_profile: str = "medium_qpd",
        seed: int = 42,
        strategies: list[str] | None = None,
        memory_types: list[str] | None = None,
        decay_policies: list[str] | None = None,
        lambda_steps: list[float] | None = None,
    ) -> list[MatrixCell]:
        """Expand full Cartesian grid.

        Args:
            workload_profile: Workload profile name to use.
            seed: Random seed.
            strategies: Subset of retrieval strategies (None = all).
            memory_types: Subset of memory types (None = all).
            decay_policies: Subset of decay policies (None = all).
            lambda_steps: Lambda values to sweep (None = all 7).

        Returns:
            List of MatrixCells, one per grid combination.

        DESIGN NOTE: Pruning threshold is held CONSTANT at 0.15 across all decay
        policies (none, exponential, logarithmic) to avoid confounding decay
        policy comparisons. This ensures differences in recall/temporal_accuracy
        are due to decay policy, not different pruning aggressiveness.
        """
        strats = strategies or RETRIEVAL_STRATEGIES
        mem_types = memory_types or MEMORY_TYPES
        policies = decay_policies or DECAY_POLICIES
        lambdas = lambda_steps or LAMBDA_STEPS

        cells: list[MatrixCell] = []

        # Use constant pruning threshold for all decay policies to ensure fair comparison
        # This value (0.15) is in the middle of the range and represents moderate pruning
        # This threshold means that pruning of data happens when the memory score falls below 15% of its original value, which allows for a fair comparison of decay policies without confounding effects from varying pruning aggressiveness.
        CONSTANT_PRUNING_THRESHOLD = 0.15

        for mem_type, strategy, policy in itertools.product(mem_types, strats, policies):
            if policy == "none":
                decay_specs = [
                    DecaySpec(
                        policy="none",
                        lambda_value=0.0,
                        pruning_threshold=CONSTANT_PRUNING_THRESHOLD,
                    )
                ]
            elif policy == "periodic":
                decay_specs = [
                    DecaySpec(
                        policy="periodic",
                        lambda_value=0.0,
                        pruning_threshold=CONSTANT_PRUNING_THRESHOLD,
                    )
                ]
            else:
                decay_specs = [
                    DecaySpec(
                        policy=policy,
                        lambda_value=lam,
                        pruning_threshold=CONSTANT_PRUNING_THRESHOLD,
                    )
                    for lam in lambdas
                ]

            for decay in decay_specs:
                cells.append(
                    MatrixCell(
                        memory_type=mem_type,
                        retrieval_strategy=strategy,
                        decay=decay,
                        workload_profile=workload_profile,
                        seed=seed,
                    )
                )

        return cells

    def expand_core_3x3(
        self,
        workload_profile: str = "medium_qpd",
        seed: int = 42,
    ) -> list[MatrixCell]:
        """Expand the core 3×3 grid for quick comparison.

        Uses the 3 most representative values per axis:
        - Memory types: episodic, semantic, preference
        - Strategies:   bm25, embeddings, hybrid
        - Decay:        none, exponential(λ=0.05), logarithmic(λ=0.05)

        Returns 3×3×3 = 27 cells.
        """
        return self.expand_full(
            workload_profile=workload_profile,
            seed=seed,
            memory_types=["episodic", "semantic", "preference"],
            strategies=["bm25", "embeddings", "hybrid"],
            decay_policies=["none", "exponential", "logarithmic"],
            lambda_steps=[0.01],  # gentle decay — 69-day half-life preserves gold signal
        )

    def expand_lambda_sweep(
        self,
        memory_type: str = "episodic",
        strategy: str = "bm25",
        decay_policy: str = "exponential",
        workload_profile: str = "medium_qpd",
        seed: int = 42,
    ) -> list[MatrixCell]:
        """Expand all lambda steps for a fixed memory+strategy+policy.

        Returns 7 cells (one per lambda step).
        """
        return self.expand_full(
            workload_profile=workload_profile,
            seed=seed,
            memory_types=[memory_type],
            strategies=[strategy],
            decay_policies=[decay_policy],
            lambda_steps=LAMBDA_STEPS,
        )

    def describe(self, cells: list[MatrixCell]) -> dict:
        """Return a summary of the expanded matrix."""
        return {
            "total_cells": len(cells),
            "memory_types": sorted({c.memory_type for c in cells}),
            "retrieval_strategies": sorted({c.retrieval_strategy for c in cells}),
            "decay_policies": sorted({c.decay.policy for c in cells}),
            "lambda_values": sorted({c.decay.lambda_value for c in cells}),
            "workload_profiles": sorted({c.workload_profile for c in cells}),
        }
