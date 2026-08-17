"""Comprehensive study matrix — 5 clean benchmark phases.

Canonical configuration dimensions:
  strategy          : bm25 | semantic | hybrid | recency
  embedding_model   : which sentence-transformer / Ollama model to use
  embedding_backend : sentence-transformers | ollama
  bm25_weight       : 0.0–1.0 (hybrid only); semantic_weight = 1 - bm25_weight
  reranker          : none | cross-encoder/ms-marco-MiniLM-L6-v2 | BAAI/bge-reranker-base
  decay             : none | exponential | logarithmic | linear | tiered

Phases:
  Phase 1 — BM25 + Recency baselines  (no embedding, no decay sweep)
  Phase 2 — Semantic embedding-model comparison  (all models × backends)
  Phase 3 — Hybrid weight sweep  (best embed × bm25_weight 0.0…1.0 step 0.1)
  Phase 4 — Temporal/decay sweep  (winning strategy × all policies × all λ)
  Phase 5 — Reranker comparison  (retrieval → top-N candidates → reranker)

Design rules:
  - `ollama` is an embedding BACKEND, not a strategy. Strategy is always one of
    {bm25, semantic, hybrid, recency}. The backend field determines whether
    sentence-transformers or Ollama serves the vectors.
  - Reranker is a second-stage component: retrieval fetches top-N, reranker
    re-scores those, returns top-K. It is NOT a strategy.
  - to_config_dict() translates canonical names → internal strategy names
    required by the resolver/factory (e.g. "semantic"→"embeddings" for
    sentence-transformers, "semantic"→"api_embeddings" for API endpoint).
  - Every CSV row fully describes the configuration; no dimension is implicit.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import math

from benchmark.workload.matrix import DecaySpec, LAMBDA_STEPS, ARCHIVAL_FLOORS

# ─── Algorithmic sweep constants (not model lists — those live in configs/study_defaults.yaml) ──

# Full BM25-weight sweep for hybrid (0=pure semantic, 1=pure BM25, step 0.1)
HYBRID_BM25_WEIGHTS = [round(w * 0.1, 1) for w in range(11)]  # 0.0…1.0

# ─── Structural defaults for phase seeding (overridable via YAML) ────────────
# These are used as fallback seeds when a phase hasn't run yet — not as model lists.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_RERANKER = "none"
DEFAULT_DECAY = DecaySpec(policy="exponential", lambda_value=0.01, pruning_threshold=0.15)

# Kept for backward-compat imports — both are empty; model lists come from YAML.
EMBEDDING_MODELS_LOCAL: list[str] = []
EMBEDDING_MODELS_OLLAMA: list[str] = []
EMBEDDING_MODELS_ST: list[str] = []
RERANKER_MODELS: list[str] = []
JUDGE_MODELS: list[str] = []


# ─── Parameter-zoom helpers ─────────────────────────────────────────────────


def fine_lambda_steps(best_lambda: float, n_steps: int = 7) -> list[float]:
    """Return a fine-resolution λ grid centred on `best_lambda`.

    Uses a log-scale neighbourhood so that steps are proportional to λ
    (appropriate because λ has a multiplicative effect on half-life).

    Example: best_lambda=0.01 →
        [0.005, 0.0063, 0.008, 0.01, 0.0126, 0.016, 0.02]

    The result is sorted ascending and rounded to 4 significant figures.
    n_steps must be odd (centre point is best_lambda).
    """
    if best_lambda <= 0:
        return [best_lambda]
    n_steps = max(3, n_steps | 1)  # ensure odd
    half = n_steps // 2
    log_centre = math.log10(best_lambda)
    # Each step is one quarter of an order of magnitude
    step_size = 0.25 / half
    steps = []
    for i in range(-half, half + 1):
        raw = 10 ** (log_centre + i * step_size)
        # Round to 4 significant figures
        if raw > 0:
            magnitude = 10 ** math.floor(math.log10(raw))
            rounded = round(raw / magnitude, 3) * magnitude
            steps.append(round(rounded, 6))
    return sorted(set(steps))


def fine_hybrid_weights(best_weight: float, step: float = 0.05) -> list[float]:
    """Return a fine-resolution BM25-weight grid around `best_weight`.

    Generates ±0.20 around the best coarse weight at `step` resolution,
    clamped to [0.0, 1.0] and deduplicated.

    Example: best_weight=0.3, step=0.05 →
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    """
    radius = 0.20
    lo = max(0.0, best_weight - radius)
    hi = min(1.0, best_weight + radius)
    n = round((hi - lo) / step)
    weights = [round(lo + i * step, 2) for i in range(n + 1)]
    return sorted(set(weights))


@dataclass(frozen=True)
class StudyCell:
    """One run specification — fully self-contained, no implicit dimensions.

    Fields
    ------
    retrieval_strategy  : canonical name — bm25 | semantic | hybrid | recency
    embedding_model     : model identifier (empty for bm25/recency)
    embedding_backend   : sentence-transformers | ollama
    bm25_weight         : 0.0–1.0, meaningful only when strategy=hybrid
    reranker_model      : "none" or a cross-encoder model name
    """

    memory_type: str
    retrieval_strategy: str          # bm25 | semantic | hybrid | recency
    decay: DecaySpec
    workload_profile: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND  # sentence-transformers | ollama
    bm25_weight: float = DEFAULT_BM25_WEIGHT
    reranker_model: str = DEFAULT_RERANKER
    ollama_base_url: str = ""
    seed: int = 42
    study_phase: str = "general"

    @property
    def semantic_weight(self) -> float:
        return round(1.0 - self.bm25_weight, 2)

    @property
    def cell_id(self) -> str:
        key = (
            f"{self.memory_type}:{self.retrieval_strategy}:{self.decay.label}"
            f":{self.embedding_model}:{self.embedding_backend}:{self.bm25_weight:.2f}"
            f":{self.reranker_model}:{self.workload_profile}:{self.seed}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        parts = [self.memory_type, self.retrieval_strategy]
        if self.retrieval_strategy in ("semantic", "hybrid"):
            short_model = self.embedding_model.split("/")[-1]
            parts.append(f"{short_model}({self.embedding_backend[:2]})")
        if self.retrieval_strategy == "hybrid":
            parts.append(f"bm25={self.bm25_weight:.1f}/sem={self.semantic_weight:.1f}")
        parts.append(self.decay.label)
        if self.reranker_model and self.reranker_model != "none":
            parts.append(f"rerank={(self.reranker_model or 'none').split('/')[-1]}")
        return " × ".join(parts)

    def _internal_retrieval_strategy(self) -> str:
        """Translate canonical strategy name → internal resolver name.

        semantic + backend=api       → api_embeddings
        semantic + any other backend → embeddings (sentence-transformers)
        hybrid                       → hybrid
        everything else              → pass through (bm25, recency, llm_rerank)
        """
        if self.retrieval_strategy == "semantic":
            if self.embedding_backend == "api":
                return "api_embeddings"
            return "embeddings"
        if self.retrieval_strategy == "hybrid":
            return "hybrid"
        return self.retrieval_strategy  # bm25, recency, llm_rerank pass through

    def to_config_dict(self, evaluation_horizon: int) -> dict:
        """Build BenchmarkConfig-compatible dict for this cell."""
        policy_block = self.decay.to_config_dict()
        memory_module = f"{self.memory_type}_store"
        internal_strategy = self._internal_retrieval_strategy()

        retrieval: dict = {}

        if internal_strategy == "embeddings":
            retrieval["embeddings"] = {"model_name": self.embedding_model}

        elif internal_strategy == "api_embeddings":
            retrieval["api_embeddings"] = {"model_name": self.embedding_model}

        elif internal_strategy == "hybrid":
            retrieval["embeddings"] = {"model_name": self.embedding_model}
            retrieval["hybrid"] = {
                "strategies": ["bm25", "embeddings"],
                "bm25_weight": self.bm25_weight,
                "confidence_threshold": 0.5,
            }

        elif internal_strategy == "llm_rerank":
            # Phase 5 reranker: BM25 fetches candidates; cross-encoder re-scores
            retrieval["embeddings"] = {"model_name": self.embedding_model}

        # Reranker block — always emitted; only consumed when strategy=llm_rerank
        if self.reranker_model == "none":
            reranker = {"strategy": "local_overlap"}
        else:
            reranker = {
                "strategy": "adaptive_api",
                "model_name": self.reranker_model,
            }

        return {
            "memory": {
                "enabled": {"short_term": [], "long_term": [memory_module]},
            },
            "policies": {
                "module_policies": {memory_module: policy_block},
            },
            "benchmark": {
                "evaluation_horizon": evaluation_horizon,
                "seed": self.seed,
                "scenarios": ["delayed_recall"],
                "retrieval_strategy": internal_strategy,
                "reranker": reranker,
                "retrieval": retrieval,
            },
            "observability": {
                "exporter": "none",
                "endpoint": "http://localhost:4317",
                "log_level": "WARNING",
            },
            "answering": {"enabled": False, "model": ""},
        }

    def to_summary_dict(self) -> dict:
        import os as _os
        top_k = int(_os.environ.get("BENCHMARK_RECALL_K", "10"))
        return {
            "cell_id": self.cell_id,
            "memory_type": self.memory_type,
            # canonical strategy — always bm25|semantic|hybrid|recency
            "retrieval_strategy": self.retrieval_strategy,
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "reranker_model": self.reranker_model,
            "decay_policy": self.decay.policy,
            "lambda": self.decay.lambda_value,
            "pruning_threshold": self.decay.pruning_threshold,
            "ranking_alpha": self.decay.ranking_alpha,
            "archival_floor": self.decay.archival_floor,
            "archival_day_threshold": self.decay.archival_day_threshold,
            "tiered_working_days": self.decay.tiered_working_days,
            "top_k": top_k,
            "ollama_base_url": self.ollama_base_url,
            "workload_profile": self.workload_profile,
            "seed": self.seed,
            "study_phase": self.study_phase,
            "label": self.label,
        }


class StudyExpander:
    """Expands the study into 5 focused phases.

    Phase 1 — BM25 + Recency baselines  (no embedding model, no decay sweep)
    Phase 2 — Semantic embedding-model comparison  (all models × all backends)
    Phase 3 — Hybrid weight sweep  (best embed × bm25_weight 0.0–1.0 step 0.1)
    Phase 4 — Temporal/decay sweep  (winning strategy × all policies × all λ)
    Phase 5 — Reranker comparison  (retrieval → candidates → reranker)
    """

    def __init__(
        self,
        memory_types: list[str] | None = None,
        workload_profile: str = "medium_qpd",
        seed: int = 42,
        ollama_base_url: str = "",
    ):
        self._mem_types = memory_types or ["episodic", "semantic", "preference"]
        self._profile = workload_profile
        self._seed = seed
        self._ollama_url = ollama_base_url

    def _cell(self, **kwargs) -> StudyCell:
        return StudyCell(
            workload_profile=self._profile,
            seed=self._seed,
            ollama_base_url=self._ollama_url,
            **kwargs,
        )

    # ── Phase 1: BM25 + Recency baselines ────────────────────────────────────

    def phase_bm25_baseline(self) -> list[StudyCell]:
        """BM25, BM25L, and Recency baselines — one cell per memory type each.

        BM25L uses lower-bounded TF normalisation (Lv & Zhai 2011) and
        typically outperforms BM25 Okapi on long-form conversational memory.
        All three are λ-agnostic so Phase 1 runs each exactly once.
        """
        no_decay = DecaySpec(policy="none", lambda_value=0.0, pruning_threshold=0.15)
        cells = []
        for mem_type in self._mem_types:
            for strat, bw in [("bm25", 1.0), ("bm25l", 1.0), ("recency", 0.0)]:
                cells.append(self._cell(
                    memory_type=mem_type,
                    retrieval_strategy=strat,
                    decay=no_decay,
                    embedding_model="none",
                    embedding_backend="none",
                    bm25_weight=bw,
                    reranker_model="none",
                    study_phase="phase1_baselines",
                ))
        return cells

    # ── Phase 2: Semantic embedding-model comparison ─────────────────────────

    def phase_embedding_model_comparison(
        self,
        local_models: list[str] | None = None,
        ollama_models: list[str] | None = None,
        api_models: list[str] | None = None,
    ) -> list[StudyCell]:
        """Compare embedding models using strategy=semantic.

        Backends:
          local_models  → sentence-transformers (in-process, no HTTP)
          api_models    → api_embeddings (OpenAI-compatible endpoint via BENCHMARK_OPENAI_BASE_URL)
          ollama_models → kept for backward compat; prefer api_models with base_url override

        All map to strategy="semantic" in canonical terms.
        Held constant: decay=exponential(λ=0.01), reranker=none.
        """
        cells = []
        decay = DEFAULT_DECAY

        st_models = local_models or []
        ol_models = ollama_models or []
        ap_models = api_models or []

        # model-outer so all memory_types for a given model run together
        # (keeps the model in VRAM/memory across the memory-type loop)
        for model, mem_type in itertools.product(st_models, self._mem_types):
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="semantic",
                decay=decay,
                embedding_model=model,
                embedding_backend="sentence-transformers",
                bm25_weight=0.0,
                reranker_model="none",
                study_phase="phase2_embedding_comparison",
            ))

        for model, mem_type in itertools.product(ap_models, self._mem_types):
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="semantic",
                decay=decay,
                embedding_model=model,
                embedding_backend="api",
                bm25_weight=0.0,
                reranker_model="none",
                study_phase="phase2_embedding_comparison",
            ))

        for model, mem_type in itertools.product(ol_models, self._mem_types):
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="semantic",
                decay=decay,
                embedding_model=model,
                embedding_backend="ollama",
                bm25_weight=0.0,
                reranker_model="none",
                study_phase="phase2_embedding_comparison",
            ))

        # ColBERT — token-level MaxSim, no embedding model required
        for mem_type in self._mem_types:
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="colbert",
                decay=decay,
                embedding_model="all-MiniLM-L6-v2",
                embedding_backend="sentence-transformers",
                bm25_weight=0.0,
                reranker_model="none",
                study_phase="phase2_embedding_comparison",
            ))

        # Adaptive — per-query routing oracle (BM25 / embeddings / hybrid by query type)
        for mem_type in self._mem_types:
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="adaptive",
                decay=decay,
                embedding_model="all-MiniLM-L6-v2",
                embedding_backend="sentence-transformers",
                bm25_weight=0.0,
                reranker_model="none",
                study_phase="phase2_embedding_comparison",
            ))

        return cells

    # ── Phase 3: Hybrid weight sweep ─────────────────────────────────────────

    def phase_hybrid_weight_sweep(
        self,
        best_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        bm25_weights: list[float] | None = None,
        stage: str = "broad",
        fine_around: float | None = None,
    ) -> list[StudyCell]:
        """Sweep bm25_weight in two stages: coarse discovery then fine zoom.

        stage="broad"  — 0.1-step sweep, 0.0…1.0 (11 points). Reveals the
                         general landscape and the region worth zooming into.
        stage="fine"   — 0.05-step sweep ±0.20 around fine_around (≤9 points).
                         Pinpoints the optimum. Requires fine_around != None.
        stage="custom" — use the explicit bm25_weights list.

        bm25_weight=0.0 → pure semantic; bm25_weight=1.0 → pure BM25.

        Held constant: decay=exponential(λ=0.01), reranker=none.
        """
        if stage == "broad":
            weights = HYBRID_BM25_WEIGHTS
            phase_tag = "phase3_hybrid_broad"
        elif stage == "fine":
            if fine_around is None:
                raise ValueError("fine_around must be set when stage='fine'")
            weights = fine_hybrid_weights(fine_around)
            phase_tag = "phase3_hybrid_fine"
        else:
            weights = bm25_weights if bm25_weights is not None else HYBRID_BM25_WEIGHTS
            phase_tag = "phase3_hybrid_weight"

        decay = DEFAULT_DECAY
        cells = []

        for w, mem_type in itertools.product(weights, self._mem_types):
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy="hybrid",
                decay=decay,
                embedding_model=best_embedding_model,
                embedding_backend=best_embedding_backend,
                bm25_weight=w,
                reranker_model="none",
                study_phase=phase_tag,
            ))
        return cells

    # ── Phase 4: Temporal/decay sweep ────────────────────────────────────────

    def phase_decay_lambda_sweep(
        self,
        best_strategy: str = "bm25",
        best_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        decay_policies: list[str] | None = None,
        lambda_steps: list[float] | None = None,
        stage: str = "broad",
        fine_around: float | None = None,
        include_no_decay_baseline: bool = True,
        bm25_weight: float | None = None,
    ) -> list[StudyCell]:
        """Sweep decay policy × λ in two stages: coarse discovery then fine zoom.

        stage="broad"  — 8-point log-scale sweep (LAMBDA_STEPS: 0.0005…0.10).
                         Reveals the full response curve and identifies the
                         interesting region. Always includes the no-decay baseline.
        stage="fine"   — 7-point log-scale zoom ±0.25 orders of magnitude around
                         fine_around. Requires fine_around != None.
                         Used after the broad sweep identifies the optimum region.
        stage="custom" — use lambda_steps exactly as given.

        ranking_alpha=0.5 for all λ > 0 ensures decay affects top-K selection,
        not just post-hoc score scaling.
        """
        policies = decay_policies or ["exponential", "logarithmic", "linear", "tiered"]

        if stage == "broad":
            lambdas = LAMBDA_STEPS
            phase_tag = "phase4_decay_broad"
        elif stage == "fine":
            if fine_around is None:
                raise ValueError("fine_around must be set when stage='fine'")
            lambdas = fine_lambda_steps(fine_around)
            phase_tag = "phase4_decay_fine"
        else:
            lambdas = lambda_steps or LAMBDA_STEPS
            phase_tag = "phase4_decay_sweep"

        cells = []

        _bm25w = bm25_weight if bm25_weight is not None else DEFAULT_BM25_WEIGHT

        for policy, lam, mem_type in itertools.product(policies, lambdas, self._mem_types):
            decay = DecaySpec(
                policy=policy, lambda_value=lam,
                pruning_threshold=0.15, ranking_alpha=0.5,
            )
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy=best_strategy,
                decay=decay,
                embedding_model=best_embedding_model,
                embedding_backend=best_embedding_backend,
                bm25_weight=_bm25w,
                reranker_model="none",
                study_phase=phase_tag,
            ))

        if include_no_decay_baseline:
            no_decay = DecaySpec(policy="none", lambda_value=0.0, pruning_threshold=0.15, ranking_alpha=0.0)
            for mem_type in self._mem_types:
                cells.append(self._cell(
                    memory_type=mem_type,
                    retrieval_strategy=best_strategy,
                    decay=no_decay,
                    embedding_model=best_embedding_model,
                    embedding_backend=best_embedding_backend,
                    bm25_weight=_bm25w,
                    reranker_model="none",
                    study_phase=phase_tag,
                ))
        return cells

    # ── Phase 4b: Archival floor sweep ───────────────────────────────────────

    def phase_archival_floor_sweep(
        self,
        best_strategy: str = "bm25",
        best_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        best_policy: str = "exponential",
        best_lambda: float = 0.01,
        archival_floors: list | None = None,
        bm25_weight: float | None = None,
    ) -> list[StudyCell]:
        """Sweep archival_floor to measure its effect on old-memory retrieval.

        Uses the winning strategy + best decay policy + best λ from Phase 4,
        varying only the archival_floor parameter across:
            [0.0, 0.25, 0.50, 0.65, 0.75, 0.90]

        0.0   = no floor — pure decay (old memories can decay to zero)
        0.65  = current default — old memories retain at least 65% weight
        None  = alias for 0.0 (no floor)

        This phase answers: "Does the 0.65 default favour Phase 4 conclusions?"
        A conclusion that flips across archival_floor values is not robust.
        """
        floors = archival_floors if archival_floors is not None else ARCHIVAL_FLOORS
        _bm25w = bm25_weight if bm25_weight is not None else DEFAULT_BM25_WEIGHT
        cells = []

        for floor, mem_type in itertools.product(floors, self._mem_types):
            decay = DecaySpec(
                policy=best_policy,
                lambda_value=best_lambda,
                pruning_threshold=0.15,
                ranking_alpha=0.5,
                archival_floor=floor,
            )
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy=best_strategy,
                decay=decay,
                embedding_model=best_embedding_model,
                embedding_backend=best_embedding_backend,
                bm25_weight=_bm25w,
                reranker_model="none",
                study_phase="phase4b_archival_floor",
            ))
        return cells

    # ── Phase 5: Reranker comparison ─────────────────────────────────────────

    def phase_reranker_comparison(
        self,
        best_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        reranker_models: list[str] | None = None,
    ) -> list[StudyCell]:
        """Compare rerankers as a second-stage component on top of retrieval.

        Flow: retrieval (semantic or hybrid) → top-N candidates → reranker → top-K.

        reranker="none"   → strategy="semantic"   (pure retrieval baseline)
        reranker=<model>  → strategy internal="llm_rerank"
                            (BM25 fetches full corpus, cross-encoder re-scores)

        The reranker block is only consumed by the llm_rerank resolver path —
        using "semantic" with a non-none reranker would silently ignore it.

        Held constant: decay=exponential(λ=0.01), best embedding model.
        Variable: reranker_model.
        """
        rerankers = reranker_models or []
        decay = DEFAULT_DECAY
        cells = []

        for reranker, mem_type in itertools.product(rerankers, self._mem_types):
            # "none" → pure semantic baseline; any real model → llm_rerank path
            strat = "semantic" if reranker == "none" else "llm_rerank"
            # llm_rerank always uses sentence-transformers internally
            backend = best_embedding_backend if reranker == "none" else "sentence-transformers"
            cells.append(self._cell(
                memory_type=mem_type,
                retrieval_strategy=strat,
                decay=decay,
                embedding_model=best_embedding_model,
                embedding_backend=backend,
                bm25_weight=0.0,
                reranker_model=reranker,
                study_phase="phase5_reranker_comparison",
            ))
        return cells

    # ── Full study ────────────────────────────────────────────────────────────

    def expand_full_study(
        self,
        local_models: list[str] | None = None,
        ollama_models: list[str] | None = None,
        reranker_models: list[str] | None = None,
        bm25_weights: list[float] | None = None,
        best_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        best_strategy: str = "bm25",
    ) -> list[StudyCell]:
        seen: set[str] = set()
        cells: list[StudyCell] = []
        for phase_cells in [
            self.phase_bm25_baseline(),
            self.phase_embedding_model_comparison(local_models, ollama_models),
            self.phase_hybrid_weight_sweep(best_embedding_model, best_embedding_backend, bm25_weights),
            self.phase_decay_lambda_sweep(best_strategy, best_embedding_model, best_embedding_backend),
            self.phase_reranker_comparison(best_embedding_model, best_embedding_backend, reranker_models),
        ]:
            for c in phase_cells:
                if c.cell_id not in seen:
                    seen.add(c.cell_id)
                    cells.append(c)
        return cells

    def describe(self, cells: list[StudyCell]) -> dict:
        by_phase: dict[str, int] = {}
        for c in cells:
            by_phase[c.study_phase] = by_phase.get(c.study_phase, 0) + 1
        return {
            "total_cells": len(cells),
            "by_phase": by_phase,
            "memory_types": sorted({c.memory_type for c in cells}),
            "retrieval_strategies": sorted({c.retrieval_strategy for c in cells}),
            "embedding_models": sorted({c.embedding_model for c in cells}),
            "embedding_backends": sorted({c.embedding_backend for c in cells}),
            "bm25_weights": sorted({c.bm25_weight for c in cells}),
            "reranker_models": sorted({c.reranker_model for c in cells}),
            "decay_policies": sorted({c.decay.policy for c in cells}),
            "lambda_values": sorted({c.decay.lambda_value for c in cells}),
        }
