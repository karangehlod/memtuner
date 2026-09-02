"""Mathematical correctness tests for the Agentic Memory Benchmark.

Covers all numerical formulas used in scoring, ranking, and statistical
analysis:
  - Decay functions (exponential, linear, logarithmic, tiered)
  - Composite score weighting and recall gate
  - Recency strategy scoring
  - Bootstrap confidence interval computation
  - Reciprocal Rank Fusion (RRF) in HybridStrategy
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.strategies.hybrid_strategy import HybridStrategy
from benchmark.memory.strategies.recency_strategy import RecencyStrategy
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.workload.scheduler import MatrixRunResult
from benchmark.workload.study_aggregator import StudyAggregator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _mem(mem_id: str, content: str = "test content", user_id: str = "u1") -> MemoryEvent:
    """Create a minimal MemoryEvent for testing."""
    return MemoryEvent(
        id=mem_id,
        user_id=user_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=_EPOCH,
        importance=0.8,
        task_id="t1",
    )


def _result(**kwargs: Any) -> MatrixRunResult:
    """Build a MatrixRunResult with sensible defaults, overriding via kwargs."""
    defaults = dict(
        cell_id="c1",
        run_id="r1",
        memory_type="episodic",
        retrieval_strategy="bm25",
        decay_policy="exponential",
        lambda_value=0.05,
        pruning_threshold=0.35,
        workload_profile="default",
        seed=42,
        recall_at_k=0.5,
        precision_at_k=0.5,
        mrr=0.5,
        temporal_accuracy=0.5,
        success=True,
    )
    defaults.update(kwargs)
    return MatrixRunResult(**defaults)


def _study_result(recall: float, strategy: str = "bm25") -> Any:
    """Build a StudyRunResult-compatible object via MatrixRunResult with a strategy tag."""
    r = _result(recall_at_k=recall, retrieval_strategy=strategy)
    return r


# ---------------------------------------------------------------------------
# 1. TestDecayFormulas
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDecayFormulas:
    """Tests for BaseLongTermStore._compute_decay_factor()."""

    @staticmethod
    def _store(decay_type: str = "exponential", lam: float = 0.05, floor: float | None = 0.65) -> EpisodicStore:
        """Create an EpisodicStore wired with specific decay params."""
        store = EpisodicStore(decay_type=decay_type, decay_lambda=lam)
        store._archival_floor = floor
        return store

    # --- boundary: t = 0 ---

    def test_exponential_at_t0_is_1(self) -> None:
        """exp decay at t=0 must equal exactly 1.0."""
        store = self._store("exponential")
        assert store._compute_decay_factor(0) == 1.0

    def test_linear_at_t0_is_1(self) -> None:
        """linear decay at t=0 must equal exactly 1.0."""
        store = self._store("linear")
        assert store._compute_decay_factor(0) == 1.0

    def test_logarithmic_at_t0_is_1(self) -> None:
        """logarithmic decay at t=0 must equal exactly 1.0."""
        store = self._store("logarithmic")
        assert store._compute_decay_factor(0) == 1.0

    def test_tiered_is_1_at_t0(self) -> None:
        """tiered decay at t=0 (working memory zone) must equal 1.0."""
        store = self._store("tiered")
        assert store._compute_decay_factor(0) == 1.0

    # --- half-life ---

    def test_exponential_half_life(self) -> None:
        """At t = ln(2) / λ, raw exponential value ≈ 0.5 (before floor)."""
        lam = 0.05
        # Set floor=None so we get raw exponential without archival floor clamping.
        store = self._store("exponential", lam=lam, floor=None)
        t_half = math.log(2) / lam  # ≈ 13.86 days
        result = store._compute_decay_factor(int(t_half))
        # At ~13 days, exp(-0.05*13) ≈ 0.52 — still close to 0.5
        assert abs(result - 0.5) < 0.03

    # --- archival floor ---

    def test_archival_floor_applied_at_90_days(self) -> None:
        """For exp policy at t=90, raw≈0.011 but floor=0.65 must be returned."""
        lam = 0.05
        store = self._store("exponential", lam=lam, floor=0.65)
        result = store._compute_decay_factor(90)
        raw = math.exp(-lam * 90)  # ≈ 0.011
        assert raw < 0.65, "sanity: raw should be below floor"
        assert result == pytest.approx(0.65, abs=1e-9)

    def test_archival_floor_not_applied_before_90(self) -> None:
        """At t=89, the result is raw exponential (below floor threshold)."""
        lam = 0.05
        store = self._store("exponential", lam=lam, floor=0.65)
        result = store._compute_decay_factor(89)
        raw = math.exp(-lam * 89)
        assert result == pytest.approx(raw, rel=1e-9)

    # --- tiered special cases ---

    def test_tiered_is_1_at_t_ge_90(self) -> None:
        """tiered at t=90 enters archival zone and restores to 1.0."""
        store = self._store("tiered")
        assert store._compute_decay_factor(90) == 1.0

    def test_tiered_decays_between_7_and_90(self) -> None:
        """tiered at t=50, λ=0.01 equals exp(-0.01*(50-7))."""
        lam = 0.01
        t = 50
        store = self._store("tiered", lam=lam)
        expected = math.exp(-lam * (t - 7))
        assert store._compute_decay_factor(t) == pytest.approx(expected, rel=1e-9)

    # --- λ=0 identity ---

    def test_lambda_zero_always_returns_1(self) -> None:
        """Any policy with λ=0 must return 1.0 for all t."""
        for decay_type in ("exponential", "linear", "logarithmic", "tiered"):
            store = self._store(decay_type, lam=0.0)
            for t in (0, 10, 50, 200):
                assert store._compute_decay_factor(t) == 1.0, (
                    f"Expected 1.0 for {decay_type} λ=0 at t={t}"
                )

    # --- monotonicity ---

    def test_decay_is_monotonically_decreasing_exp(self) -> None:
        """exp values at t=[0,10,20,30] are strictly decreasing (no floor active)."""
        lam = 0.05
        store = self._store("exponential", lam=lam, floor=None)
        values = [store._compute_decay_factor(t) for t in (0, 10, 20, 30)]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], (
                f"Expected strict decrease: values[{i}]={values[i]} vs values[{i+1}]={values[i+1]}"
            )

    # --- caching ---

    def test_decay_cache_returns_same_value(self) -> None:
        """Calling _compute_decay_factor twice with the same t returns identical float."""
        store = self._store("exponential")
        v1 = store._compute_decay_factor(15)
        v2 = store._compute_decay_factor(15)
        assert v1 is v2 or v1 == v2  # cache hit: same object or same value

    # --- negative t safety ---

    def test_negative_t_clamped_to_zero(self) -> None:
        """Negative days_elapsed must not crash and must be treated as t=0."""
        store = self._store("exponential")
        result = store._compute_decay_factor(-5)
        assert result == store._compute_decay_factor(0)


# ---------------------------------------------------------------------------
# 2. TestCompositeScore
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompositeScore:
    """Tests for MatrixRunResult.composite_score()."""

    def test_formula_exact(self) -> None:
        """Manual calculation: 0.40*0.6 + 0.25*0.5 + 0.20*0.4 + 0.15*0.3."""
        r = _result(recall_at_k=0.6, precision_at_k=0.5, mrr=0.4, temporal_accuracy=0.3)
        expected = 0.40 * 0.6 + 0.25 * 0.5 + 0.20 * 0.4 + 0.15 * 0.3
        assert r.composite_score() == pytest.approx(expected, rel=1e-9)

    def test_recall_gate_zero(self) -> None:
        """recall=0.009 is below the 0.01 gate so composite must be 0.0."""
        r = _result(recall_at_k=0.009, precision_at_k=1.0, mrr=1.0, temporal_accuracy=1.0)
        assert r.composite_score() == 0.0

    def test_recall_gate_threshold(self) -> None:
        """recall=0.010 is at the gate; composite must be greater than 0."""
        r = _result(recall_at_k=0.010, precision_at_k=1.0, mrr=1.0, temporal_accuracy=1.0)
        assert r.composite_score() > 0.0

    def test_recall_gate_exactly_zero_recall(self) -> None:
        """recall=0.0 forces composite to 0.0 regardless of other metrics."""
        r = _result(recall_at_k=0.0, precision_at_k=1.0, mrr=1.0, temporal_accuracy=1.0)
        assert r.composite_score() == 0.0

    def test_perfect_retrieval(self) -> None:
        """All metrics = 1.0 must produce composite = 1.0."""
        r = _result(recall_at_k=1.0, precision_at_k=1.0, mrr=1.0, temporal_accuracy=1.0)
        assert r.composite_score() == pytest.approx(1.0, rel=1e-9)

    def test_weights_sum_correctly(self) -> None:
        """With all metrics=1.0, composite equals the sum of weights (must be 1.0)."""
        r = _result(recall_at_k=1.0, precision_at_k=1.0, mrr=1.0, temporal_accuracy=1.0)
        # 0.40 + 0.25 + 0.20 + 0.15 = 1.0
        assert r.composite_score() == pytest.approx(1.0, rel=1e-9)

    def test_higher_recall_wins(self) -> None:
        """Two results differing only in recall — higher recall wins."""
        r_low = _result(recall_at_k=0.5, precision_at_k=0.5, mrr=0.5, temporal_accuracy=0.5)
        r_high = _result(recall_at_k=0.8, precision_at_k=0.5, mrr=0.5, temporal_accuracy=0.5)
        assert r_high.composite_score() > r_low.composite_score()

    def test_range_is_0_to_1(self) -> None:
        """composite_score is always in [0.0, 1.0] for valid metric ranges."""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0, 1.0),
            (0.3, 0.9, 0.1, 0.7),
        ]
        for recall, precision, mrr, temporal in test_cases:
            r = _result(recall_at_k=recall, precision_at_k=precision, mrr=mrr, temporal_accuracy=temporal)
            score = r.composite_score()
            assert 0.0 <= score <= 1.0, f"composite={score} out of range for inputs {(recall, precision, mrr, temporal)}"


# ---------------------------------------------------------------------------
# 3. TestRecencyStrategy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRecencyStrategy:
    """Tests for RecencyStrategy retrieval scoring."""

    @staticmethod
    def _indexed(n: int, user_id: str = "u1") -> RecencyStrategy:
        """Create a RecencyStrategy indexed with n memories (oldest first)."""
        strategy = RecencyStrategy()
        memories = [_mem(f"mem_{i}", user_id=user_id) for i in range(n)]
        strategy.index(memories)
        return strategy

    def test_most_recent_first(self) -> None:
        """With 10 memories indexed oldest-first, top-3 should be mem_9, mem_8, mem_7."""
        strategy = self._indexed(10)
        results = strategy.retrieve("any query", top_k=3)
        ids = [r[0] for r in results]
        assert ids == ["mem_9", "mem_8", "mem_7"]

    def test_score_decreases_monotonically(self) -> None:
        """Each successive result in top-5 must have a strictly lower score."""
        strategy = self._indexed(10)
        results = strategy.retrieve("any query", top_k=5)
        scores = [r[1] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Score not decreasing at position {i}: {scores[i]} <= {scores[i+1]}"
            )

    def test_scores_bounded_0_to_1(self) -> None:
        """All recency scores must be in (0, 1]."""
        strategy = self._indexed(10)
        results = strategy.retrieve("x", top_k=10)
        for mem_id, score in results:
            assert 0.0 < score <= 1.0, f"Score {score} out of range for {mem_id}"

    def test_query_is_ignored(self) -> None:
        """RecencyStrategy is query-agnostic — different queries return the same result."""
        strategy = self._indexed(5)
        r1 = strategy.retrieve("hello world", top_k=5)
        r2 = strategy.retrieve("completely different query text", top_k=5)
        assert [x[0] for x in r1] == [x[0] for x in r2]

    def test_user_filter_respected(self) -> None:
        """With user_id filter, only that user's memories are returned."""
        strategy = RecencyStrategy()
        mems = [_mem(f"u1_mem_{i}", user_id="user1") for i in range(3)]
        mems += [_mem(f"u2_mem_{i}", user_id="user2") for i in range(3)]
        strategy.index(mems)
        results = strategy.retrieve("query", top_k=10, user_id="user1")
        ids = [r[0] for r in results]
        assert all(mid.startswith("u1_") for mid in ids)
        assert len(ids) == 3

    def test_empty_index_returns_empty(self) -> None:
        """Retrieve on an empty index must return an empty list."""
        strategy = RecencyStrategy()
        assert strategy.retrieve("query", top_k=5) == []

    def test_single_memory_score_is_1(self) -> None:
        """A single indexed memory must have score = 1.0 (position 0 of 1)."""
        strategy = RecencyStrategy()
        strategy.index([_mem("solo")])
        results = strategy.retrieve("query", top_k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(1.0, rel=1e-9)

    def test_top_k_respected(self) -> None:
        """retrieve(top_k=2) must return exactly 2 results."""
        strategy = self._indexed(10)
        results = strategy.retrieve("query", top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 4. TestBootstrapCI
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBootstrapCI:
    """Tests for StudyAggregator.bootstrap_ci() and significance_table()."""

    @staticmethod
    def _agg(group_values: dict[str, list[float]], strategy_key: str = "retrieval_strategy") -> StudyAggregator:
        """Build a StudyAggregator from {strategy_name: [recall, ...]} mapping."""
        results = []
        for strategy, recalls in group_values.items():
            for recall in recalls:
                r = _result(recall_at_k=recall, retrieval_strategy=strategy)
                results.append(r)
        return StudyAggregator(results)

    def test_mean_matches_sample_mean(self) -> None:
        """Bootstrap CI mean must closely approximate the true sample mean (within 0.01)."""
        values = [0.5, 0.6, 0.55, 0.58, 0.52]
        agg = self._agg({"bm25": values})
        ci = agg.bootstrap_ci(seed=42, n_bootstrap=1000)
        expected_mean = sum(values) / len(values)
        assert abs(ci["bm25"]["mean"] - expected_mean) < 0.01

    def test_ci_contains_true_mean(self) -> None:
        """95% CI must contain the true sample mean."""
        values = [0.4, 0.5, 0.45, 0.48, 0.51, 0.47]
        agg = self._agg({"embeddings": values})
        ci = agg.bootstrap_ci(seed=42, n_bootstrap=1000)
        stats = ci["embeddings"]
        true_mean = sum(values) / len(values)
        assert stats["ci_low"] <= true_mean <= stats["ci_high"]

    def test_ci_low_le_mean_le_ci_high(self) -> None:
        """ci_low ≤ mean ≤ ci_high must always hold."""
        agg = self._agg({"bm25": [0.3, 0.6, 0.45, 0.55, 0.5]})
        ci = agg.bootstrap_ci(seed=42)
        for _group, stats in ci.items():
            assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]

    def test_wider_variance_gives_wider_ci(self) -> None:
        """Higher spread in data must produce a wider CI than lower spread."""
        narrow_values = [0.50] * 10          # zero variance
        wide_values = [0.1, 0.9] * 5         # high variance, same mean
        agg_narrow = self._agg({"g": narrow_values})
        agg_wide = self._agg({"g": wide_values})
        ci_narrow = agg_narrow.bootstrap_ci(seed=42, n_bootstrap=1000)["g"]
        ci_wide = agg_wide.bootstrap_ci(seed=42, n_bootstrap=1000)["g"]
        width_narrow = ci_narrow["ci_high"] - ci_narrow["ci_low"]
        width_wide = ci_wide["ci_high"] - ci_wide["ci_low"]
        assert width_wide > width_narrow

    def test_different_groups_ranked_correctly(self) -> None:
        """embeddings (0.66) > bm25 (0.53) > recency (0.18) in significance table."""
        agg = self._agg({
            "embeddings": [0.66] * 8,
            "bm25": [0.53] * 8,
            "recency": [0.18] * 8,
        })
        table = agg.significance_table(n_bootstrap=200)
        groups = [row["group"] for row in table]
        assert groups.index("embeddings") < groups.index("bm25") < groups.index("recency")

    def test_nonoverlapping_ci_marked_significant(self) -> None:
        """Groups with clearly non-overlapping CIs must have sig_vs_next=True."""
        # embeddings clearly above bm25 with no overlap possible
        agg = self._agg({
            "embeddings": [0.90] * 20,
            "bm25": [0.10] * 20,
        })
        table = agg.significance_table(n_bootstrap=500)
        top_row = table[0]  # the better group
        assert top_row["sig_vs_next"] is True

    def test_overlapping_ci_not_significant(self) -> None:
        """Groups with identical means must have sig_vs_next=False."""
        agg = self._agg({
            "a": [0.50] * 5,
            "b": [0.50] * 5,
        })
        table = agg.significance_table(n_bootstrap=500)
        # Both groups have the same mean, so CIs fully overlap
        assert table[0]["sig_vs_next"] is False


# ---------------------------------------------------------------------------
# 5. TestRRFHybridFusion
# ---------------------------------------------------------------------------

class _FixedStrategy(RetrievalStrategy):
    """Mock retrieval strategy that returns a pre-configured ranked list."""

    def __init__(self, ranked: list[tuple[str, float]], strategy_name: str = "mock") -> None:
        self._ranked = ranked
        self._name = strategy_name

    def index(self, memories: list[MemoryEvent]) -> None:
        """No-op index for mock."""

    def retrieve(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[tuple[str, float]]:
        """Return fixed ranked list truncated to top_k."""
        return self._ranked[:top_k]

    def name(self) -> str:
        return self._name

    def clear(self) -> None:
        self._ranked = []

    @classmethod
    def is_available(cls) -> bool:
        return True


def _hybrid(bm25_ranked: list[tuple[str, float]], embed_ranked: list[tuple[str, float]], bm25_weight: float = 0.5) -> HybridStrategy:
    """Assemble a HybridStrategy with mock BM25 and embeddings sub-strategies."""
    bm25 = _FixedStrategy(bm25_ranked, "bm25")
    embed = _FixedStrategy(embed_ranked, "embeddings")
    return HybridStrategy(
        strategies=["bm25", "embeddings"],
        bm25_weight=bm25_weight,
        bm25_strategy=bm25,
        embeddings_strategy=embed,
    )


@pytest.mark.unit
class TestRRFHybridFusion:
    """Tests for HybridStrategy.retrieve() RRF logic."""

    def test_document_in_both_lists_scores_higher(self) -> None:
        """A doc at rank 1 in both BM25 and embeddings outranks a doc in only one list."""
        # "shared" appears at rank 0 in both; "only_bm25" only in BM25
        bm25_ranked = [("shared", 1.0), ("only_bm25", 0.9), ("other", 0.5)]
        embed_ranked = [("shared", 1.0), ("only_embed", 0.9), ("other2", 0.5)]
        hybrid = _hybrid(bm25_ranked, embed_ranked)
        results = hybrid.retrieve("query", top_k=5)
        scores = dict(results)
        assert scores["shared"] > scores.get("only_bm25", 0.0)
        assert scores["shared"] > scores.get("only_embed", 0.0)

    def test_rrf_k60_constant_used(self) -> None:
        """Verify the RRF formula uses k=60 by computing expected score manually."""
        # Single doc at rank 0 in both strategies; weight=0.5 for each
        bm25_ranked = [("doc_a", 1.0)]
        embed_ranked = [("doc_a", 1.0)]
        hybrid = _hybrid(bm25_ranked, embed_ranked, bm25_weight=0.5)
        results = hybrid.retrieve("q", top_k=1)
        # Expected: 0.5/(60+0+1) + 0.5/(60+0+1) = 1.0/61
        expected_score = 0.5 / (60 + 1) + 0.5 / (60 + 1)
        assert results[0][1] == pytest.approx(expected_score, rel=1e-9)

    def test_bm25_weight_1_returns_bm25_order(self) -> None:
        """When bm25_weight=1.0, fused ranking matches pure BM25 order."""
        bm25_ranked = [("bm25_first", 1.0), ("bm25_second", 0.5)]
        embed_ranked = [("bm25_second", 1.0), ("bm25_first", 0.5)]
        hybrid = _hybrid(bm25_ranked, embed_ranked, bm25_weight=1.0)
        results = hybrid.retrieve("q", top_k=2)
        ids = [r[0] for r in results]
        # With bm25_weight=1.0, embeddings contribute 0/61 per rank; BM25 dominates
        assert ids[0] == "bm25_first"
        assert ids[1] == "bm25_second"

    def test_bm25_weight_0_returns_embed_order(self) -> None:
        """When bm25_weight=0.0, fused ranking matches pure embedding order."""
        bm25_ranked = [("bm25_first", 1.0), ("embed_first", 0.5)]
        embed_ranked = [("embed_first", 1.0), ("bm25_first", 0.5)]
        hybrid = _hybrid(bm25_ranked, embed_ranked, bm25_weight=0.0)
        results = hybrid.retrieve("q", top_k=2)
        ids = [r[0] for r in results]
        assert ids[0] == "embed_first"
        assert ids[1] == "bm25_first"

    def test_top_k_respected(self) -> None:
        """retrieve(top_k=3) must return exactly 3 results."""
        bm25_ranked = [(f"d{i}", float(10 - i)) for i in range(10)]
        embed_ranked = [(f"d{i}", float(10 - i)) for i in range(10)]
        hybrid = _hybrid(bm25_ranked, embed_ranked)
        results = hybrid.retrieve("q", top_k=3)
        assert len(results) == 3
