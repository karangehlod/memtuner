"""Tests for arena backend wiring and judge-score propagation (ARENA_PLAN 1.3).

Two independent guarantees:
1. Judge scores survive every result round-trip (worker dict → parent result →
   grid CSV). Pre-fix, the scenario runner computed llm_judge_score and the
   study scheduler silently dropped it — judge-enabled runs discarded their
   own judge data.
2. External-backend cells (backend="mem0" etc.) carry their backend through
   cell → config → results, and never contaminate native strategy rankings.
"""

from __future__ import annotations

import pytest

from benchmark.workload.scheduler import MatrixRunResult
from benchmark.workload.study_matrix import DecaySpec, StudyCell
from benchmark.workload.study_scheduler import StudyRunResult, StudyScheduler


def _decay() -> DecaySpec:
    return DecaySpec(policy="none", lambda_value=0.0, pruning_threshold=0.15)


def _cell(backend: str = "native") -> StudyCell:
    return StudyCell(
        memory_type="episodic",
        retrieval_strategy="bm25",
        decay=_decay(),
        workload_profile="medium_qpd",
        backend=backend,
    )


def _study_result(**overrides) -> StudyRunResult:
    kwargs = dict(
        cell_id="c1", run_id="r1",
        memory_type="episodic", retrieval_strategy="bm25",
        decay_policy="none", lambda_value=0.0, pruning_threshold=0.0,
        workload_profile="medium_qpd", seed=42,
        recall_at_k=0.5, precision_at_k=0.1, mrr=0.4, ndcg=0.3,
        success=True,
    )
    kwargs.update(overrides)
    return StudyRunResult(**kwargs)


@pytest.mark.unit
class TestJudgeScorePropagation:
    """llm_judge_score must survive every serialization boundary."""

    def test_matrix_result_dict_roundtrip_preserves_judge_score(self) -> None:
        r = MatrixRunResult(
            cell_id="c", run_id="r", memory_type="episodic",
            retrieval_strategy="bm25", decay_policy="none", lambda_value=0.0,
            pruning_threshold=0.0, workload_profile="medium_qpd", seed=42,
            llm_judge_score=0.7312, llm_judge_queries=470,
        )
        d = r.to_dict()
        assert d["metrics"]["llm_judge_score"] == pytest.approx(0.7312)
        assert d["metrics"]["llm_judge_queries"] == 470

    def test_worker_dict_to_result_preserves_judge_score(self) -> None:
        # The exact path a study cell result takes back from the worker.
        source = _study_result(llm_judge_score=0.65, llm_judge_queries=100)
        d = source.to_dict()
        d["_study"] = {"study_phase": "phase1_baselines"}
        restored = StudyScheduler._dict_to_result(d)
        assert restored.llm_judge_score == pytest.approx(0.65)
        assert restored.llm_judge_queries == 100

    def test_judge_disabled_stays_none_not_zero(self) -> None:
        # None means "judge not enabled"; 0.0 would mean "every answer wrong".
        source = _study_result()  # judge fields defaulted
        d = source.to_dict()
        d["_study"] = {}
        restored = StudyScheduler._dict_to_result(d)
        assert restored.llm_judge_score is None
        assert restored.llm_judge_queries == 0

    def test_grid_csv_renders_blank_for_disabled_judge(self) -> None:
        from benchmark.workload.study_aggregator import StudyAggregator
        rows = StudyAggregator([_study_result()]).build_grid_table()
        assert rows[0]["llm_judge_score"] == ""

    def test_grid_csv_renders_value_for_enabled_judge(self) -> None:
        from benchmark.workload.study_aggregator import StudyAggregator
        rows = StudyAggregator(
            [_study_result(llm_judge_score=0.55, llm_judge_queries=30)]
        ).build_grid_table()
        assert rows[0]["llm_judge_score"] == pytest.approx(0.55)
        assert rows[0]["llm_judge_queries"] == 30

    def test_from_csv_row_parses_blank_judge_as_none(self) -> None:
        row = {"cell_id": "c", "llm_judge_score": "", "llm_judge_queries": ""}
        r = MatrixRunResult.from_csv_row(row)
        assert r.llm_judge_score is None
        assert r.llm_judge_queries == 0

    def test_from_csv_row_parses_judge_value(self) -> None:
        row = {"cell_id": "c", "llm_judge_score": "0.61", "llm_judge_queries": "42"}
        r = MatrixRunResult.from_csv_row(row)
        assert r.llm_judge_score == pytest.approx(0.61)
        assert r.llm_judge_queries == 42


@pytest.mark.unit
class TestExternalBackendCells:
    """backend="mem0" must flow cell → config → results, and stay out of rankings."""

    def test_native_default_and_distinct_cell_ids(self) -> None:
        native, mem0 = _cell(), _cell(backend="mem0")
        assert native.backend == "native"
        assert native.scoring_mode == "gold_ids"
        assert mem0.scoring_mode == "judge_primary"
        assert native.cell_id != mem0.cell_id

    def test_external_config_selects_adapter_module(self) -> None:
        cfg = _cell(backend="mem0").to_config_dict(evaluation_horizon=30)
        assert cfg["memory"]["enabled"]["long_term"] == ["mem0_store"]
        assert cfg["memory"]["enabled"]["short_term"] == []
        # No strategy → composer loads no embedding models for external cells.
        assert cfg["benchmark"]["retrieval_strategy"] == ""
        assert cfg["policies"]["module_policies"] == {}

    def test_external_config_loads_and_validates(self) -> None:
        from benchmark.config.loader import load_config_from_dict
        cfg = load_config_from_dict(_cell(backend="mem0").to_config_dict(30))
        assert cfg.memory.enabled.long_term == ["mem0_store"]

    def test_composer_registers_mem0_store(self) -> None:
        from benchmark.application.composer import BenchmarkComposer
        composer = BenchmarkComposer()
        assert composer._registry.is_registered("mem0_store")

    def test_backend_survives_cell_dict_roundtrip(self) -> None:
        cell = _cell(backend="mem0")
        summary = cell.to_summary_dict()
        assert summary["backend"] == "mem0"
        assert summary["scoring_mode"] == "judge_primary"

    def test_backend_survives_result_roundtrip(self) -> None:
        source = _study_result(backend="mem0", scoring_mode="judge_primary")
        d = source.to_dict()
        d["_study"] = {"backend": "mem0", "scoring_mode": "judge_primary"}
        restored = StudyScheduler._dict_to_result(d)
        assert restored.backend == "mem0"
        assert restored.scoring_mode == "judge_primary"

    def test_grid_csv_carries_backend_columns(self) -> None:
        from benchmark.workload.study_aggregator import StudyAggregator
        rows = StudyAggregator(
            [_study_result(backend="mem0", scoring_mode="judge_primary")]
        ).build_grid_table()
        assert rows[0]["backend"] == "mem0"
        assert rows[0]["scoring_mode"] == "judge_primary"

    def test_external_cells_excluded_from_native_rankings(self) -> None:
        from benchmark.workload.study_aggregator import StudyAggregator
        native = _study_result(cell_id="n", recall_at_k=0.3)
        external = _study_result(
            cell_id="x", backend="mem0", scoring_mode="judge_primary",
            retrieval_strategy="bm25", recall_at_k=0.9,
        )
        agg = StudyAggregator([native, external])
        ranking = agg.rank_by_retrieval_strategy()
        bm25 = next(row for row in ranking if row["retrieval_strategy"] == "bm25")
        # External 0.9 must not inflate the native bm25 average of 0.3.
        assert bm25["avg_recall"] == pytest.approx(0.3)


@pytest.mark.unit
class TestArenaCells:
    """Arena runner cell construction (scripts/arena_runner.py)."""

    def test_full_system_config_enables_all_stores(self) -> None:
        cell = StudyCell(
            memory_type="all", retrieval_strategy="bm25l", decay=_decay(),
            workload_profile="arena",
        )
        cfg = cell.to_config_dict(evaluation_horizon=30)
        assert cfg["memory"]["enabled"]["long_term"] == [
            "episodic_store", "semantic_store", "preference_store",
        ]
        assert set(cfg["policies"]["module_policies"]) == {
            "episodic_store", "semantic_store", "preference_store",
        }

    def test_build_arena_cells_shape(self) -> None:
        from scripts.arena_runner import build_arena_cells
        cells = build_arena_cells(
            systems=["native", "mem0"], seeds=[42, 123],
            test_holdout_fraction=0.2,
        )
        assert len(cells) == 4  # 2 systems × 2 seeds
        native = [c for c in cells if c.backend == "native"]
        external = [c for c in cells if c.backend == "mem0"]
        assert len(native) == len(external) == 2
        assert all(c.memory_type == "all" for c in cells)
        assert all(c.study_phase == "arena" for c in cells)
        assert all(c.test_holdout_fraction == 0.2 for c in cells)
        # External cells run sequentially (vendor API), never in the thread pool.
        from benchmark.workload.study_scheduler import _STRATEGIES_SAFE_IN_THREADPOOL
        assert all(
            c.retrieval_strategy not in _STRATEGIES_SAFE_IN_THREADPOOL
            for c in external
        )

    def test_build_arena_cells_applies_frozen_baseline(self) -> None:
        from scripts.arena_runner import build_arena_cells
        baseline = {
            "retrieval_strategy": "hybrid",
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "embedding_backend": "sentence-transformers",
            "bm25_weight": 0.6,
        }
        (cell,) = build_arena_cells(["native"], [42], 0.2, baseline)
        assert cell.retrieval_strategy == "hybrid"
        assert cell.embedding_model == "BAAI/bge-base-en-v1.5"
        assert cell.bm25_weight == 0.6


@pytest.mark.unit
class TestPerModuleStrategyInstances:
    """Multi-store configs must not share one retrieval-strategy instance.

    Stores index their own memories into their strategy; a shared instance is
    wiped by whichever store indexes last (an empty semantic store zeroed the
    episodic store's recall — found by the first memory_type="all" arena run).
    """

    def test_each_store_gets_its_own_strategy(self) -> None:
        from benchmark.application.composer import BenchmarkComposer

        cell = StudyCell(
            memory_type="all", retrieval_strategy="bm25", decay=_decay(),
            workload_profile="arena",
        )
        from benchmark.config.loader import load_config_from_dict
        config = load_config_from_dict(cell.to_config_dict(evaluation_horizon=30))

        composer = BenchmarkComposer()
        from benchmark.factory.resolver import ConfigResolver
        resolver = ConfigResolver(composer._registry, composer._strategy_registry)
        strategy = composer._resolve_strategy(
            config, config.benchmark.retrieval_strategy, resolver, allow_fallback=False
        )
        modules = resolver.resolve_memory_modules(
            config,
            retrieval_strategy=strategy,
            strategy_factory=lambda: composer._resolve_strategy(
                config, config.benchmark.retrieval_strategy, resolver, allow_fallback=False
            ),
        )
        strategies = [m._retrieval_strategy for m in modules.values()]
        assert len(strategies) == 3
        assert len({id(s) for s in strategies}) == 3, (
            "stores share a retrieval-strategy instance — the last store to "
            "index wipes the others' index"
        )
