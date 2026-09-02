"""Tests for the matrix expander — Phase 7 / 11."""

from __future__ import annotations

import pytest

from benchmark.workload.matrix import (
    LAMBDA_STEPS,
    DecaySpec,
    MatrixCell,
    MatrixExpander,
)


@pytest.mark.unit
class TestDecaySpec:
    def test_none_policy_label(self):
        d = DecaySpec(policy="none")
        assert d.label == "none"

    def test_exponential_label_includes_lambda(self):
        d = DecaySpec(policy="exponential", lambda_value=0.05)
        assert "0.05" in d.label
        assert "exponential" in d.label

    def test_periodic_label(self):
        d = DecaySpec(policy="periodic")
        assert "periodic" in d.label

    def test_config_dict_none(self):
        d = DecaySpec(policy="none")
        cfg = d.to_config_dict()
        assert cfg["decay"]["lambda"] == 0.0

    def test_config_dict_exponential(self):
        d = DecaySpec(policy="exponential", lambda_value=0.10)
        cfg = d.to_config_dict()
        assert cfg["decay"]["type"] == "exponential"
        assert cfg["decay"]["lambda"] == 0.10

    def test_config_dict_linear(self):
        d = DecaySpec(policy="linear", lambda_value=0.07)
        cfg = d.to_config_dict()
        assert cfg["decay"]["type"] == "linear"

    def test_config_dict_includes_pruning(self):
        d = DecaySpec(policy="exponential", lambda_value=0.05, pruning_threshold=0.30)
        cfg = d.to_config_dict()
        assert "pruning" in cfg
        assert cfg["pruning"]["threshold"] == 0.30


@pytest.mark.unit
class TestMatrixCell:
    def _make_cell(self, memory_type="episodic", strategy="bm25", policy="none"):
        decay = DecaySpec(policy=policy)
        return MatrixCell(
            memory_type=memory_type,
            retrieval_strategy=strategy,
            decay=decay,
            workload_profile="medium_qpd",
        )

    def test_cell_id_is_deterministic(self):
        c1 = self._make_cell()
        c2 = self._make_cell()
        assert c1.cell_id == c2.cell_id

    def test_cell_id_differs_for_different_cells(self):
        c1 = self._make_cell(memory_type="episodic")
        c2 = self._make_cell(memory_type="semantic")
        assert c1.cell_id != c2.cell_id

    def test_label_contains_all_axes(self):
        c = self._make_cell(memory_type="preference", strategy="embeddings", policy="none")
        assert "preference" in c.label
        assert "embeddings" in c.label

    def test_to_config_dict_sets_memory_module(self):
        c = self._make_cell(memory_type="semantic")
        cfg = c.to_config_dict(evaluation_horizon=14)
        assert "semantic_store" in cfg["memory"]["enabled"]["long_term"]

    def test_to_config_dict_sets_strategy(self):
        c = self._make_cell(strategy="hybrid")
        cfg = c.to_config_dict(evaluation_horizon=14)
        assert cfg["benchmark"]["retrieval_strategy"] == "hybrid"

    def test_to_config_dict_sets_evaluation_horizon(self):
        c = self._make_cell()
        cfg = c.to_config_dict(evaluation_horizon=30)
        assert cfg["benchmark"]["evaluation_horizon"] == 30

    def test_to_summary_dict_is_serializable(self):
        import json
        c = self._make_cell()
        d = c.to_summary_dict()
        # Must serialize without error
        json.dumps(d)

    def test_to_summary_dict_has_all_keys(self):
        c = self._make_cell()
        d = c.to_summary_dict()
        required = {"cell_id", "memory_type", "retrieval_strategy", "decay_policy", "lambda", "workload_profile"}
        assert required.issubset(d.keys())


@pytest.mark.unit
class TestMatrixExpander:
    def test_core_3x3_returns_27_cells(self):
        expander = MatrixExpander()
        cells = expander.expand_core_3x3()
        # 3 memory × 3 strategies × 3 decay policies × 1 lambda step
        assert len(cells) == 27

    def test_core_3x3_has_correct_axes(self):
        expander = MatrixExpander()
        cells = expander.expand_core_3x3()
        memory_types = {c.memory_type for c in cells}
        strategies = {c.retrieval_strategy for c in cells}
        decay_policies = {c.decay.policy for c in cells}
        assert memory_types == {"episodic", "semantic", "preference"}
        assert strategies == {"bm25", "embeddings", "hybrid"}
        assert "none" in decay_policies
        assert "exponential" in decay_policies
        assert "logarithmic" in decay_policies

    def test_all_cells_have_unique_ids(self):
        expander = MatrixExpander()
        cells = expander.expand_core_3x3()
        ids = [c.cell_id for c in cells]
        assert len(ids) == len(set(ids))

    def test_lambda_sweep_returns_7_cells(self):
        expander = MatrixExpander()
        cells = expander.expand_lambda_sweep(
            memory_type="episodic",
            strategy="bm25",
            decay_policy="exponential",
        )
        assert len(cells) == len(LAMBDA_STEPS)

    def test_lambda_sweep_covers_all_steps(self):
        expander = MatrixExpander()
        cells = expander.expand_lambda_sweep(
            memory_type="episodic",
            strategy="bm25",
            decay_policy="exponential",
        )
        lambdas = sorted(c.decay.lambda_value for c in cells)
        assert lambdas == sorted(LAMBDA_STEPS)

    def test_none_policy_has_zero_lambda(self):
        expander = MatrixExpander()
        cells = expander.expand_full(decay_policies=["none"])
        for c in cells:
            assert c.decay.lambda_value == 0.0

    def test_describe_returns_correct_counts(self):
        expander = MatrixExpander()
        cells = expander.expand_core_3x3()
        desc = expander.describe(cells)
        assert desc["total_cells"] == 27
        assert len(desc["memory_types"]) == 3
        assert len(desc["retrieval_strategies"]) == 3

    def test_expand_full_memory_types_filter(self):
        expander = MatrixExpander()
        cells = expander.expand_full(memory_types=["episodic"])
        for c in cells:
            assert c.memory_type == "episodic"

    def test_expand_full_strategy_filter(self):
        expander = MatrixExpander()
        cells = expander.expand_full(strategies=["bm25"], memory_types=["episodic"], decay_policies=["none"])
        for c in cells:
            assert c.retrieval_strategy == "bm25"

    def test_deterministic_same_seed(self):
        expander = MatrixExpander()
        c1 = expander.expand_core_3x3(seed=42)
        c2 = expander.expand_core_3x3(seed=42)
        assert [c.cell_id for c in c1] == [c.cell_id for c in c2]

    def test_different_seeds_same_structure(self):
        expander = MatrixExpander()
        c1 = expander.expand_core_3x3(seed=42)
        c2 = expander.expand_core_3x3(seed=99)
        # Same number of cells, same axes — only seed differs
        assert len(c1) == len(c2)
