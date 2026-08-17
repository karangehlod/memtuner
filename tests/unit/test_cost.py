"""Unit tests for cost tracking: CostTracker, TokenCostCalculator, StorageCostCalculator."""

from __future__ import annotations

import pytest

from benchmark.cost.storage_cost import StorageCostCalculator
from benchmark.cost.token_cost import TokenCostCalculator
from benchmark.cost.tracker import CostEntry, InMemoryCostTracker
from benchmark.models.answer import TokenUsage


@pytest.mark.unit
class TestInMemoryCostTracker:
    """Tests for the InMemoryCostTracker."""

    def test_empty_tracker_zero_cost(self) -> None:
        tracker = InMemoryCostTracker()
        assert tracker.total_cost_usd() == 0.0

    def test_record_single_entry(self) -> None:
        tracker = InMemoryCostTracker()
        entry = CostEntry(source="test", amount_usd=0.01)
        tracker.record(entry)
        assert tracker.total_cost_usd() == pytest.approx(0.01)

    def test_record_multiple_entries_accumulates(self) -> None:
        tracker = InMemoryCostTracker()
        tracker.record(CostEntry(source="read", amount_usd=0.001))
        tracker.record(CostEntry(source="write", amount_usd=0.005))
        tracker.record(CostEntry(source="read", amount_usd=0.002))
        assert tracker.total_cost_usd() == pytest.approx(0.008)

    def test_entries_by_source(self) -> None:
        tracker = InMemoryCostTracker()
        tracker.record(CostEntry(source="read", amount_usd=0.001))
        tracker.record(CostEntry(source="read", amount_usd=0.002))
        tracker.record(CostEntry(source="write", amount_usd=0.005))
        by_source = tracker.entries_by_source()
        assert by_source["read"] == pytest.approx(0.003)
        assert by_source["write"] == pytest.approx(0.005)

    def test_reset_clears_entries(self) -> None:
        tracker = InMemoryCostTracker()
        tracker.record(CostEntry(source="test", amount_usd=1.0))
        tracker.reset()
        assert tracker.total_cost_usd() == 0.0


@pytest.mark.unit
class TestTokenCostCalculator:
    """Tests for the TokenCostCalculator."""

    def test_gpt4o_cost(self) -> None:
        calculator = TokenCostCalculator()
        usage = TokenUsage(prompt=1000, completion=500)
        entry = calculator.compute_cost(usage, "gpt-4o")
        expected = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
        assert entry.amount_usd == pytest.approx(expected)
        assert entry.source == "llm_tokens"

    def test_unknown_model_uses_default(self) -> None:
        calculator = TokenCostCalculator()
        usage = TokenUsage(prompt=1000, completion=1000)
        entry = calculator.compute_cost(usage, "unknown-model-xyz")
        expected = (1000 / 1000) * 0.01 + (1000 / 1000) * 0.03
        assert entry.amount_usd == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self) -> None:
        calculator = TokenCostCalculator()
        usage = TokenUsage(prompt=0, completion=0)
        entry = calculator.compute_cost(usage, "gpt-4o")
        assert entry.amount_usd == 0.0

    def test_cost_entry_has_details(self) -> None:
        calculator = TokenCostCalculator()
        usage = TokenUsage(prompt=100, completion=50)
        entry = calculator.compute_cost(usage, "gpt-4o")
        assert "prompt_tokens" in entry.details
        assert "completion_tokens" in entry.details


@pytest.mark.unit
class TestStorageCostCalculator:
    """Tests for the StorageCostCalculator."""

    def test_single_read_cost(self) -> None:
        calculator = StorageCostCalculator()
        entry = calculator.compute_read_cost()
        assert entry.amount_usd == pytest.approx(0.000001)
        assert entry.source == "storage_read"

    def test_batch_read_cost(self) -> None:
        calculator = StorageCostCalculator()
        entry = calculator.compute_read_cost(operation_count=1000)
        assert entry.amount_usd == pytest.approx(0.001)

    def test_single_write_cost(self) -> None:
        calculator = StorageCostCalculator()
        entry = calculator.compute_write_cost()
        assert entry.amount_usd == pytest.approx(0.000005)
        assert entry.source == "storage_write"

    def test_batch_write_cost(self) -> None:
        calculator = StorageCostCalculator()
        entry = calculator.compute_write_cost(operation_count=1000)
        assert entry.amount_usd == pytest.approx(0.005)
