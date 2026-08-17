"""Comprehensive tests for MemoryConsolidationEngine."""

import pytest
from datetime import datetime, timedelta
from benchmark.memory.strategies.memory_consolidation_engine import (
    MemoryConsolidationEngine,
    MemoryRecord,
)


class TestMemoryConsolidationEngineInitialization:
    """Test engine initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default capacities."""
        engine = MemoryConsolidationEngine()
        assert engine.capacities["working"] == 100
        assert engine.capacities["episodic"] == 1000
        assert engine.capacities["semantic"] == 10000

    def test_initialization_with_custom_capacities(self):
        """Test initialization with custom capacities."""
        engine = MemoryConsolidationEngine(
            working_capacity=50,
            episodic_capacity=500,
            semantic_capacity=5000,
        )
        assert engine.capacities["working"] == 50
        assert engine.capacities["episodic"] == 500


class TestAddMemory:
    """Test adding memories to tiers."""

    def test_add_memory_to_working_tier(self):
        """Test adding memory to working tier."""
        engine = MemoryConsolidationEngine()
        result = engine.add_memory("m1", {"content": "test"}, tier="working")
        assert result is True

    def test_add_memory_invalid_tier_raises_error(self):
        """Test adding to invalid tier raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError, match="Unknown tier"):
            engine.add_memory("m1", {"content": "test"}, tier="invalid")

    def test_add_memory_invalid_value_score_raises_error(self):
        """Test invalid value score raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError, match="Value score must be"):
            engine.add_memory("m1", {"content": "test"}, value_score=1.5)

    def test_add_memory_respects_capacity(self):
        """Test that capacity limits are enforced."""
        engine = MemoryConsolidationEngine(working_capacity=2)
        assert engine.add_memory("m1", {"content": "test"}) is True
        assert engine.add_memory("m2", {"content": "test"}) is True
        assert engine.add_memory("m3", {"content": "test"}) is False

    def test_add_memory_updates_tier_mapping(self):
        """Test that tier mapping is updated."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="episodic")
        assert engine.get_memory_tier("m1") == "episodic"


class TestPromotionDemotion:
    """Test memory promotion and demotion."""

    def test_promote_memory_up_tier(self):
        """Test promoting memory to next tier."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        result = engine.promote("m1")
        assert result is True
        assert engine.get_memory_tier("m1") == "episodic"

    def test_promote_to_specific_tier(self):
        """Test promoting to specific target tier."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        result = engine.promote("m1", target_tier="semantic")
        assert result is True
        assert engine.get_memory_tier("m1") == "semantic"

    def test_promote_nonexistent_memory_raises_error(self):
        """Test promoting nonexistent memory raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError):
            engine.promote("m_nonexistent")

    def test_promote_already_top_tier_returns_false(self):
        """Test promoting semantic tier returns False."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="semantic")

        result = engine.promote("m1")
        assert result is False

    def test_demote_memory_down_tier(self):
        """Test demoting memory to lower tier."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="episodic")

        result = engine.demote("m1")
        assert result is True
        assert engine.get_memory_tier("m1") == "working"

    def test_demote_to_specific_tier(self):
        """Test demoting to specific tier."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="semantic")

        result = engine.demote("m1", target_tier="working")
        assert result is True
        assert engine.get_memory_tier("m1") == "working"

    def test_demote_already_bottom_tier_returns_false(self):
        """Test demoting working tier returns False."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        result = engine.demote("m1")
        assert result is False


class TestArchiveReactivate:
    """Test archiving and reactivating memories."""

    def test_archive_memory(self):
        """Test archiving a memory."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        result = engine.archive("m1")
        assert result is True
        assert engine.get_memory_tier("m1") is None

    def test_archive_nonexistent_memory_raises_error(self):
        """Test archiving nonexistent memory raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError):
            engine.archive("m_nonexistent")

    def test_reactivate_archived_memory(self):
        """Test reactivating archived memory."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")
        engine.archive("m1")

        result = engine.reactivate("m1", target_tier="episodic")
        assert result is True
        assert engine.get_memory_tier("m1") == "episodic"

    def test_reactivate_nonarchived_memory_raises_error(self):
        """Test reactivating non-archived memory raises error."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        with pytest.raises(ValueError):
            engine.reactivate("m1")

    def test_reactivate_to_full_tier_returns_false(self):
        """Test reactivate fails if target tier is full."""
        engine = MemoryConsolidationEngine(working_capacity=1)
        engine.add_memory("m1", {"content": "test"}, tier="working")
        engine.add_memory("m2", {"content": "test"}, tier="episodic")
        engine.archive("m2")

        result = engine.reactivate("m2", target_tier="working")
        assert result is False


class TestConsolidation:
    """Test consolidation operations."""

    def test_consolidate_executes_successfully(self):
        """Test basic consolidation."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, value_score=0.9)
        engine.add_memory("m2", {"content": "test"}, value_score=0.1)

        metrics = engine.consolidate()
        assert metrics.total_consolidations == 1
        assert metrics.elapsed_ms >= 0

    def test_consolidate_promotes_high_value(self):
        """Test that high-value items are promoted."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working", value_score=0.75)

        engine.consolidate()
        # Should promote at least to episodic (0.7 threshold)
        tier = engine.get_memory_tier("m1")
        assert tier in ["episodic", "semantic"]

    def test_consolidate_demotes_low_value(self):
        """Test that low-value items are demoted."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="episodic", value_score=0.1)

        engine.consolidate()
        assert engine.get_memory_tier("m1") == "working"

    def test_consolidate_applies_decay(self):
        """Test that decay is applied."""
        engine = MemoryConsolidationEngine()
        now = datetime.now()
        old_time = now - timedelta(days=10)

        engine.add_memory("m1", {"content": "test"}, value_score=1.0)
        record = engine._tiers["working"]["m1"]
        record.created_at = old_time

        engine.consolidate(reference_time=now)

        # Value should be decayed
        assert record.value_score < 1.0


class TestDecayStrategies:
    """Test different decay strategies."""

    def test_set_decay_strategy_linear(self):
        """Test setting linear decay."""
        engine = MemoryConsolidationEngine()
        engine.set_decay_strategy("linear", lambda_param=0.1)
        assert engine._decay_strategy == "linear"

    def test_set_decay_strategy_exponential(self):
        """Test setting exponential decay."""
        engine = MemoryConsolidationEngine()
        engine.set_decay_strategy("exponential", lambda_param=0.1)
        assert engine._decay_strategy == "exponential"

    def test_set_decay_strategy_power_law(self):
        """Test setting power law decay."""
        engine = MemoryConsolidationEngine()
        engine.set_decay_strategy("power_law", lambda_param=0.5)
        assert engine._decay_strategy == "power_law"

    def test_set_decay_strategy_selective(self):
        """Test setting selective decay."""
        engine = MemoryConsolidationEngine()
        engine.set_decay_strategy("selective", lambda_param=0.1)
        assert engine._decay_strategy == "selective"

    def test_set_invalid_decay_strategy_raises_error(self):
        """Test invalid decay strategy raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError):
            engine.set_decay_strategy("invalid")

    def test_decay_linear_diminishes_value(self):
        """Test linear decay diminishes value."""
        engine = MemoryConsolidationEngine()
        engine.set_decay_strategy("linear", lambda_param=0.1)

        engine.add_memory("m1", {"content": "test"}, value_score=1.0)
        initial_value = engine._tiers["working"]["m1"].value_score

        engine.consolidate(reference_time=datetime.now() + timedelta(days=5))

        final_value = engine._tiers["working"]["m1"].value_score
        assert final_value < initial_value

    def test_decay_exponential_slower_than_linear(self):
        """Test exponential decay slower than linear."""
        engine1 = MemoryConsolidationEngine()
        engine1.set_decay_strategy("linear", lambda_param=0.1)
        engine1.add_memory("m1", {"content": "test"}, value_score=1.0)

        engine2 = MemoryConsolidationEngine()
        engine2.set_decay_strategy("exponential", lambda_param=0.1)
        engine2.add_memory("m1", {"content": "test"}, value_score=1.0)

        future = datetime.now() + timedelta(days=5)
        engine1.consolidate(reference_time=future)
        engine2.consolidate(reference_time=future)

        # Exponential should decay slower (higher final value)
        assert engine2._tiers["working"]["m1"].value_score > engine1._tiers["working"]["m1"].value_score


class TestTierStatistics:
    """Test statistics and metrics."""

    def test_get_tier_statistics(self):
        """Test getting tier statistics."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, value_score=0.8)
        engine.add_memory("m2", {"content": "test"}, value_score=0.6)

        stats = engine.get_tier_statistics("working")
        assert stats["size"] == 2
        assert stats["capacity"] == 100
        assert stats["utilization"] == pytest.approx(0.02)

    def test_get_tier_statistics_empty_tier(self):
        """Test statistics for empty tier."""
        engine = MemoryConsolidationEngine()

        stats = engine.get_tier_statistics("working")
        assert stats["size"] == 0
        assert stats["avg_value"] == 0.0

    def test_get_tier_statistics_invalid_tier_raises_error(self):
        """Test invalid tier raises error."""
        engine = MemoryConsolidationEngine()
        with pytest.raises(ValueError):
            engine.get_tier_statistics("invalid")

    def test_get_all_tier_statistics(self):
        """Test getting statistics for all tiers."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")
        engine.add_memory("m2", {"content": "test"}, tier="episodic")

        all_stats = engine.get_all_tier_statistics()
        assert "working" in all_stats
        assert "episodic" in all_stats
        assert "semantic" in all_stats
        assert all_stats["working"]["size"] == 1
        assert all_stats["episodic"]["size"] == 1


class TestAccessTracking:
    """Test access counting and updates."""

    def test_record_access_increments_count(self):
        """Test recording access increments counter."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"})

        engine.record_access("m1")
        engine.record_access("m1")

        record = engine._tiers["working"]["m1"]
        assert record.access_count == 2

    def test_record_access_updates_timestamp(self):
        """Test recording access updates last access time."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"})

        old_time = engine._tiers["working"]["m1"].last_accessed
        engine.record_access("m1")
        new_time = engine._tiers["working"]["m1"].last_accessed

        assert new_time > old_time

    def test_record_access_nonexistent_returns_false(self):
        """Test recording access on nonexistent memory returns False."""
        engine = MemoryConsolidationEngine()
        result = engine.record_access("m_nonexistent")
        assert result is False

    def test_update_value_score(self):
        """Test updating value score."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, value_score=0.5)

        result = engine.update_value_score("m1", 0.9)
        assert result is True

        record = engine._tiers["working"]["m1"]
        assert record.value_score == 0.9

    def test_update_value_score_invalid_score_raises_error(self):
        """Test updating with invalid score raises error."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"})

        with pytest.raises(ValueError):
            engine.update_value_score("m1", 1.5)

    def test_update_value_score_nonexistent_returns_false(self):
        """Test updating nonexistent memory returns False."""
        engine = MemoryConsolidationEngine()
        result = engine.update_value_score("m_nonexistent", 0.5)
        assert result is False


class TestCapacityEnforcement:
    """Test capacity limits are enforced."""

    def test_capacity_working_tier(self):
        """Test working tier capacity is enforced."""
        engine = MemoryConsolidationEngine(working_capacity=3)

        for i in range(3):
            assert engine.add_memory(f"m{i}", {"content": "test"}) is True

        assert engine.add_memory("m3", {"content": "test"}) is False

    def test_capacity_episodic_tier(self):
        """Test episodic tier capacity is enforced."""
        engine = MemoryConsolidationEngine(episodic_capacity=2)

        engine.add_memory("m1", {"content": "test"}, tier="episodic")
        engine.add_memory("m2", {"content": "test"}, tier="episodic")

        result = engine.add_memory("m3", {"content": "test"}, tier="episodic")
        assert result is False


class TestMetricsTracking:
    """Test metrics tracking."""

    def test_metrics_promoted_incremented(self):
        """Test promoted count is tracked."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="working")

        engine.promote("m1")
        assert engine.metrics.promoted == 1

    def test_metrics_demoted_incremented(self):
        """Test demoted count is tracked."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, tier="episodic")

        engine.demote("m1")
        assert engine.metrics.demoted == 1

    def test_metrics_archived_incremented(self):
        """Test archived count is tracked."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"})

        engine.archive("m1")
        assert engine.metrics.archived == 1

    def test_metrics_reactivated_incremented(self):
        """Test reactivated count is tracked."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"})
        engine.archive("m1")

        engine.reactivate("m1")
        assert engine.metrics.reactivated == 1


class TestMemoryRecordAgeCalculation:
    """Test MemoryRecord age calculation."""

    def test_memory_record_age_calculation(self):
        """Test age calculation in days."""
        now = datetime.now()
        old_time = now - timedelta(days=5)

        record = MemoryRecord(
            memory_id="m1",
            content={},
            tier="working",
            created_at=old_time,
            last_accessed=now,
        )

        age = record.age_days(now)
        assert 4.9 < age < 5.1

    def test_memory_record_last_access_days(self):
        """Test last access days calculation."""
        now = datetime.now()
        past = now - timedelta(days=3)

        record = MemoryRecord(
            memory_id="m1",
            content={},
            tier="working",
            created_at=now - timedelta(days=10),
            last_accessed=past,
        )

        days_since = record.last_access_days(now)
        assert 2.9 < days_since < 3.1


class TestConsolidationEdgeCases:
    """Test edge cases and special scenarios."""

    def test_consolidation_idempotent(self):
        """Test that consolidation is idempotent."""
        engine = MemoryConsolidationEngine()
        engine.add_memory("m1", {"content": "test"}, value_score=0.5)

        engine.consolidate()
        tier1 = engine.get_memory_tier("m1")

        engine.consolidate()
        tier2 = engine.get_memory_tier("m1")

        assert tier1 == tier2

    def test_empty_engine_consolidation(self):
        """Test consolidation on empty engine."""
        engine = MemoryConsolidationEngine()

        metrics = engine.consolidate()
        assert metrics.total_consolidations == 1

    def test_large_scale_consolidation(self):
        """Test consolidation with many memories."""
        engine = MemoryConsolidationEngine(
            working_capacity=500,
            episodic_capacity=5000,
            semantic_capacity=50000,
        )

        # Add 1000 memories
        for i in range(1000):
            value = 0.2 + (i % 10) / 10.0  # Vary value 0.2-1.1
            engine.add_memory(f"m{i}", {"content": f"test {i}"}, value_score=min(1.0, value))

        metrics = engine.consolidate()
        assert metrics.total_consolidations == 1
        assert metrics.elapsed_ms >= 0
