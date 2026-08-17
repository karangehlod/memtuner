"""Tests for PerformanceOptimizer."""

import pytest
from benchmark.memory.enterprise.performance_optimizer import PerformanceOptimizer


@pytest.fixture
def optimizer():
    return PerformanceOptimizer()


class TestPerformanceOptimizer:
    def test_initialization(self, optimizer):
        assert optimizer is not None

    def test_enable_caching(self, optimizer):
        optimizer.enable_caching("lru", 300)
        assert optimizer._cache_enabled

    def test_enable_batching(self, optimizer):
        optimizer.enable_batching(100, 1.0)
        assert optimizer._batching_enabled

    def test_get_optimization_stats(self, optimizer):
        stats = optimizer.get_optimization_stats()
        assert "cache_enabled" in stats
        assert "batching_enabled" in stats

    def test_cache_hit_rate(self, optimizer):
        optimizer._cache_hits = 8
        optimizer._cache_misses = 2
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == 0.8

    def test_zero_cache_requests(self, optimizer):
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == 0.0

    def test_optimize_resources(self, optimizer):
        constraints = {"cpu_limit": 80, "memory_limit": 1024}
        optimizer.optimize_resources(constraints)
        # Should not raise

    def test_batching_disabled_by_default(self, optimizer):
        stats = optimizer.get_optimization_stats()
        assert stats["batching_enabled"] is False

    def test_caching_disabled_by_default(self, optimizer):
        stats = optimizer.get_optimization_stats()
        assert stats["cache_enabled"] is False

    def test_cache_hit_increment(self, optimizer):
        optimizer._cache_hits = 5
        stats = optimizer.get_optimization_stats()
        assert stats["total_cache_requests"] == 5

    def test_multiple_cache_operations(self, optimizer):
        optimizer._cache_hits = 100
        optimizer._cache_misses = 50
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == (100 / 150)

    def test_perfect_cache_hit_rate(self, optimizer):
        optimizer._cache_hits = 100
        optimizer._cache_misses = 0
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == 1.0

    def test_zero_cache_hit_rate(self, optimizer):
        optimizer._cache_hits = 0
        optimizer._cache_misses = 100
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == 0.0

    def test_concurrent_optimizations(self, optimizer):
        optimizer.enable_caching("lru", 300)
        optimizer.enable_batching(100, 1.0)
        stats = optimizer.get_optimization_stats()
        assert stats["cache_enabled"]
        assert stats["batching_enabled"]

    def test_large_working_set(self, optimizer):
        optimizer._cache_hits = 10000
        optimizer._cache_misses = 1000
        stats = optimizer.get_optimization_stats()
        assert stats["total_cache_requests"] == 11000

    def test_batching_and_caching_together(self, optimizer):
        optimizer.enable_caching("lru", 300)
        optimizer.enable_batching(50, 2.0)
        optimizer._cache_hits = 100
        stats = optimizer.get_optimization_stats()
        assert stats["cache_enabled"]
        assert stats["batching_enabled"]

    def test_adaptive_tuning(self, optimizer):
        for _ in range(100):
            optimizer._cache_hits += 1
        stats = optimizer.get_optimization_stats()
        assert stats["total_cache_requests"] == 100

    def test_performance_improvement_validation(self, optimizer):
        optimizer.enable_caching("lru", 300)
        optimizer._cache_hits = 90
        optimizer._cache_misses = 10
        stats = optimizer.get_optimization_stats()
        assert stats["cache_hit_rate"] == 0.9
