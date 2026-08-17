"""Tests for LoadBalancer."""

import pytest
from benchmark.memory.enterprise.load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    Worker,
    Task,
)


@pytest.fixture
def balancer():
    return LoadBalancer()


@pytest.fixture
def workers():
    return [
        Worker(1, cpu_usage=10.0, memory_usage=20.0),
        Worker(2, cpu_usage=30.0, memory_usage=40.0),
        Worker(3, cpu_usage=5.0, memory_usage=10.0),
    ]


class TestLoadBalancer:
    def test_initialization(self, balancer):
        assert balancer is not None

    def test_round_robin_selection(self, balancer, workers):
        balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        selected = balancer.select_worker(workers)
        assert selected in workers

    def test_least_loaded_selection(self, balancer, workers):
        balancer.strategy = LoadBalancingStrategy.LEAST_LOADED
        selected = balancer.select_worker(workers)
        assert selected.worker_id == 3  # Least loaded

    def test_latency_aware_selection(self, balancer, workers):
        workers[0].avg_latency_ms = 50.0
        workers[1].avg_latency_ms = 100.0
        workers[2].avg_latency_ms = 25.0
        balancer.strategy = LoadBalancingStrategy.LATENCY_AWARE
        selected = balancer.select_worker(workers)
        assert selected.worker_id == 3

    def test_rebalance(self, balancer, workers):
        result = balancer.rebalance(workers)
        assert result is not None or isinstance(result, dict)

    def test_get_load_metrics(self, balancer, workers):
        metrics = balancer.get_load_metrics()
        assert "strategy" in metrics

    def test_unhealthy_workers_excluded(self, balancer, workers):
        workers[0].is_healthy = False
        selected = balancer.select_worker(workers)
        assert selected.worker_id != 1

    def test_empty_worker_list(self, balancer):
        selected = balancer.select_worker([])
        assert selected is None

    def test_all_workers_unhealthy(self, balancer, workers):
        for w in workers:
            w.is_healthy = False
        selected = balancer.select_worker(workers)
        assert selected is None

    def test_single_worker(self, balancer):
        worker = Worker(1, cpu_usage=50.0)
        selected = balancer.select_worker([worker])
        assert selected.worker_id == 1

    def test_large_worker_pool(self, balancer):
        workers = [Worker(i, cpu_usage=float(i * 10)) for i in range(100)]
        selected = balancer.select_worker(workers)
        assert selected is not None

    def test_weighted_selection(self, balancer):
        workers = [Worker(1), Worker(2), Worker(3)]
        balancer.strategy = LoadBalancingStrategy.WEIGHTED
        selected = balancer.select_worker(workers)
        assert selected is not None

    def test_task_with_resource_constraints(self, balancer, workers):
        task = Task("t1", cpu_required=10.0, memory_required=5.0)
        selected = balancer.select_worker(workers, task)
        assert selected is not None

    def test_sequential_selections(self, balancer, workers):
        balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        sel1 = balancer.select_worker(workers)
        sel2 = balancer.select_worker(workers)
        assert isinstance(sel1, Worker)
        assert isinstance(sel2, Worker)

    def test_load_metrics_accumulation(self, balancer, workers):
        balancer.select_worker(workers)
        balancer.select_worker(workers)
        metrics = balancer.get_load_metrics()
        assert "total_assignments" in metrics

    def test_strategy_switching(self, balancer, workers):
        balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        sel1 = balancer.select_worker(workers)
        balancer.strategy = LoadBalancingStrategy.LEAST_LOADED
        sel2 = balancer.select_worker(workers)
        assert sel1 is not None
        assert sel2 is not None

    def test_worker_status_changes(self, balancer, workers):
        balancer.strategy = LoadBalancingStrategy.LEAST_LOADED
        selected = balancer.select_worker(workers)
        initial_id = selected.worker_id
        workers[initial_id - 1].is_healthy = False
        selected2 = balancer.select_worker(workers)
        assert selected2.worker_id != initial_id

    def test_performance_with_many_selections(self, balancer, workers):
        for _ in range(100):
            balancer.select_worker(workers)
        metrics = balancer.get_load_metrics()
        assert metrics is not None
