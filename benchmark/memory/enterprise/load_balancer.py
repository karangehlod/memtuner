"""Load balancing for distributed execution."""

import logging
from typing import Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    LATENCY_AWARE = "latency_aware"


@dataclass
class Worker:
    """Worker information."""
    worker_id: int
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tasks_assigned: int = 0
    avg_latency_ms: float = 0.0
    is_healthy: bool = True


@dataclass
class Task:
    """Task to be distributed."""
    task_id: str
    cpu_required: float = 1.0
    memory_required: float = 1.0
    priority: int = 0


class LoadBalancer:
    """Load balancer for distributing work."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED):
        """Initialize load balancer.

        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._round_robin_idx = 0
        self._worker_loads: Dict[int, float] = {}
        self._assignment_history: List[Dict[str, Any]] = []

    def select_worker(
        self,
        available_workers: List[Worker],
        task: Optional[Task] = None,
    ) -> Optional[Worker]:
        """Select worker for task.

        Args:
            available_workers: Available workers
            task: Task to assign

        Returns:
            Selected worker or None
        """
        if not available_workers:
            return None

        healthy_workers = [w for w in available_workers if w.is_healthy]

        if not healthy_workers:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_least_loaded(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
            return self._select_latency_aware(healthy_workers)
        else:
            return healthy_workers[0]

    def rebalance(
        self,
        workers: List[Worker],
    ) -> Dict[int, List[Task]]:
        """Rebalance work across workers.

        Args:
            workers: All workers

        Returns:
            Dict mapping worker ID to assigned tasks
        """
        rebalancing = {}

        for worker in workers:
            rebalancing[worker.worker_id] = []

        return rebalancing

    def get_load_metrics(self) -> Dict[str, Any]:
        """Get load metrics.

        Returns:
            Metrics dict
        """
        return {
            "strategy": self.strategy.value,
            "total_assignments": len(self._assignment_history),
            "worker_loads": self._worker_loads.copy(),
        }

    def _select_round_robin(self, workers: List[Worker]) -> Worker:
        """Select using round-robin."""
        worker = workers[self._round_robin_idx % len(workers)]
        self._round_robin_idx += 1
        return worker

    def _select_least_loaded(self, workers: List[Worker]) -> Worker:
        """Select least loaded worker."""
        return min(workers, key=lambda w: w.cpu_usage + w.memory_usage)

    def _select_latency_aware(self, workers: List[Worker]) -> Worker:
        """Select based on latency."""
        return min(workers, key=lambda w: w.avg_latency_ms)
