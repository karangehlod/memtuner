"""Benchmark runner — thin orchestrator that coordinates benchmark execution.

The runner knows WHEN things happen, not HOW.
All logic is delegated to injected dependencies.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

from benchmark.evaluation.aggregator import MetricAggregator
from benchmark.models.run_result import BenchmarkRunResult, CostSummary, ScenarioMetrics
from benchmark.observability.logger import get_logger, log_decision
from benchmark.observability.tracer import create_span
from benchmark.orchestrator.scenario_runner import ScenarioRunner

if TYPE_CHECKING:
    from benchmark.config.schema import BenchmarkConfig
    from benchmark.cost.tracker import CostTracker
    from benchmark.evaluation.base import MetricEvaluator
    from benchmark.gold.oracle import GoldOracle
    from benchmark.scenario.base import BenchmarkScenario
    from benchmark.time.provider import TimeProvider

logger = get_logger(__name__)


class BenchmarkRunner:
    """Thin coordinator for benchmark execution.

    Delegates all computation to injected dependencies:
    - TimeProvider for simulated clock
    - GoldOracle for expected results
    - MemoryReader/Writer for memory operations
    - MetricEvaluator for scoring
    - CostTracker for cost accounting
    """

    def __init__(
        self,
        time_provider: TimeProvider,
        gold_oracle: GoldOracle,
        memory_modules: dict[str, Any],
        evaluators: list[MetricEvaluator],
        cost_tracker: CostTracker,
        config: BenchmarkConfig,
        lifecycle_policies: dict[str, Any] | None = None,
        answer_evaluator: Any | None = None,
    ) -> None:
        """Initialize the benchmark runner with all dependencies.

        Args:
            time_provider: Simulated clock.
            gold_oracle: Gold truth repository.
            memory_modules: Resolved memory modules (name → instance).
            evaluators: List of metric evaluators.
            cost_tracker: Cost tracking service.
            config: The benchmark configuration.
            lifecycle_policies: Optional per-module lifecycle policies.
        """
        self._time_provider = time_provider
        self._gold_oracle = gold_oracle
        self._memory_modules = memory_modules
        self._evaluators = evaluators
        self._cost_tracker = cost_tracker
        self._config = config
        self._aggregator = MetricAggregator()
        self._scenario_runner = ScenarioRunner(
            time_provider=time_provider,
            gold_oracle=gold_oracle,
            memory_modules=memory_modules,
            evaluators=evaluators,
            cost_tracker=cost_tracker,
            lifecycle_policies=lifecycle_policies,
            answer_evaluator=answer_evaluator,
        )

    def run(self, scenarios: list[BenchmarkScenario]) -> BenchmarkRunResult:
        """Execute a complete benchmark run across all scenarios.

        Args:
            scenarios: List of scenarios to execute.

        Returns:
            Complete BenchmarkRunResult with all metrics.
        """
        run_id = self._generate_run_id()
        config_hash = self._compute_config_hash()

        with create_span(
            "benchmark.run",
            attributes={
                "run_id": run_id,
                "config_hash": config_hash,
                "scenario_count": len(scenarios),
            },
        ):
            log_decision(
                logger,
                "Starting benchmark run",
                run_id=run_id,
                scenarios=[s.name() for s in scenarios],
            )

            started_at = self._time_provider.current_timestamp()
            scenario_results: list[ScenarioMetrics] = []

            for scenario in scenarios:
                result = self._scenario_runner.run_scenario(scenario, run_id)
                scenario_results.append(result)
                self._time_provider.reset()

            completed_at = self._time_provider.current_timestamp()
            cost_summary = self._build_cost_summary(scenario_results)
            aggregates = self._compute_aggregates(scenario_results)

            return BenchmarkRunResult(
                run_id=run_id,
                config_hash=config_hash,
                started_at=started_at,
                completed_at=completed_at,
                seed=self._config.benchmark.seed,
                memory_modules_enabled=list(self._memory_modules.keys()),
                scenario_results=scenario_results,
                cost_summary=cost_summary,
                **aggregates,
            )

    def _generate_run_id(self) -> str:
        """Generate a unique run identifier.

        Returns:
            Short UUID string.
        """
        return uuid.uuid4().hex[:12]

    def _compute_config_hash(self) -> str:
        """Compute a hash of the config for reproducibility tracking.

        Returns:
            SHA-256 hash of the serialized config.
        """
        config_json = self._config.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def _build_cost_summary(self, scenario_results: list[ScenarioMetrics]) -> CostSummary:
        """Build cost summary from tracker data and scenario results.

        Args:
            scenario_results: Per-scenario metric results.

        Returns:
            CostSummary with computed costs.
        """
        total_cost = self._cost_tracker.total_cost_usd()
        total_correct = sum(sr.correct_recalls for sr in scenario_results)
        cost_per_recall = total_cost / max(total_correct, 1)

        return CostSummary(
            total_cost=total_cost,
            cost_per_correct_recall=cost_per_recall,
        )

    def _compute_aggregates(
        self,
        scenario_results: list[ScenarioMetrics],
    ) -> dict[str, float]:
        """Compute weighted aggregate metrics across scenarios.

        Args:
            scenario_results: Per-scenario results.

        Returns:
            Dict with aggregate_recall_at_k, aggregate_temporal_accuracy,
            aggregate_contamination_rate.
        """
        if not scenario_results:
            return {
                "aggregate_recall_at_k": 0.0,
                "aggregate_temporal_accuracy": 0.0,
                "aggregate_contamination_rate": 0.0,
            }

        total_queries = sum(sr.total_queries for sr in scenario_results)
        if total_queries == 0:
            return {
                "aggregate_recall_at_k": 0.0,
                "aggregate_temporal_accuracy": 0.0,
                "aggregate_contamination_rate": 0.0,
            }

        weighted_recall = (
            sum(sr.recall_at_k * sr.total_queries for sr in scenario_results) / total_queries
        )
        weighted_temporal = (
            sum(sr.temporal_accuracy * sr.total_queries for sr in scenario_results) / total_queries
        )
        weighted_contamination = (
            sum(sr.contamination_rate * sr.total_queries for sr in scenario_results) / total_queries
        )

        return {
            "aggregate_recall_at_k": weighted_recall,
            "aggregate_temporal_accuracy": weighted_temporal,
            "aggregate_contamination_rate": weighted_contamination,
        }
