"""Strategy testing service for benchmark execution.

Extracts strategy comparison and testing logic from analyze_command.py
into a reusable, testable service. Enables testing individual strategies
in isolation.

This service reduces 69 lines of testing logic in analyze_command.py
by providing a clean, focused API.
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.application.composer import BenchmarkComposer
from benchmark.application.errors import CompositionError
from benchmark.config.loader import load_config_from_dict
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyTestResult:
    """Result of testing a single strategy."""

    strategy_name: str
    """The name of the strategy tested."""

    status: Literal["success", "failed", "skipped"]
    """The test outcome: success, failed, or skipped."""

    metrics: dict[str, float] | None = None
    """Performance metrics if successful (recall, precision, etc.)."""

    elapsed: float | None = None
    """Elapsed time in seconds if successful."""

    reason: str | None = None
    """Human-readable reason if failed or skipped."""

    composed: Any | None = None
    """Composed benchmark object (if successful) - for state capture."""

    config: dict[str, Any] | None = None
    """Configuration used (if successful) - for state capture."""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "strategy": self.strategy_name,
            "status": self.status,
        }
        if self.metrics:
            result.update(self.metrics)
        if self.elapsed is not None:
            result["elapsed"] = self.elapsed
        if self.reason:
            result["reason"] = self.reason
        return result


class StrategyTestingService:
    """Test and compare retrieval strategies.

    This service encapsulates the strategy testing logic, making it
    easy to test individual strategies or multiple strategies in sequence.

    Usage:
        service = StrategyTestingService(composer, logger)
        result = service.test_strategy('embeddings', config_overrides)
        if result.status == 'success':
            print(f"Recall: {result.metrics['recall']:.1%}")
    """

    def __init__(self, composer: BenchmarkComposer, logger_instance: Any = None) -> None:
        """Initialize strategy testing service.

        Args:
            composer: BenchmarkComposer for running benchmarks.
            logger_instance: Logger for output (uses default if None).
        """
        self.composer = composer
        self.logger = logger_instance or logger

    def test_strategy(
        self,
        strategy_name: str,
        base_config_dict: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
        gold_dataset: Any | None = None,
        judge_evaluator: Any | None = None,
    ) -> StrategyTestResult:
        """Test a single strategy with given configuration.

        Args:
            strategy_name: Name of the strategy to test.
            base_config_dict: Base configuration dictionary.
            config_overrides: Optional config overrides for this strategy.
            gold_dataset: Optional dataset override.
            judge_evaluator: Optional judge evaluator for LLM judging.

        Returns:
            StrategyTestResult with outcome and metrics (if successful).
        """
        try:
            # Build effective configuration
            effective_config_dict = dict(base_config_dict)
            if config_overrides:
                # Merge overrides recursively
                for key, value in config_overrides.items():
                    if key in effective_config_dict and isinstance(
                        effective_config_dict[key], dict
                    ):
                        effective_config_dict[key].update(value)
                    else:
                        effective_config_dict[key] = value

            # Load and validate configuration
            config = load_config_from_dict(effective_config_dict)

            # Compose benchmark
            composed = self.composer.compose(
                config=config,
                dataset_override=gold_dataset,
                answer_evaluator=judge_evaluator,
            )

            # Run benchmark and measure time
            start = time.monotonic()
            result = composed.runner.run(composed.scenarios)
            elapsed = time.monotonic() - start

            # Extract metrics from first scenario
            sr = result.scenario_results[0]
            metrics = {
                "recall": sr.recall_at_k,
                "precision": sr.precision_at_k,
                "mrr": sr.mrr,
                "ndcg": sr.ndcg,
                "contamination": sr.contamination_rate,
                "ms_per_query": elapsed * 1000 / max(sr.total_queries, 1),
            }

            # Add LLM judge score if available
            if hasattr(sr, "llm_judge_score") and sr.llm_judge_score is not None:
                metrics["llm_judge_score"] = sr.llm_judge_score

            self.logger.info(f"Strategy {strategy_name} test passed")
            return StrategyTestResult(
                strategy_name=strategy_name,
                status="success",
                metrics=metrics,
                elapsed=elapsed,
                composed=composed,
                config=config,
            )

        except CompositionError as e:
            self.logger.error(f"Strategy {strategy_name} composition failed: {e}")
            return StrategyTestResult(
                strategy_name=strategy_name,
                status="failed",
                reason=f"Composition error: {e!s}",
            )
        except Exception as e:
            self.logger.exception(f"Strategy {strategy_name} test failed")
            return StrategyTestResult(
                strategy_name=strategy_name,
                status="failed",
                reason=str(e),
            )

    def test_strategies(
        self,
        strategy_names: list[str],
        base_config_dict: dict[str, Any],
        config_overrides_per_strategy: dict[str, dict[str, Any]] | None = None,
        gold_dataset: Any | None = None,
        judge_evaluator: Any | None = None,
    ) -> list[StrategyTestResult]:
        """Test multiple strategies sequentially.

        Args:
            strategy_names: List of strategy names to test.
            base_config_dict: Base configuration dictionary.
            config_overrides_per_strategy: Per-strategy config overrides.
            gold_dataset: Optional dataset override.
            judge_evaluator: Optional judge evaluator.

        Returns:
            List of StrategyTestResult objects.
        """
        results = []

        for strategy_name in strategy_names:
            overrides = (
                (config_overrides_per_strategy or {}).get(strategy_name)
                if config_overrides_per_strategy
                else None
            )

            result = self.test_strategy(
                strategy_name,
                base_config_dict,
                overrides,
                gold_dataset,
                judge_evaluator,
            )
            results.append(result)

        return results

    def get_success_count(self, results: list[StrategyTestResult]) -> int:
        """Count successful tests.

        Args:
            results: List of test results.

        Returns:
            Number of successful tests.
        """
        return len([r for r in results if r.status == "success"])

    def get_failed_count(self, results: list[StrategyTestResult]) -> int:
        """Count failed tests.

        Args:
            results: List of test results.

        Returns:
            Number of failed tests.
        """
        return len([r for r in results if r.status == "failed"])

    def log_summary(self, results: list[StrategyTestResult]) -> None:
        """Log test results summary.

        Args:
            results: List of test results.
        """
        success = self.get_success_count(results)
        failed = self.get_failed_count(results)
        total = len(results)

        if success == total:
            self.logger.info(f"All {total} strategies passed ✓")
        else:
            self.logger.warning(
                f"Strategy tests: {success}/{total} passed, {failed} failed"
            )
