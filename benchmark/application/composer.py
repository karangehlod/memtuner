"""Benchmark composer — single composition root for all execution paths.

Both `benchmark run` (CLI) and the matrix worker call this service.
No other code should perform registry bootstrap, strategy resolution,
policy construction, or evaluator creation independently.

Responsibilities:
- Registry bootstrap (memory modules + retrieval strategies)
- Configuration validation
- Strategy resolution (fail-fast, no implicit fallback)
- Memory module instantiation
- Lifecycle policy construction
- Dataset loading and validation
- Evaluator construction with dataset-driven K
- Scenario creation with validated horizon
- RunPlan generation for auditability

This class knows HOW to wire things. The orchestrator knows WHEN to run them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmark.application.errors import (
    LifecyclePolicyError,
    StrategyResolutionError,
)
from benchmark.application.run_plan import (
    RunPlan,
    compute_dataset_fingerprint,
)
from benchmark.config.schema import BenchmarkConfig, PruningStrategy
from benchmark.cost.tracker import InMemoryCostTracker
from benchmark.evaluation.false_positive import FalsePositiveEvaluator
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator
from benchmark.factory.bootstrap import bootstrap_retrieval_strategies
from benchmark.factory.registry import MemoryModuleRegistry, RetrievalStrategyRegistry
from benchmark.factory.resolver import ConfigResolver
from benchmark.gold.oracle import GoldOracle
from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.memory.policies.pruning import (
    AgeBasedPruningPolicy,
    ScoreThresholdPruningPolicy,
)
from benchmark.memory.short_term.context_buffer import ContextBuffer
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer
from benchmark.memory.short_term.scratchpad import Scratchpad
from benchmark.observability.logger import get_logger, log_decision
from benchmark.orchestrator.benchmark_runner import BenchmarkRunner
from benchmark.scenario.loader import GoldDatasetScenario
from benchmark.time.simulated_clock import SimulatedClock

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


@dataclass
class ComposedBenchmark:
    """The fully-wired benchmark ready for execution.

    All dependencies are resolved and validated. Call `runner.run(scenarios)`
    to execute.
    """

    runner: BenchmarkRunner
    scenarios: list[GoldDatasetScenario]
    run_plan: RunPlan
    time_provider: SimulatedClock
    cost_tracker: InMemoryCostTracker


class BenchmarkComposer:
    """Single composition root for all benchmark execution paths.

    Usage:
        composer = BenchmarkComposer()
        composed = composer.compose(config, dataset_path=Path("data/gold.json"))
        result = composed.runner.run(composed.scenarios)
    """

    def __init__(self) -> None:
        """Initialize registries."""
        self._registry = MemoryModuleRegistry()
        self._strategy_registry = RetrievalStrategyRegistry()
        self._bootstrap_registries()

    def _bootstrap_registries(self) -> None:
        """Register all known memory modules and retrieval strategies."""
        self._registry.register("episodic_buffer", EpisodicBuffer)
        self._registry.register("context_buffer", ContextBuffer)
        self._registry.register("scratchpad", Scratchpad)
        self._registry.register("episodic_store", EpisodicStore)
        self._registry.register("preference_store", PreferenceStore)
        self._registry.register("semantic_store", SemanticStore)
        self._registry.register("entity_store", EntityStore)
        bootstrap_retrieval_strategies(self._strategy_registry)

    def compose(
        self,
        config: BenchmarkConfig,
        dataset_path: Path | None = None,
        dataset_override: Any | None = None,
        allow_strategy_fallback: bool = False,
        answer_evaluator: Any | None = None,
    ) -> ComposedBenchmark:
        """Compose a complete benchmark from configuration.

        Args:
            config: Validated benchmark configuration.
            dataset_path: Path to gold dataset JSON file.
            dataset_override: Pre-loaded GoldDataset (for packs/tests).
            allow_strategy_fallback: If True, fall back to default scoring
                on strategy failure. Default False (fail-fast).

        Returns:
            A ComposedBenchmark with all dependencies wired.

        Raises:
            StrategyResolutionError: If strategy cannot be resolved.
            LifecyclePolicyError: If policy construction fails.
            DatasetValidationError: If dataset is invalid.
        """
        resolver = ConfigResolver(self._registry, self._strategy_registry)

        # 1. Validate configuration against registry
        validation_errors = resolver.validate_config_against_registry(config)
        if validation_errors:
            from benchmark.application.errors import CompositionError

            raise CompositionError(
                f"Configuration validation failed: {'; '.join(validation_errors)}"
            )

        # 2. Resolve retrieval strategy (fail-fast)
        retrieval_strategy = self._resolve_strategy(
            config,
            config.benchmark.retrieval_strategy,
            resolver,
            allow_fallback=allow_strategy_fallback,
        )
        effective_strategy = (
            type(retrieval_strategy).__name__ if retrieval_strategy else "default_similarity"
        )

        # 3. Resolve memory modules
        logger.info(f"[TRACE] Composer calling resolver with allow_strategy_fallback={allow_strategy_fallback}")
        memory_modules = resolver.resolve_memory_modules(
            config,
            retrieval_strategy=retrieval_strategy,
            allow_strategy_fallback=allow_strategy_fallback,
        )

        # 4. Construct lifecycle policies
        lifecycle_policies = self._build_lifecycle_policies(config)

        # 5. Load dataset
        gold_oracle = GoldOracle()
        dataset = self._load_dataset(gold_oracle, dataset_path, dataset_override)

        # 6. Determine effective K from dataset
        recall_k = dataset.evaluation_criteria.recall_k

        # 7. Build evaluators with dataset-driven K
        evaluators = self._build_evaluators(recall_k)

        # 8. Create scenario with validated horizon
        effective_horizon = self._compute_effective_horizon(config, dataset)
        scenario = GoldDatasetScenario(dataset, evaluation_horizon=effective_horizon)

        # 9. Build runner
        time_provider = SimulatedClock()
        cost_tracker = InMemoryCostTracker()

        runner = BenchmarkRunner(
            time_provider=time_provider,
            gold_oracle=gold_oracle,
            memory_modules=memory_modules,
            evaluators=evaluators,
            cost_tracker=cost_tracker,
            config=config,
            lifecycle_policies=lifecycle_policies,
            answer_evaluator=answer_evaluator,
        )

        # 10. Build run plan for auditability
        normalization_meta = gold_oracle.get_normalization_metadata(dataset.scenario)
        run_plan = self._build_run_plan(
            config=config,
            dataset=dataset,
            requested_strategy=config.benchmark.retrieval_strategy,
            effective_strategy=effective_strategy,
            memory_modules=memory_modules,
            lifecycle_policies=lifecycle_policies,
            recall_k=recall_k,
            effective_horizon=effective_horizon,
            normalization_meta=normalization_meta,
        )

        log_decision(
            logger,
            "Benchmark composed successfully",
            strategy=effective_strategy,
            modules=list(memory_modules.keys()),
            policies=list(lifecycle_policies.keys()),
            recall_k=recall_k,
            horizon=effective_horizon,
            queries=dataset.queries.__len__() if dataset.queries else 0,
        )

        return ComposedBenchmark(
            runner=runner,
            scenarios=[scenario],
            run_plan=run_plan,
            time_provider=time_provider,
            cost_tracker=cost_tracker,
        )

    def _resolve_strategy(
        self,
        config: BenchmarkConfig,
        strategy_name: str,
        resolver: ConfigResolver,
        allow_fallback: bool = False,
    ) -> Any | None:
        """Resolve retrieval strategy with fail-fast semantics.

        Args:
            strategy_name: Name of the strategy to resolve.
            resolver: The config resolver.
            allow_fallback: If True, return None on failure instead of raising.

        Returns:
            The resolved strategy instance, or None if no strategy configured.

        Raises:
            StrategyResolutionError: If strategy fails and fallback is not allowed.
        """
        if not strategy_name:
            return None

        try:
            strategy = resolver.resolve_retrieval_strategy(config, strategy_name)
            log_decision(
                logger,
                "Strategy resolved",
                strategy=strategy_name,
                resolved_class=type(strategy).__name__,
            )
            return strategy
        except Exception as exc:
            if allow_fallback:
                log_decision(
                    logger,
                    "Strategy resolution failed, using fallback",
                    strategy=strategy_name,
                    error=str(exc),
                )
                return None

            raise StrategyResolutionError(
                strategy_name=strategy_name,
                reason=str(exc),
            ) from exc

    def _build_lifecycle_policies(self, config: BenchmarkConfig) -> dict[str, Any]:
        """Construct lifecycle policies for all configured modules.

        Args:
            config: The benchmark configuration.

        Returns:
            Dict mapping module_name → policy instance.

        Raises:
            LifecyclePolicyError: If a policy cannot be constructed.
        """
        policies: dict[str, Any] = {}

        for module_name in config.memory.enabled.long_term:
            policy_config = config.policies.module_policies.get(module_name)
            if not policy_config:
                continue

            try:
                pruning_strategy = policy_config.pruning.strategy
                pruning_threshold = policy_config.pruning.threshold

                if pruning_strategy == PruningStrategy.AGE_BASED:
                    policies[module_name] = AgeBasedPruningPolicy(threshold=pruning_threshold)
                elif pruning_strategy == PruningStrategy.SCORE_THRESHOLD:
                    policies[module_name] = ScoreThresholdPruningPolicy(threshold=pruning_threshold)
                else:
                    raise LifecyclePolicyError(
                        module_name=module_name,
                        reason=f"Unsupported pruning strategy: {pruning_strategy}",
                    )

                log_decision(
                    logger,
                    "Lifecycle policy constructed",
                    module=module_name,
                    strategy=pruning_strategy.value,
                    threshold=pruning_threshold,
                )

            except LifecyclePolicyError:
                raise
            except Exception as exc:
                raise LifecyclePolicyError(module_name=module_name, reason=str(exc)) from exc

        return policies

    def _load_dataset(
        self,
        gold_oracle: GoldOracle,
        dataset_path: Path | None,
        dataset_override: Any | None,
    ) -> Any:
        """Load dataset from path or use override, then validate.

        Args:
            gold_oracle: The gold oracle instance.
            dataset_path: Path to the dataset file.
            dataset_override: Pre-loaded dataset object.

        Returns:
            The loaded and validated GoldDataset.

        Raises:
            DatasetValidationError: If dataset fails integrity checks.
        """
        from benchmark.gold.validator import DatasetValidator

        if dataset_override is not None:
            DatasetValidator().validate(dataset_override)
            return dataset_override

        if dataset_path is None:
            from benchmark.application.errors import CompositionError

            raise CompositionError("No dataset provided. Pass dataset_path or dataset_override.")

        dataset = gold_oracle.load_dataset(dataset_path)
        DatasetValidator().validate(dataset)
        return dataset

    def _build_evaluators(self, recall_k: int) -> list[Any]:
        """Build evaluators using dataset-driven K.

        Args:
            recall_k: The K value from dataset evaluation criteria.

        Returns:
            List of evaluator instances.
        """
        from benchmark.evaluation.precision import StandardPrecisionEvaluator

        evaluators: list[Any] = [
            RecallEvaluator(top_k=recall_k),
            StandardPrecisionEvaluator(top_k=recall_k),
            FalsePositiveEvaluator(),
            TemporalAccuracyEvaluator(),
        ]

        # Add ranking evaluators — MRR, NDCG, Precision@1
        try:
            from benchmark.evaluation.ranking import (
                MRREvaluator, NDCGEvaluator, PrecisionAtKEvaluator,
            )

            evaluators.append(MRREvaluator(top_k=recall_k))
            evaluators.append(NDCGEvaluator(top_k=recall_k))
            # Precision@1: did the top result match the gold set?
            # Produces "benchmark.precision_at_1" consumed by ScenarioMetrics.
            evaluators.append(PrecisionAtKEvaluator(top_k=1))
        except ImportError:
            pass

        # Import optional evaluators that may not always be needed
        try:
            from benchmark.evaluation.followup_accuracy import (
                FollowUpAccuracyEvaluator,
            )

            evaluators.append(FollowUpAccuracyEvaluator())
        except ImportError:
            pass

        try:
            from benchmark.evaluation.contradiction_resolution import (
                ContradictionResolutionEvaluator,
            )

            evaluators.append(ContradictionResolutionEvaluator())
        except ImportError:
            pass

        return evaluators

    def _compute_effective_horizon(self, config: BenchmarkConfig, dataset: Any) -> int:
        """Compute effective horizon from config and dataset.

        If evaluation_horizon is configured, validate it covers the dataset.
        Otherwise, derive from the dataset's natural span.

        Args:
            config: The benchmark configuration.
            dataset: The loaded gold dataset.

        Returns:
            Effective number of dataset days to replay.
        """
        # Derive natural span from dataset
        all_days: set[int] = set()
        if dataset.events:
            for day_events in dataset.events:
                all_days.add(day_events.day)
        if dataset.queries:
            for query in dataset.queries:
                all_days.add(query.day)

        natural_span = max(all_days) + 1 if all_days else 1

        configured_days = config.benchmark.evaluation_horizon

        # Use natural span if it exceeds configured days
        # This prevents silently skipping queries
        effective = max(configured_days, natural_span)

        if effective > configured_days:
            log_decision(
                logger,
                "Horizon expanded to cover dataset",
                configured=configured_days,
                effective=effective,
                natural_span=natural_span,
            )

        return effective

    def _build_run_plan(
        self,
        config: BenchmarkConfig,
        dataset: Any,
        requested_strategy: str,
        effective_strategy: str,
        memory_modules: dict[str, Any],
        lifecycle_policies: dict[str, Any],
        recall_k: int,
        effective_horizon: int,
        normalization_meta: dict[str, Any],
    ) -> RunPlan:
        """Build the immutable run plan for auditability.

        Args:
            config: The benchmark configuration.
            dataset: The loaded gold dataset.
            requested_strategy: Strategy name from config.
            effective_strategy: Actual resolved class name.
            memory_modules: Resolved module map.
            lifecycle_policies: Constructed policy map.
            recall_k: Evaluation K.
            effective_horizon: Final horizon days.
            normalization_meta: Normalization metadata from oracle.

        Returns:
            Immutable RunPlan instance.
        """
        config_hash = hashlib.sha256(config.model_dump_json(indent=None).encode()).hexdigest()[:16]

        fingerprint = compute_dataset_fingerprint(
            scenario=dataset.scenario,
            query_count=len(dataset.queries) if dataset.queries else 0,
            memory_count=dataset.total_conversation_turns,
            user_ids=list(dataset.user_ids) if dataset.user_ids else [],
        )

        return RunPlan(
            requested_strategy=requested_strategy,
            effective_strategy=effective_strategy,
            resolved_strategy_class=effective_strategy,
            memory_modules=tuple(memory_modules.keys()),
            lifecycle_policies=tuple(lifecycle_policies.keys()),
            dataset_fingerprint=fingerprint,
            dataset_query_count=(len(dataset.queries) if dataset.queries else 0),
            dataset_memory_count=dataset.total_conversation_turns,
            dataset_user_count=(len(dataset.user_ids) if dataset.user_ids else 0),
            dataset_event_day_count=(len(dataset.events) if dataset.events else 0),
            recall_k=recall_k,
            requested_horizon=config.benchmark.evaluation_horizon,
            effective_horizon=effective_horizon,
            normalization_applied=normalization_meta.get("applied", False),
            normalization_delta_days=normalization_meta.get("delta_days", 0),
            config_hash=config_hash,
            seed=config.benchmark.seed,
        )
