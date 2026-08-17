"""Config resolver — resolves benchmark config to wired implementations.

Translates a BenchmarkConfig into live, wired objects (memory modules, policies, etc.)
using the MemoryModuleRegistry.
"""

from __future__ import annotations

from typing import Any

from benchmark.config.schema import BenchmarkConfig, ModulePolicyConfig, RetrievalConfig
from benchmark.factory.registry import (
    MemoryModuleRegistry,
    RetrievalStrategyRegistry,
)


class ConfigResolver:
    """Resolves a BenchmarkConfig into instantiated implementations.

    Uses the MemoryModuleRegistry to map config names to concrete classes.
    Validates that all enabled modules are actually registered.
    """

    def __init__(
        self,
        registry: MemoryModuleRegistry,
        strategy_registry: RetrievalStrategyRegistry | None = None,
    ) -> None:
        """Initialize with populated registries.

        Args:
            registry: The memory module registry.
            strategy_registry: Optional registry for retrieval strategies.
        """
        self._registry = registry
        self._strategy_registry = strategy_registry

    def resolve_memory_modules(
        self,
        config: BenchmarkConfig,
        retrieval_strategy: object | None = None,
        allow_strategy_fallback: bool = False,
    ) -> dict[str, any]:
        """Resolve all enabled memory modules from config.

        Args:
            config: The benchmark configuration.
            retrieval_strategy: Optional retrieval strategy to pass to modules.
            allow_strategy_fallback: Whether to allow fallback to default scoring.

        Returns:
            Dictionary mapping module_name → instantiated module.

        Raises:
            RegistryResolutionError: If any enabled module is not registered.
        """
        from benchmark.observability.logger import get_logger
        logger = get_logger(__name__)

        resolved: dict[str, any] = {}
        all_enabled = config.memory.enabled.short_term + config.memory.enabled.long_term

        logger.info(f"[TRACE] Resolving memory modules: allow_strategy_fallback={allow_strategy_fallback}")

        for module_name in all_enabled:
            policy_config = config.policies.module_policies.get(module_name)
            constructor_kwargs = self._build_constructor_kwargs(policy_config)
            # Add retrieval strategy if provided
            if retrieval_strategy is not None:
                constructor_kwargs["retrieval_strategy"] = retrieval_strategy
            # Add strategy fallback flag
            constructor_kwargs["allow_strategy_fallback"] = allow_strategy_fallback
            logger.info(f"[TRACE] Creating {module_name} with allow_strategy_fallback={constructor_kwargs.get('allow_strategy_fallback')}")
            resolved[module_name] = self._registry.resolve(module_name, **constructor_kwargs)

        return resolved

    def validate_config_against_registry(self, config: BenchmarkConfig) -> list[str]:
        """Check that all enabled modules in config are registered.

        Args:
            config: The benchmark configuration.

        Returns:
            List of error messages for unregistered modules (empty if all valid).
        """
        errors: list[str] = []
        all_enabled = config.memory.enabled.short_term + config.memory.enabled.long_term

        for module_name in all_enabled:
            if not self._registry.is_registered(module_name):
                errors.append(
                    f"Memory module '{module_name}' is enabled in config "
                    f"but not registered. Available: {self._registry.registered_names()}"
                )

        return errors

    def _build_constructor_kwargs(
        self,
        policy_config: ModulePolicyConfig | None,
    ) -> dict[str, Any]:
        """Build constructor keyword arguments from policy config.

        Args:
            policy_config: Optional policy config for the module.

        Returns:
            Dictionary of constructor keyword arguments.
        """
        if policy_config is None:
            return {}

        return {
            "decay_type": policy_config.decay.type.value,
            "decay_lambda": policy_config.decay.lambda_factor,
            "decay_ranking_alpha": policy_config.decay.ranking_alpha,
            "archival_floor": policy_config.decay.archival_floor,
            "archival_day_threshold": policy_config.decay.archival_day_threshold,
            "tiered_working_days": policy_config.decay.tiered_working_days,
            "pruning_strategy": policy_config.pruning.strategy.value,
            "pruning_threshold": policy_config.pruning.threshold,
        }

    def resolve_retrieval_strategy(
        self,
        config: BenchmarkConfig,
        strategy_name: str,
    ) -> Any:
        """Resolve a retrieval strategy by name.

        Args:
            config: The validated benchmark configuration.
            strategy_name: Name of the strategy (e.g., "bm25", "embeddings").

        Returns:
            Instantiated strategy instance.

        Raises:
            RegistryResolutionError: If strategy not registered or instantiation fails.
        """
        if not self._strategy_registry:
            from benchmark.factory.registry import RetrievalStrategyRegistry

            self._strategy_registry = RetrievalStrategyRegistry()

        constructor_kwargs = self._build_retrieval_strategy_kwargs(
            strategy_name,
            config.benchmark.retrieval,
            config,
        )
        return self._strategy_registry.resolve(strategy_name, **constructor_kwargs)

    def _build_retrieval_strategy_kwargs(
        self,
        strategy_name: str,
        retrieval_config: RetrievalConfig,
        config: BenchmarkConfig | None = None,
    ) -> dict[str, Any]:
        """Build retrieval strategy constructor kwargs from validated config."""
        if strategy_name == "hybrid":
            hybrid_config = retrieval_config.hybrid
            kwargs: dict[str, Any] = {
                "strategies": hybrid_config.strategies,
                "confidence_threshold": hybrid_config.confidence_threshold,
                "bm25_weight": hybrid_config.bm25_weight,
            }

            for nested_strategy_name in hybrid_config.strategies:
                if nested_strategy_name == "bm25":
                    kwargs["bm25_strategy"] = self.strategy_registry().resolve("bm25")
                elif nested_strategy_name == "embeddings":
                    kwargs["embeddings_strategy"] = self.strategy_registry().resolve(
                        "embeddings",
                        **self._build_retrieval_strategy_kwargs("embeddings", retrieval_config, config),
                    )
                elif nested_strategy_name == "llm":
                    kwargs["llm_strategy"] = self.strategy_registry().resolve("llm")
            return kwargs

        if strategy_name == "embeddings":
            kwargs: dict[str, Any] = {}
            embeddings_config = retrieval_config.embeddings
            if embeddings_config.model_name:
                kwargs["model_name"] = embeddings_config.model_name
            if embeddings_config.cache_dir:
                kwargs["cache_dir"] = embeddings_config.cache_dir
            if embeddings_config.comparison_models:
                kwargs["comparison_models"] = embeddings_config.comparison_models
            return kwargs

        if strategy_name == "api_embeddings":
            api_config = retrieval_config.api_embeddings
            kwargs: dict[str, Any] = {}
            if api_config.model_name:
                kwargs["model_name"] = api_config.model_name
            if api_config.base_url:
                kwargs["base_url"] = api_config.base_url
            if api_config.api_key:
                kwargs["api_key"] = api_config.api_key
            kwargs["timeout"] = api_config.timeout
            kwargs["batch_size"] = api_config.batch_size
            return kwargs

        if strategy_name == "llm_rerank" and config is not None:
            reranker_config = config.benchmark.reranker
            kwargs = {
                "reranker_strategy": reranker_config.strategy,
                "model_name": reranker_config.model_name,
                "api_provider_order": reranker_config.api_provider_order,
                "local_size_threshold_mb": reranker_config.local_size_threshold_mb,
                "reranker_top_n": reranker_config.reranker_top_n,
            }
            return kwargs

        return {}

    def strategy_registry(self) -> RetrievalStrategyRegistry:
        """Get or create the strategy registry.

        Returns:
            The retrieval strategy registry instance.
        """
        if not self._strategy_registry:
            from benchmark.factory.registry import RetrievalStrategyRegistry

            self._strategy_registry = RetrievalStrategyRegistry()
        return self._strategy_registry
