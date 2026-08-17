"""Application-level errors for the composition layer.

These are fail-fast errors raised during benchmark setup,
before any scenario execution begins.
"""

from __future__ import annotations

from benchmark.exceptions.config_errors import BenchmarkError


class CompositionError(BenchmarkError):
    """Base error for application composition failures."""


class StrategyResolutionError(CompositionError):
    """Raised when a requested retrieval strategy cannot be resolved.

    This is a hard failure — no implicit fallback to default scoring.
    """

    # Maps strategy names to the pip extra required to install them
    STRATEGY_TO_EXTRA: dict[str, str] = {
        "bm25": "bm25",
        "embeddings": "embeddings",
        "api_embeddings": "api",
        "hybrid": "embeddings",
        "pgvector": "embeddings",
        "llm_rerank": "bm25",
        "llm": "llm",
        "database": "database",
    }

    def __init__(self, strategy_name: str, reason: str) -> None:
        self.strategy_name = strategy_name
        self.reason = reason
        extra = self.STRATEGY_TO_EXTRA.get(strategy_name, strategy_name)
        install_hint = f"pip install memtuner[{extra}]"
        super().__init__(
            f"Cannot resolve retrieval strategy '{strategy_name}': {reason}. "
            f"No implicit fallback is allowed. Fix the strategy name, install "
            f"the required dependency ({install_hint}), or remove the strategy "
            f"from config."
        )


class StrategyDependencyError(StrategyResolutionError):
    """Raised when a strategy's optional dependency is not installed."""

    def __init__(self, strategy_name: str, package: str) -> None:
        self.package = package
        super().__init__(
            strategy_name,
            f"Required package '{package}' is not installed. "
            f"Install with: pip install memtuner[{strategy_name}]",
        )


class LifecyclePolicyError(CompositionError):
    """Raised when a lifecycle policy cannot be constructed."""

    def __init__(self, module_name: str, reason: str) -> None:
        self.module_name = module_name
        super().__init__(f"Cannot construct lifecycle policy for module '{module_name}': {reason}")


class DatasetValidationError(CompositionError):
    """Raised when a gold dataset fails validation."""

    def __init__(self, dataset_path: str, errors: list[str]) -> None:
        self.dataset_path = dataset_path
        self.validation_errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(f"Dataset validation failed for '{dataset_path}':\n  - {error_list}")


class HorizonError(CompositionError):
    """Raised when the configured horizon is insufficient for the dataset."""

    def __init__(self, requested_days: int, required_days: int, reason: str) -> None:
        self.requested_days = requested_days
        self.required_days = required_days
        super().__init__(
            f"Horizon insufficient: requested {requested_days} days but dataset "
            f"requires {required_days} days. {reason}"
        )
