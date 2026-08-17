"""Provider configuration service for strategy-specific settings.

Consolidates provider-specific configuration building from scattered
environment variable parsing and config overrides.
"""

import os
from typing import Any, Optional

from benchmark.config.schema import BenchmarkConfig
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


class ProviderConfigurationService:
    """Build provider configuration for retrieval strategies.

    Usage:
        service = ProviderConfigurationService(config)
        api_settings = service.build_api_embeddings_settings()
        overrides = service.build_strategy_overrides('api_embeddings')
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize with benchmark configuration.

        Args:
            config: The benchmark configuration containing provider settings.
        """
        self.config = config

    def build_api_embeddings_settings(self) -> Optional[dict[str, Any]]:
        """Build API embeddings provider settings.

        Returns:
            API embeddings config dict, or None if not configured.
        """
        api_config = self.config.benchmark.retrieval.api_embeddings
        base_url = api_config.base_url or os.environ.get("BENCHMARK_OPENAI_BASE_URL")
        if not base_url or not api_config.model_name:
            return None

        settings: dict[str, Any] = {
            "model_name": api_config.model_name,
            "base_url": base_url,
        }

        api_key = api_config.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            settings["api_key"] = api_key

        if api_config.timeout is not None:
            settings["timeout"] = api_config.timeout
        if api_config.batch_size is not None:
            settings["batch_size"] = api_config.batch_size

        return settings

    def build_reranker_settings(self) -> Optional[dict[str, Any]]:
        """Build reranker provider settings.

        Returns:
            Reranker provider config dict, or None if not configured.
        """
        reranker_config = self.config.benchmark.reranker
        if not reranker_config or not reranker_config.model_name:
            return None

        settings: dict[str, Any] = {
            "model_name": reranker_config.model_name,
        }
        return settings

    def build_strategy_overrides(
        self,
        strategy_name: str,
        embedding_candidates: Optional[list[tuple[str, str]]] = None,
    ) -> Optional[dict[str, Any]]:
        """Build strategy-specific config overrides.

        Args:
            strategy_name: The strategy to build overrides for.
            embedding_candidates: Available local embedding model candidates.

        Returns:
            Config override dict, or None if no overrides needed.
        """
        overrides: dict[str, Any] = {}

        if strategy_name == "embeddings":
            if embedding_candidates:
                first_model = embedding_candidates[0][0]
                overrides["retrieval"] = {"embeddings": {"model_name": first_model}}

        elif strategy_name == "api_embeddings":
            api_settings = self.build_api_embeddings_settings()
            if api_settings:
                overrides["retrieval"] = {"api_embeddings": api_settings}

        elif strategy_name == "hybrid":
            if embedding_candidates:
                first_model = embedding_candidates[0][0]
                overrides["retrieval"] = {
                    "embeddings": {"model_name": first_model},
                }
                overrides["hybrid"] = {
                    "strategies": ["bm25", "embeddings"],
                    "confidence_threshold": 0.5,
                    "bm25_weight": 0.5,
                }

        return overrides if overrides else None

    def validate_providers(self) -> list[str]:
        """Validate provider configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        api_settings = self.build_api_embeddings_settings()
        if self.config.benchmark.retrieval.api_embeddings.model_name and not api_settings:
            errors.append(
                "api_embeddings model_name is set but base_url is not configured "
                "(set benchmark.retrieval.api_embeddings.base_url or BENCHMARK_OPENAI_BASE_URL)"
            )
        return errors

    def log_configuration(self) -> None:
        """Log provider configuration status."""
        api_settings = self.build_api_embeddings_settings()
        reranker_settings = self.build_reranker_settings()

        if api_settings:
            logger.debug(
                f"API embeddings provider configured: {api_settings.get('model_name', 'unknown')}"
            )
        if reranker_settings:
            logger.debug(
                f"Reranker configured: {reranker_settings.get('model_name', 'unknown')}"
            )
