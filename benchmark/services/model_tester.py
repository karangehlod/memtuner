"""
Model tester for comparing different embedding and LLM models across providers.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from benchmark.services.provider_service import ProviderConfig, ProviderType
from benchmark.memory.strategies.api_embeddings_strategy import ApiEmbeddingsStrategy

logger = logging.getLogger(__name__)


@dataclass
class ModelTestResult:
    """Result from testing a single model"""
    provider: str
    model_name: str
    test_type: str  # "embedding" or "reranker"
    success: bool
    recall: Optional[float] = None
    precision: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    time_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelTester:
    """Test different models and providers"""

    @staticmethod
    def test_embedding_model(
        provider_config: ProviderConfig,
        test_texts: List[str],
        timeout: float = 120.0
    ) -> ModelTestResult:
        """
        Test an embedding model

        Args:
            provider_config: Provider configuration
            test_texts: Sample texts to embed
            timeout: API timeout

        Returns:
            ModelTestResult with success/error info
        """
        try:
            logger.info(
                f"Testing embedding model: "
                f"{provider_config.provider_type.value}/"
                f"{provider_config.model_name}"
            )

            strategy = ApiEmbeddingsStrategy(
                model_name=provider_config.model_name,
                base_url=provider_config.base_url,
                api_key=provider_config.api_key,
                timeout=timeout,
            )

            # Test embedding
            embeddings = strategy._embed_texts(test_texts)

            success = (
                embeddings is not None and
                len(embeddings) == len(test_texts) and
                len(embeddings[0]) > 0
            )

            result = ModelTestResult(
                provider=provider_config.provider_type.value,
                model_name=provider_config.model_name,
                test_type="embedding",
                success=success
            )

            logger.info(
                f"✓ Embedding model test passed: "
                f"{provider_config.provider_type.value}/"
                f"{provider_config.model_name}"
            )

            return result

        except Exception as e:
            logger.error(
                f"✗ Embedding model test failed: "
                f"{type(e).__name__}: {e}"
            )

            return ModelTestResult(
                provider=provider_config.provider_type.value,
                model_name=provider_config.model_name,
                test_type="embedding",
                success=False,
                error=str(e)
            )

    @staticmethod
    def test_embedding_models_batch(
        provider_configs: List[ProviderConfig],
        test_texts: List[str],
        timeout: float = 120.0
    ) -> List[ModelTestResult]:
        """
        Test multiple embedding models

        Args:
            provider_configs: List of provider configurations
            test_texts: Sample texts to embed
            timeout: API timeout

        Returns:
            List of ModelTestResult for each model
        """
        results = []

        for config in provider_configs:
            result = ModelTester.test_embedding_model(
                config,
                test_texts,
                timeout
            )
            results.append(result)

        return results

    @staticmethod
    def get_openai_embedding_models() -> List[ProviderConfig]:
        """Get OpenAI embedding models to test"""
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI models")
            return []

        return [
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                model_name="text-embedding-3-small"
            ),
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                model_name="text-embedding-3-large"
            ),
        ]

    @staticmethod
    def get_api_embedding_models() -> List[ProviderConfig]:
        """Get API embedding models configured via BENCHMARK_OPENAI_BASE_URL + BENCHMARK_API_EMBEDDING_MODELS."""
        import os

        base_url = os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
        models_raw = os.environ.get("BENCHMARK_API_EMBEDDING_MODELS", "")
        if not base_url or not models_raw:
            return []

        api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        return [
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                base_url=base_url,
                api_key=api_key,
                model_name=m.strip(),
            )
            for m in models_raw.split(",")
            if m.strip()
        ]

    @staticmethod
    def get_all_embedding_models() -> List[ProviderConfig]:
        """Get all embedding models to test (Ollama + OpenAI)"""
        models = []

        # Try Ollama models
        try:
            models.extend(ModelTester.get_ollama_embedding_models())
        except Exception as e:
            logger.warning(f"Could not load Ollama models: {e}")

        # Try OpenAI models
        models.extend(ModelTester.get_openai_embedding_models())

        return models
