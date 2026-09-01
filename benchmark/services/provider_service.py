"""
Provider service for handling different LLM/embedding providers.
Supports: Ollama, OpenAI, HuggingFace, Anthropic (extensible)
"""

import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"


class ProviderConfig:
    """Configuration for a provider"""

    def __init__(
        self,
        provider_type: ProviderType,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 120
    ):
        self.provider_type = provider_type
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider_type.value,
            "base_url": self.base_url,
            "model": self.model_name,
            "timeout": self.timeout
        }

    def __repr__(self) -> str:
        return f"{self.provider_type.value}://{self.model_name}"


class ProviderService:
    """Service to manage provider configurations"""

    @staticmethod
    def get_embedding_provider() -> ProviderConfig:
        """Get embedding provider configuration from environment"""
        provider_name = os.environ.get(
            "BENCHMARK_EMBEDDING_PROVIDER",
            "ollama"
        ).lower()

        provider_type = ProviderType(provider_name)

        if provider_type == ProviderType.OLLAMA:
            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_OLLAMA_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=os.environ.get(
                    "BENCHMARK_OLLAMA_API_KEY",
                    "not-needed"
                ),
                model_name=os.environ.get(
                    "BENCHMARK_EMBEDDING_MODEL_NAME",
                    "nomic-embed-text"
                ),
                timeout=int(os.environ.get(
                    "BENCHMARK_OLLAMA_TIMEOUT",
                    "120"
                ))
            )

        elif provider_type == ProviderType.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set"
                )

            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_OPENAI_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=api_key,
                model_name=os.environ.get(
                    "BENCHMARK_EMBEDDING_MODEL_NAME",
                    "text-embedding-3-small"
                ),
                timeout=120
            )

        elif provider_type == ProviderType.HUGGINGFACE:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable not set")

            return ProviderConfig(
                provider_type=provider_type,
                base_url="https://api-inference.huggingface.co",
                api_key=hf_token,
                model_name=os.environ.get(
                    "BENCHMARK_EMBEDDING_MODEL_NAME",
                    "BAAI/bge-base-en-v1.5"
                ),
                timeout=120
            )

        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

    @staticmethod
    def get_llm_provider() -> ProviderConfig:
        """Get LLM/reranker provider configuration from environment"""
        provider_name = os.environ.get(
            "BENCHMARK_LLM_PROVIDER",
            "ollama"
        ).lower()

        provider_type = ProviderType(provider_name)

        if provider_type == ProviderType.OLLAMA:
            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_LLM_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=os.environ.get(
                    "BENCHMARK_OLLAMA_API_KEY",
                    "not-needed"
                ),
                model_name=os.environ.get(
                    "BENCHMARK_LLM_MODEL_NAME",
                    ""
                ),
                timeout=int(os.environ.get(
                    "BENCHMARK_OLLAMA_TIMEOUT",
                    "120"
                ))
            )

        elif provider_type == ProviderType.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set"
                )

            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_OPENAI_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=api_key,
                model_name=os.environ.get(
                    "BENCHMARK_LLM_MODEL_NAME",
                    ""
                ),
                timeout=120
            )

        elif provider_type == ProviderType.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set"
                )

            return ProviderConfig(
                provider_type=provider_type,
                base_url="https://api.anthropic.com/v1",
                api_key=api_key,
                model_name=os.environ.get(
                    "BENCHMARK_LLM_MODEL_NAME",
                    ""
                ),
                timeout=120
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    @staticmethod
    def get_judge_provider() -> ProviderConfig:
        """Get judge/LLM provider configuration from environment"""
        provider_name = os.environ.get(
            "BENCHMARK_JUDGE_PROVIDER",
            "ollama"
        ).lower()

        provider_type = ProviderType(provider_name)

        if provider_type == ProviderType.OLLAMA:
            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_JUDGE_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=os.environ.get(
                    "BENCHMARK_OLLAMA_API_KEY",
                    "not-needed"
                ),
                model_name=os.environ.get(
                    "BENCHMARK_JUDGE_MODEL_NAME",
                    ""
                ),
                timeout=int(os.environ.get(
                    "BENCHMARK_OLLAMA_TIMEOUT",
                    "120"
                ))
            )

        elif provider_type == ProviderType.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set"
                )

            return ProviderConfig(
                provider_type=provider_type,
                base_url=os.environ.get(
                    "BENCHMARK_OPENAI_BASE_URL",
                    os.environ.get("BENCHMARK_OPENAI_BASE_URL", "")
                ),
                api_key=api_key,
                model_name=os.environ.get(
                    "BENCHMARK_JUDGE_MODEL_NAME",
                    ""
                ),
                timeout=120
            )

        elif provider_type == ProviderType.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set"
                )

            return ProviderConfig(
                provider_type=provider_type,
                base_url="https://api.anthropic.com/v1",
                api_key=api_key,
                model_name=os.environ.get(
                    "BENCHMARK_JUDGE_MODEL_NAME",
                    ""
                ),
                timeout=120
            )

        else:
            raise ValueError(f"Unsupported judge provider: {provider_name}")
