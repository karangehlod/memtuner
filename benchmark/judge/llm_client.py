"""LLM client — OpenAI-compatible API wrapper for any endpoint.

Works with:
- Ollama (http://localhost:11434/v1)
- vLLM (http://localhost:8000/v1)
- HuggingFace TGI (http://localhost:8080/v1)
- OpenAI (https://api.openai.com/v1)
- Any OpenAI-compatible server

Configuration via environment variables:
    BENCHMARK_LLM_BASE_URL: Base URL for the API
    BENCHMARK_LLM_API_KEY: API key (use "not-needed" for local)
    BENCHMARK_LLM_MODEL: Model name
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM client."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int = 500
    temperature: float = 0.0  # Deterministic for benchmarking


def get_llm_config() -> LLMConfig:
    """Load LLM config from environment variables.

    Returns:
        LLMConfig with values from env or defaults.
    """
    return LLMConfig(
        base_url=os.environ.get("BENCHMARK_LLM_BASE_URL")
              or os.environ.get("BENCHMARK_OPENAI_BASE_URL", ""),
        api_key=os.environ.get("BENCHMARK_LLM_API_KEY", "not-needed"),
        model=os.environ.get("BENCHMARK_LLM_MODEL", ""),
    )


def get_judge_config() -> LLMConfig:
    """Load judge-specific config from environment variables.

    Falls back to the LLM config if judge-specific vars not set.

    Returns:
        LLMConfig for the judge model.
    """
    llm_config = get_llm_config()
    return LLMConfig(
        base_url=os.environ.get("BENCHMARK_JUDGE_BASE_URL", llm_config.base_url),
        api_key=os.environ.get("BENCHMARK_JUDGE_API_KEY", llm_config.api_key),
        model=os.environ.get("BENCHMARK_JUDGE_MODEL", llm_config.model),
    )


class LLMClient:
    """Synchronous client for OpenAI-compatible chat completions API.

    Uses httpx for HTTP calls — no dependency on the openai package.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize with config. Defaults to env-based config.

        Args:
            config: LLM configuration. If None, reads from env.
        """
        self._config = config or get_llm_config()
        self._client = httpx.Client(
            base_url=self._config.base_url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: User message.
            system: System message (instructions for the model).

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If the API call fails.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LLM API error ({exc.response.status_code}): {exc.response.text[:200]}"
            ) from exc
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise RuntimeError(
                f"Cannot connect to LLM at {self._config.base_url}. "
                f"Is your LLM server running? Error: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable.

        Returns:
            True if the endpoint responds, False otherwise.
        """
        try:
            response = self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
