"""LLM-powered retrieval strategy using OpenAI or Claude.

Uses an LLM to understand queries deeply and select best matches.
Highest accuracy but highest cost and latency.

Latency: 500-2000ms | Cost: High | Accuracy: Best | Setup: 45 min
"""

import os
import time

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    OpenAI = None
    RateLimitError = None

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent


class LLMStrategy(RetrievalStrategy):
    """LLM-powered retrieval strategy."""

    def __init__(
        self,
        model: str = "",
        api_key: str | None = None,
    ) -> None:
        """Initialize LLM strategy.

        Args:
            model: Model to use (e.g., "gpt-4o-mini", "claude-3-haiku").
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        """
        if OpenAI is None:
            raise ImportError("openai not installed. Install: pip install openai")

        if not model:
            raise ValueError(
                "LLMStrategy requires a model name. "
                "Set BENCHMARK_LLM_MODEL in .env or pass model= explicitly."
            )
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set. Set via env var or pass api_key param.")

        # Initialize OpenAI client
        try:
            self._client = OpenAI(api_key=self._api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        self._memories: dict[str, MemoryEvent] = {}

    def index(self, memories: list[MemoryEvent]) -> None:
        """Store memories for retrieval.

        Args:
            memories: List of memories to store.
        """
        self._memories = {mem.id: mem for mem in memories}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve using LLM ranking with exponential backoff on rate limits.

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, score) tuples.

        Raises:
            RuntimeError: If query fails after 3 rate limit retries or other errors.
        """
        if not self._memories:
            return []

        # Filter by user
        candidates = self._memories
        if user_id:
            candidates = {mid: mem for mid, mem in self._memories.items() if mem.user_id == user_id}

        if not candidates:
            return []

        # Limit to first 20 to avoid context explosion and cost
        candidate_list = list(candidates.items())[:20]

        # Format for LLM
        memory_text = "\n\n".join([f"[{mid}] {mem.content}" for mid, mem in candidate_list])

        prompt = f"""Given this query: "{query}"

Select the TOP {top_k} most relevant memory IDs from the list below.
Return ONLY the memory IDs, one per line, in order of relevance (best first).
Do not include brackets or explanations.

Memories:
{memory_text}"""

        # Retry with exponential backoff on rate limits
        max_retries = 3
        base_wait = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response
                response_text = response.content[0].text if response.content else ""
                retrieved_ids = []

                for line in response_text.split("\n"):
                    line = line.strip()
                    # Try to extract memory ID
                    if line.startswith("M-"):
                        retrieved_ids.append(line)
                    elif line.startswith("[M-"):
                        retrieved_ids.append(line.replace("[", "").replace("]", ""))

                # Score by rank
                scored = [
                    (mem_id, float(1.0 - i / len(retrieved_ids)))
                    for i, mem_id in enumerate(retrieved_ids)
                    if mem_id in self._memories
                ]

                return scored[:top_k]

            except RateLimitError as e:
                if attempt < max_retries:
                    wait_time = base_wait * (2**attempt)  # Exponential backoff
                    print(
                        f"Rate limited on attempt {attempt + 1}/{max_retries + 1}. Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"LLM rate limited after {max_retries} retries. Query failed: {e}"
                    ) from e

            except Exception as e:
                # Non-rate-limit errors should be raised, not silently handled
                raise RuntimeError(f"LLM retrieval failed (attempt {attempt + 1}): {e}") from e

        # Should not reach here
        raise RuntimeError("LLM retrieval exhausted all retry attempts")

    def name(self) -> str:
        """Return strategy name."""
        return "llm"

    def clear(self) -> None:
        """Clear all stored memories."""
        self._memories.clear()

    @classmethod
    def is_available(cls) -> bool:
        """Check if openai is installed."""
        return OpenAI is not None
