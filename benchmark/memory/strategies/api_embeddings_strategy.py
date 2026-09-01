"""API-based embedding retrieval strategy.

Uses any OpenAI-compatible embeddings endpoint: OpenAI, Ollama, vLLM, HF TEI, etc.
All three previously-separate strategies (ollama_embeddings, hf_inference_embeddings,
openai_embeddings) collapse into this one — the only difference is base_url + api_key,
which come from environment variables or explicit config.

Environment variables (read if constructor args are None):
    BENCHMARK_OPENAI_BASE_URL   e.g. https://api.openai.com/v1
                                     http://localhost:11434/v1  (Ollama)
                                     https://router.huggingface.co/v1  (HF TEI)
    OPENAI_API_KEY              API key (Ollama accepts any non-empty string)
"""

from __future__ import annotations

import os

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_TIMEOUT = 60.0


class ApiEmbeddingsStrategy(RetrievalStrategy):
    """Embedding retrieval via any OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai not installed. Install: pip install 'openai>=1.0,<2.0'")

        resolved_base_url = (
            base_url
            or os.environ.get("BENCHMARK_OPENAI_BASE_URL")
        )
        if not resolved_base_url:
            raise RuntimeError(
                "api_embeddings requires a base_url. "
                "Set BENCHMARK_OPENAI_BASE_URL in .env "
                "(e.g. https://api.openai.com/v1 or http://localhost:11434/v1 for Ollama)."
            )
        resolved_api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or "not-needed"
        )
        if not model_name:
            raise RuntimeError(
                "api_embeddings requires a model_name. "
                "Set it in the YAML study config under embedding.api_models."
            )

        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._client = OpenAI(
            base_url=resolved_base_url.rstrip("/"),
            api_key=resolved_api_key,
            timeout=timeout,
            max_retries=3,
        )

        self._memories: dict[str, MemoryEvent] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_ids: list[str] = []
        self._norms: np.ndarray | None = None
        self._query_cache: dict[str, np.ndarray] = {}
        self._user_mask_cache: dict[str, np.ndarray] = {}

    def name(self) -> str:
        return "api_embeddings"

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Call the API in batches and return one ndarray per text."""
        if not texts:
            return []
        results: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model_name, input=batch)
            sorted_data = sorted(response.data, key=lambda item: item.index)
            results.extend(np.array(item.embedding, dtype=np.float32) for item in sorted_data)
        return results

    def index(self, memories: list[MemoryEvent]) -> None:
        incoming = {mem.id: mem for mem in memories}
        new_ids = set(incoming) - set(self._memories)
        removed_ids = set(self._memories) - set(incoming)

        if new_ids or removed_ids:
            self._user_mask_cache = {}

        # Incremental path: only embed newly added memories
        if self._embedding_matrix is not None and new_ids and not removed_ids:
            new_mems = [incoming[mid] for mid in sorted(new_ids)]
            new_vecs = self._embed_texts([m.content for m in new_mems])
            if len(new_vecs) != len(new_mems):
                raise RuntimeError("Embedding count did not match input count")
            self._embedding_matrix = np.vstack(
                [self._embedding_matrix, np.array(new_vecs, dtype=np.float32)]
            )
            for mem, vec in zip(new_mems, new_vecs):
                self._embeddings[mem.id] = vec
                self._embedding_ids.append(mem.id)
            self._norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._norms = np.maximum(self._norms, 1e-8)
            self._memories = incoming
            return

        # Full rebuild
        self._memories = incoming
        self._embeddings = {}
        if not memories:
            self._embedding_matrix = None
            self._embedding_ids = []
            self._norms = None
            return

        ids = [mem.id for mem in memories]
        vecs = self._embed_texts([mem.content for mem in memories])
        if len(vecs) != len(ids):
            raise RuntimeError("Embedding count did not match input count")

        for mem_id, vec in zip(ids, vecs):
            self._embeddings[mem_id] = vec

        self._embedding_matrix = np.array(vecs, dtype=np.float32)
        self._embedding_ids = ids
        self._norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        self._norms = np.maximum(self._norms, 1e-8)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if not self._memories or self._embedding_matrix is None:
            return []

        if query in self._query_cache:
            query_vec = self._query_cache[query].reshape(1, -1)
        else:
            q_vecs = self._embed_texts([query])
            if not q_vecs:
                return []
            query_vec = q_vecs[0].reshape(1, -1)
            self._query_cache[query] = q_vecs[0]

        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-8:
            return []
        query_unit = query_vec / q_norm

        scores = (self._embedding_matrix @ query_unit.T).flatten() / self._norms.flatten()

        if user_id:
            if user_id not in self._user_mask_cache:
                self._user_mask_cache[user_id] = np.array(
                    [self._memories[mid].user_id == user_id for mid in self._embedding_ids],
                    dtype=bool,
                )
            scores = np.where(self._user_mask_cache[user_id], scores, -1.0)

        ids = self._embedding_ids
        actual_k = min(top_k, len(ids))
        top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(ids[i], float(scores[i])) for i in top_indices if scores[i] > -1.0]

    def clear(self) -> None:
        self._memories = {}
        self._embeddings = {}
        self._embedding_matrix = None
        self._embedding_ids = []
        self._norms = None
        self._query_cache = {}
        self._user_mask_cache = {}
