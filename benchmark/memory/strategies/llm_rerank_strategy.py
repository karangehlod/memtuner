"""LLM-Rerank retrieval strategy.

Two-stage retrieval: BM25 fetches all candidates, then a cross-encoder
reranker scores each (query, memory) pair and re-orders the results.

Provider priority (resolved automatically):
  1. local_crossencoder — sentence-transformers CrossEncoder loaded directly
     onto the GPU. Zero HTTP overhead, both embedding and reranker models
     reside in VRAM simultaneously (all benchmark models fit in 16GB together).
  2. API providers (hf_inference, ollama) — used only if sentence-transformers
     is unavailable or model_name is not a known cross-encoder.
  3. local_overlap — character n-gram Jaccard fallback (no model required).

VRAM usage when embedding + reranker run together (A4000 16GB):
  bge-m3 (embed, 1134MB) + bge-reranker-base (220MB) + activations = ~1685MB
  That leaves ~13.8GB free — no contention, no model swapping needed.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import httpx

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _CROSSENCODER_AVAILABLE = True
except ImportError:
    _CrossEncoder = None
    _CROSSENCODER_AVAILABLE = False

try:
    from benchmark.resources.hw_probe import DEVICE as _GPU_DEVICE
    _CUDA_AVAILABLE = (_GPU_DEVICE == "cuda")
except Exception:
    _GPU_DEVICE = "cpu"
    _CUDA_AVAILABLE = False

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.memory.strategies.bm25_strategy import BM25Strategy
from benchmark.models.memory_event import MemoryEvent

# CrossEncoder model cache — same pattern as EmbeddingsStrategy model cache.
# Avoids reloading weights for every cell.
_CE_MODEL_CACHE: dict[str, _CrossEncoder] = {}

# CrossEncoder pairwise scoring hangs on both MPS (Apple Silicon) and CPU with
# Python 3.13 / macOS ARM — the same class of issues affecting loky subprocesses.
# SentenceTransformer embeddings (single-encoder) work fine on MPS; CrossEncoder
# (dual-input attention) does not complete predict() calls on this platform.
# CUDA (NVIDIA GPU) is required for neural CrossEncoder reranking.
_CE_AVAILABLE = _CUDA_AVAILABLE   # only CUDA supports CrossEncoder on this codebase


def _get_crossencoder(model_name: str) -> _CrossEncoder:
    if not _CE_AVAILABLE:
        raise RuntimeError(
            f"CrossEncoder reranking requires CUDA. "
            f"Current device: {_GPU_DEVICE}. "
            f"On MPS/CPU, CrossEncoder predict() hangs — use reranker_model='none' "
            f"or run on a machine with an NVIDIA GPU."
        )
    if model_name not in _CE_MODEL_CACHE:
        _CE_MODEL_CACHE[model_name] = _CrossEncoder(model_name, device=_GPU_DEVICE)
    return _CE_MODEL_CACHE[model_name]


def _ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of character n-grams between two strings."""
    def ngrams(s: str) -> set:
        s = s.lower()
        return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}
    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class LLMRerankStrategy(RetrievalStrategy):
    """Two-stage retrieval: BM25 full-corpus fetch → CrossEncoder reranking.

    Stage 1: BM25 retrieves ALL memories (full corpus, not a top-N subset).
             Fetching only top-200 previously meant gold memories outside that
             window could never be recovered by the reranker.
    Stage 2: CrossEncoder scores every (query, memory) pair directly on GPU.
             No HTTP calls, no model swapping, no Ollama required.
    """

    # Default: 0 = fetch full corpus (resolved to len(memories) in retrieve()).
    # Override via reranker_top_n constructor argument to cap candidate depth.
    _DEFAULT_BM25_FETCH_K: int = 0

    # Known cross-encoder models supported by sentence-transformers CrossEncoder.
    # These run directly on GPU — no API, no Ollama, no HTTP overhead.
    _LOCAL_CROSSENCODER_MODELS = frozenset({
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L12-v2",
        "cross-encoder/ms-marco-electra-base",
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-large",
        "BAAI/bge-reranker-v2-m3",
        "jinaai/jina-reranker-v2-base-multilingual",
    })

    _RERANK_SIZE_MB = {
        "cross-encoder/ms-marco-MiniLM-L6-v2": 90,
        "cross-encoder/ms-marco-MiniLM-L12-v2": 130,
        "BAAI/bge-reranker-base": 210,
        "BAAI/bge-reranker-large": 1300,
        "BAAI/bge-reranker-v2-m3": 2200,
    }
    _MAX_RETRIES = 3
    _BASE_RETRY_DELAY_SECONDS = 1.0

    def __init__(
        self,
        reranker_strategy: str = "local_overlap",
        model_name: str | None = None,
        api_provider_order: list[str] | None = None,
        local_size_threshold_mb: int = 100,
        reranker_top_n: int = 0,
        hf_base_url: str | None = None,
        hf_api_token: str | None = None,
        hf_timeout: float | None = None,
        hf_endpoint_url: str | None = None,
        ollama_base_url: str | None = None,
        ollama_api_key: str | None = None,
        ollama_timeout: float | None = None,
    ) -> None:
        self._bm25 = BM25Strategy()
        # reranker_top_n=0 → cap at 100 (standard two-stage depth); N>0 → use N.
        # Scoring the full corpus (5879 pairs × 1977 queries = 11.6M pairs) would
        # take hours on GPU. CrossEncoder reranking is only useful on a short-list.
        self._bm25_fetch_k: int = reranker_top_n if reranker_top_n > 0 else 100
        self._memories: dict[str, MemoryEvent] = {}
        self._reranker_strategy = reranker_strategy
        self._model_name = model_name
        self._api_provider_order = list(api_provider_order or [])
        self._local_size_threshold_mb = local_size_threshold_mb
        self._hf_base_url = hf_base_url
        self._hf_api_token = hf_api_token
        self._hf_timeout = hf_timeout if hf_timeout is not None else 60.0
        self._hf_endpoint_url = hf_endpoint_url
        self._ollama_base_url = ollama_base_url
        self._ollama_api_key = ollama_api_key
        self._ollama_timeout = ollama_timeout if ollama_timeout is not None else 60.0

        self._provider = self._resolve_provider()
        self._client = self._build_client()

    @classmethod
    def is_available(cls) -> bool:
        """Always available — depends only on rank_bm25."""
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
            return True
        except ImportError:
            return False

    def index(self, memories: list[MemoryEvent]) -> None:
        self._bm25.index(memories)
        self._memories = {mem.id: mem for mem in memories}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 fetch followed by n-gram rerank.

        Args:
            query: Query text.
            top_k: Final result count.
            user_id: Optional user filter passed through to BM25.

        Returns:
            List of (memory_id, combined_score) sorted descending.
        """
        # Stage 1 — BM25 candidate fetch.
        # _bm25_fetch_k=0 means "all"; any positive value caps candidate depth.
        fetch_k = len(self._memories) if self._bm25_fetch_k == 0 else self._bm25_fetch_k
        candidates = self._bm25.retrieve(query, top_k=max(fetch_k, top_k), user_id=user_id)
        if not candidates:
            return []

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(score for _, score in candidates) or 1.0
        normalised = [(mid, score / max_bm25) for mid, score in candidates]

        if self._provider == "local_crossencoder":
            reranked = self._crossencoder_rerank(query, normalised)
        elif self._provider == "local_overlap":
            reranked = self._local_rerank(query, normalised)
        else:
            reranked = self._api_rerank(query, normalised)

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def name(self) -> str:
        return "llm_rerank"

    def clear(self) -> None:
        self._bm25.clear()
        self._memories.clear()

    def _crossencoder_rerank(
        self,
        query: str,
        normalised: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Rerank using a CrossEncoder model loaded directly on GPU.

        Processes all (query, memory_content) pairs in one batched predict()
        call — no HTTP, no Ollama, both the embedding model and this reranker
        stay resident in VRAM simultaneously (total <2GB on A4000).
        """
        model = _get_crossencoder(self._model_name)  # type: ignore[arg-type]
        memory_ids = []
        pairs = []
        bm25_scores: dict[str, float] = {}
        for memory_id, bm25_norm in normalised:
            event = self._memories.get(memory_id)
            if event is None:
                continue
            memory_ids.append(memory_id)
            pairs.append([query, event.content])
            bm25_scores[memory_id] = bm25_norm

        if not pairs:
            return []

        # predict() handles internal batching; returns a numpy array of logits
        batch_size = int(os.environ.get("BENCHMARK_RERANKER_BATCH_SIZE", "512"))
        try:
            scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                with contextlib.suppress(Exception):
                    _torch.cuda.empty_cache()
                scores = model.predict(pairs, batch_size=max(1, batch_size // 2),
                                       show_progress_bar=False)
            else:
                raise

        # Normalize CrossEncoder logits to [0, 1] via sigmoid, fuse with BM25
        reranked = []
        for i, memory_id in enumerate(memory_ids):
            raw = float(scores[i])
            ce_score = 1.0 / (1.0 + math.exp(-raw))  # sigmoid
            combined = 0.3 * bm25_scores[memory_id] + 0.7 * ce_score
            reranked.append((memory_id, combined))
        return reranked

    def _local_rerank(
        self,
        query: str,
        normalised: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        reranked: list[tuple[str, float]] = []
        for memory_id, bm25_norm in normalised:
            event = self._memories.get(memory_id)
            if event is None:
                continue
            overlap = _ngram_overlap(query, event.content, n=3)
            combined = 0.7 * bm25_norm + 0.3 * overlap
            reranked.append((memory_id, combined))
        return reranked

    def _api_rerank(
        self,
        query: str,
        normalised: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        if httpx is None or self._client is None or not self._model_name:
            raise RuntimeError("Provider-backed reranking requires configured httpx client and model")

        documents = []
        document_ids = []
        base_scores: dict[str, float] = {}
        for memory_id, bm25_norm in normalised:
            event = self._memories.get(memory_id)
            if event is None:
                continue
            documents.append(event.content)
            document_ids.append(memory_id)
            base_scores[memory_id] = bm25_norm

        if not documents:
            return []

        provider_scores = self._request_rerank_scores(query, documents)
        reranked: list[tuple[str, float]] = []
        for index, memory_id in enumerate(document_ids):
            provider_score = provider_scores[index] if index < len(provider_scores) else 0.0
            combined = 0.65 * base_scores[memory_id] + 0.35 * provider_score
            reranked.append((memory_id, combined))
        return reranked

    def _resolve_provider(self) -> str:
        # Explicit n-gram or no model → n-gram
        if self._reranker_strategy == "local_overlap" or not self._model_name:
            return "local_overlap"

        # Priority 1: direct CrossEncoder on GPU (CUDA only).
        # MPS and CPU hang on CrossEncoder predict() on macOS ARM / Python 3.13.
        if (_CROSSENCODER_AVAILABLE
                and _CE_AVAILABLE
                and self._model_name in self._LOCAL_CROSSENCODER_MODELS):
            return "local_crossencoder"

        # Priority 2: API providers (for models not in sentence-transformers)
        for provider in self._api_provider_order:
            if provider == "hf_inference" and (self._hf_endpoint_url or self._hf_base_url):
                return provider
            if provider == "ollama" and self._ollama_base_url:
                return provider

        # Fallback: n-gram (no real reranking)
        return "local_overlap"

    def _build_client(self):
        if httpx is None:
            return None
        if self._provider == "hf_inference":
            headers = {"Content-Type": "application/json"}
            if self._hf_api_token:
                headers["Authorization"] = f"Bearer {self._hf_api_token}"
            if self._hf_endpoint_url:
                return httpx.Client(headers=headers, timeout=self._hf_timeout)
            return httpx.Client(base_url=self._hf_base_url, headers=headers, timeout=self._hf_timeout)
        if self._provider == "ollama":
            headers = {"Content-Type": "application/json"}
            if self._ollama_api_key:
                headers["Authorization"] = f"Bearer {self._ollama_api_key}"
            return httpx.Client(
                base_url=self._ollama_base_url,
                headers=headers,
                timeout=self._ollama_timeout,
                trust_env=False,
            )
        return None

    def _request_rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        if self._provider == "hf_inference":
            return self._request_hf_rerank_scores(query, documents)
        if self._provider == "ollama":
            return self._request_ollama_rerank_scores(query, documents)
        raise RuntimeError(f"Unsupported reranker provider: {self._provider}")

    def _request_hf_rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        endpoint = self._hf_endpoint_url or f"/models/{self._model_name}"
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": documents,
            },
            "options": {"wait_for_model": True},
        }
        response = self._post_with_retry(endpoint, payload)
        data = response.json()
        if isinstance(data, list):
            return [float(item) for item in data]
        if isinstance(data, dict) and "scores" in data:
            return [float(item) for item in data["scores"]]
        raise RuntimeError("Unexpected HF reranker response format")

    def _request_ollama_rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        payload = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
        }
        response = self._post_with_retry("/rerank", payload)
        data = response.json()
        if isinstance(data, dict) and "data" in data:
            scored = sorted(data["data"], key=lambda item: item.get("index", 0))
            return [float(item.get("score", 0.0)) for item in scored]
        if isinstance(data, list):
            return [float(item.get("score", 0.0)) for item in data]
        raise RuntimeError("Unexpected Ollama reranker response format")

    def _post_with_retry(self, endpoint: str, payload: dict) -> httpx.Response:
        if self._client is None:
            raise RuntimeError("Reranker client is not configured")

        last_error = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 429 and attempt < self._MAX_RETRIES:
                    time.sleep(self._retry_delay_seconds(exc.response, attempt))
                    continue
                raise RuntimeError(
                    f"Reranker API error ({exc.response.status_code}) at {endpoint}: "
                    f"{exc.response.text[:200]}"
                ) from exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise RuntimeError(f"Cannot connect to reranker provider at {endpoint}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Reranker request failed: {exc}") from exc

        raise RuntimeError(f"Reranker request failed after retries: {last_error}")

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self._BASE_RETRY_DELAY_SECONDS * (2**attempt)
