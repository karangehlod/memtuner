"""Embedding-based semantic retrieval strategy.

Uses SentenceTransformers for semantic similarity matching.
Understands synonyms, paraphrasing, and semantic relevance.

OPTIMIZATIONS:
- Query embedding cache: Reuse embeddings for repeated queries (20-30% faster if queries repeat)
- Batch encoding: Memories indexed in batches (already implemented)
- Vectorized similarity: Matrix operations for fast retrieval

Latency: 50-200ms | Cost: Low | Accuracy: Excellent | Setup: 1 hour
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from benchmark.resources.hw_probe import (
    CUDA_AVAILABLE as _CUDA_AVAILABLE,
    MPS_AVAILABLE as _MPS_AVAILABLE,
    MLX_AVAILABLE as _MLX_AVAILABLE,
    DEVICE as _EMBEDDING_DEVICE,
    CPU_CORES as _CPU_CORES,
    embed_batch_size as _hw_embed_batch_size,
)
from benchmark.resources.mlx_embedder import MLXEmbedder as _MLXEmbedder

# Cap PyTorch intra-op threads to half of logical cores when a GPU is active,
# so the CPU doesn't compete with CUDA/MPS kernels and the OS scheduler.
try:
    import torch as _torch
    if _CUDA_AVAILABLE or _MPS_AVAILABLE:
        _cpu_cap = max(1, _CPU_CORES // 2)
        try:
            _torch.set_num_threads(_cpu_cap)
        except RuntimeError:
            pass  # already started — OMP_NUM_THREADS handles it instead
except ImportError:
    pass

# Model singleton cache — avoids reloading 90-450 MB weights for every cell.
# Key: "<model_name>:<device>" so different devices stay separate.
# Max 2 models resident at once — evicts the oldest when full to free VRAM.
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}
_MODEL_CACHE_ORDER: list[str] = []  # insertion-order LRU tracking
_MODEL_CACHE_MAX = 2                 # keep at most 2 models in VRAM at once


def _evict_model_cache_if_needed() -> None:
    """Evict the oldest cached model when the cache is at capacity."""
    while len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        oldest_key = _MODEL_CACHE_ORDER.pop(0)
        evicted = _MODEL_CACHE.pop(oldest_key, None)
        if evicted is not None:
            # Move weights off GPU before deleting so VRAM is freed immediately
            try:
                evicted.to("cpu")
            except Exception:
                pass
            del evicted
            try:
                if _CUDA_AVAILABLE:
                    _torch.cuda.empty_cache()
                elif _MPS_AVAILABLE:
                    _torch.mps.empty_cache()
            except Exception:
                pass

# Index cache — stores the encoded matrix so identical corpus+model combos skip .encode()
# Key: "model_name:device:corpus_hash"  Value: (matrix, ids, norms)
_INDEX_CACHE: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
_INDEX_CACHE_ORDER: list[str] = []
_INDEX_CACHE_MAX = 16  # 16 × ~17MB bge-base matrices ≈ 272MB; well within 64GB RAM


def _evict_index_cache_if_needed() -> None:
    while len(_INDEX_CACHE) >= _INDEX_CACHE_MAX:
        _INDEX_CACHE.pop(_INDEX_CACHE_ORDER.pop(0), None)


from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent


class EmbeddingsStrategy(RetrievalStrategy):
    """Embedding-based (semantic similarity) retrieval strategy."""

    _MODEL_SIZE_MB = {
        "all-MiniLM-L6-v2": 90,
        "sentence-transformers/all-MiniLM-L6-v2": 90,
        "BAAI/bge-base-en-v1.5": 210,
        "BAAI/bge-large-en-v1.5": 1300,
        "BAAI/bge-m3": 1100,                   # multilingual, 1024-dim
        "Qwen/Qwen3-Embedding-0.6B": 1200,     # 0.6B, 1024-dim
        "Qwen/Qwen3-Embedding-4B": 7600,       # 4B, 2560-dim — fits alone on 16GB A4000
        "nvidia/NV-Embed-v2": 16000,
    }

    # Embedding models ranked by quality (MTEB retrieval scores, approximate):
    #   all-MiniLM-L6-v2:           ~56 (22M params,   384-dim, fast baseline)
    #   BAAI/bge-base-en-v1.5:      ~64 (110M params,  768-dim, balanced English)
    #   BAAI/bge-large-en-v1.5:     ~66 (335M params, 1024-dim, high quality English)
    #   BAAI/bge-m3:                ~71 (567M params, 1024-dim, multilingual)
    #   Qwen/Qwen3-Embedding-0.6B:  ~68 (596M params, 1024-dim, strong on reasoning)
    #   Qwen/Qwen3-Embedding-4B:    ~74 (4B params,  2560-dim, best quality, needs 8GB VRAM alone)
    #   nvidia/NV-Embed-v2:         ~72 (7B params,  4096-dim, SOTA but 16GB alone)

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        comparison_models: str | None = None,
    ) -> None:
        """Initialize embeddings strategy.

        Args:
            model_name: HuggingFace model for embeddings. If None, uses the default
                all-MiniLM-L6-v2.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. Install: pip install sentence-transformers"
            )

        resolved_model = model_name or self.DEFAULT_MODEL
        self._cache_dir = cache_dir
        self._comparison_models = comparison_models

        # Models that require trust_remote_code=True (custom modeling classes).
        _TRUST_REMOTE_CODE_MODELS = frozenset({
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-1.7B",
            "Qwen/Qwen3-Embedding-4B",
        })

        # MLX backend: opt-in only via BENCHMARK_EMBEDDING_BACKEND=mlx.
        # Do NOT auto-select MLX — it uses 4-bit quantized models which change
        # embedding quality and make results non-reproducible across platforms.
        # The VRAM fix in hw_probe.py already gives 8× larger batch sizes on MPS.
        _use_mlx = (
            _MPS_AVAILABLE
            and _MLX_AVAILABLE
            and os.environ.get("BENCHMARK_EMBEDDING_BACKEND", "").lower() == "mlx"
        )
        _backend_tag = "mlx" if _use_mlx else _EMBEDDING_DEVICE

        try:
            _cache_key = f"{resolved_model}:{_backend_tag}"
            if _cache_key not in _MODEL_CACHE:
                _evict_model_cache_if_needed()
                if _use_mlx:
                    _MODEL_CACHE[_cache_key] = _MLXEmbedder(resolved_model, cache_dir=cache_dir)
                else:
                    if SentenceTransformer is None:
                        raise ImportError(
                            "sentence-transformers not installed. "
                            "Install: pip install sentence-transformers"
                        )
                    init_kwargs: dict = {"device": _EMBEDDING_DEVICE}
                    if cache_dir:
                        init_kwargs["cache_folder"] = cache_dir
                    if resolved_model in _TRUST_REMOTE_CODE_MODELS:
                        init_kwargs["trust_remote_code"] = True
                    _MODEL_CACHE[_cache_key] = SentenceTransformer(resolved_model, **init_kwargs)
                _MODEL_CACHE_ORDER.append(_cache_key)
            else:
                # Promote to most-recently-used
                if _cache_key in _MODEL_CACHE_ORDER:
                    _MODEL_CACHE_ORDER.remove(_cache_key)
                _MODEL_CACHE_ORDER.append(_cache_key)
            self._model = _MODEL_CACHE[_cache_key]
        except Exception as e:
            if _use_mlx:
                # MLX failed — fall back to sentence-transformers automatically
                _backend_tag = _EMBEDDING_DEVICE
                _cache_key = f"{resolved_model}:{_backend_tag}"
                if _cache_key not in _MODEL_CACHE:
                    _evict_model_cache_if_needed()
                    if SentenceTransformer is None:
                        raise ImportError(
                            "sentence-transformers not installed. "
                            "Install: pip install sentence-transformers"
                        ) from e
                    init_kwargs = {"device": _EMBEDDING_DEVICE}
                    if cache_dir:
                        init_kwargs["cache_folder"] = cache_dir
                    if resolved_model in _TRUST_REMOTE_CODE_MODELS:
                        init_kwargs["trust_remote_code"] = True
                    _MODEL_CACHE[_cache_key] = SentenceTransformer(resolved_model, **init_kwargs)
                    _MODEL_CACHE_ORDER.append(_cache_key)
                else:
                    if _cache_key in _MODEL_CACHE_ORDER:
                        _MODEL_CACHE_ORDER.remove(_cache_key)
                    _MODEL_CACHE_ORDER.append(_cache_key)
                self._model = _MODEL_CACHE[_cache_key]
            else:
                raise RuntimeError(
                    f"Failed to load embedding model {resolved_model}. "
                    f"Install: pip install sentence-transformers\n{e}"
                ) from e

        self._backend = _backend_tag

        self._model_name = resolved_model

        # Refine batch size using the actual embedding dimension of the loaded
        # model. Falls back to the class-level default if the dimension can't
        # be read (e.g., third-party model without a standard config attribute).
        try:
            # get_embedding_dimension() is the current name; fall back to the
            # deprecated get_sentence_embedding_dimension() for older versions.
            if hasattr(self._model, "get_embedding_dimension"):
                _actual_dim = self._model.get_embedding_dimension() or 384
            else:
                _actual_dim = self._model.get_sentence_embedding_dimension() or 384
        except Exception:
            _actual_dim = 384
        self._batch_size: int = _hw_embed_batch_size(_actual_dim)

        self._memories: dict[str, MemoryEvent] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        # Pre-computed matrix for vectorized cosine similarity
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_ids: list[str] = []
        self._norms: np.ndarray | None = None

        # OPTIMIZATION: Query embedding cache (20-30% speedup if queries repeat)
        self._query_embedding_cache: dict[str, np.ndarray] = {}
        # OPTIMIZATION: User filter mask cache — mask is identical for all queries
        # with the same user_id within one cell; built once, reused 1977×
        self._user_mask_cache: dict[str, np.ndarray] = {}
        # Enable/disable cache via environment (default: enabled)
        self._cache_enabled = os.environ.get("BENCHMARK_EMBEDDING_QUERY_CACHE", "true").lower() == "true"

    def index(self, memories: list[MemoryEvent]) -> None:
        """Index memories by computing embeddings.

        Uses incremental encoding: only newly added memories are encoded;
        existing embeddings are reused from the previous index state.
        This reduces total encode work from O(N²) across days to O(N)
        (each memory is encoded exactly once per cell run).

        Args:
            memories: List of memories to index.

        Raises:
            RuntimeError: If embedding computation fails.
        """
        incoming = {mem.id: mem for mem in memories}

        # Invalidate user-mask cache whenever corpus changes
        new_ids = set(incoming) - set(self._memories)
        removed_ids = set(self._memories) - set(incoming)
        if new_ids or removed_ids:
            self._user_mask_cache = {}

        # Incremental path: keep embeddings for unchanged memories, encode only new ones
        if self._embedding_matrix is not None and new_ids and not removed_ids:
            new_mems = [incoming[mid] for mid in sorted(new_ids)]
            new_contents = [m.content for m in new_mems]
            try:
                new_vecs = self._model.encode(
                    new_contents,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self._batch_size,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to encode {len(new_mems)} new memories: {e}")

            # Append new rows to the existing matrix
            new_matrix = np.array(new_vecs, dtype=np.float32)
            self._embedding_matrix = np.vstack([self._embedding_matrix, new_matrix])
            for i, mem in enumerate(new_mems):
                self._embeddings[mem.id] = new_vecs[i]
                self._embedding_ids.append(mem.id)
            self._norms = np.ones((len(self._embedding_ids), 1), dtype=np.float32)
            self._memories = incoming

            # Update the index cache with the grown matrix so that future instances
            # starting from the same full corpus can skip encoding entirely.
            _all_ids = list(incoming.keys())
            _full_hash = hashlib.md5("|".join(sorted(_all_ids)).encode()).hexdigest()[:16]
            _full_key = f"{self._model_name}:{self._backend}:{_full_hash}"
            if _full_key not in _INDEX_CACHE:
                _evict_index_cache_if_needed()
                _INDEX_CACHE[_full_key] = (self._embedding_matrix, list(self._embedding_ids), self._norms)
                _INDEX_CACHE_ORDER.append(_full_key)
            return

        # Full rebuild path: first call, or memories were removed (pruning)
        self._memories = incoming
        self._embeddings = {}

        if not memories:
            self._embedding_matrix = None
            self._embedding_ids = []
            self._norms = None
            return

        ids = [mem.id for mem in memories]

        # Index cache: same model + same corpus → skip re-encoding entirely
        _corpus_hash = hashlib.md5("|".join(sorted(ids)).encode()).hexdigest()[:16]
        _index_key = f"{self._model_name}:{self._backend}:{_corpus_hash}"
        if _index_key in _INDEX_CACHE:
            cached_matrix, cached_ids, cached_norms = _INDEX_CACHE[_index_key]
            self._embedding_matrix = cached_matrix
            self._embedding_ids = list(cached_ids)  # copy — incremental appends must not mutate the cache
            self._norms = cached_norms
            for i, mid in enumerate(cached_ids):
                self._embeddings[mid] = cached_matrix[i]
            if _index_key in _INDEX_CACHE_ORDER:
                _INDEX_CACHE_ORDER.remove(_index_key)
            _INDEX_CACHE_ORDER.append(_index_key)
            return

        contents = [mem.content for mem in memories]
        batch_size = self._batch_size
        all_embeddings = None
        last_err: Exception | None = None
        while batch_size >= 1:
            try:
                all_embeddings = self._model.encode(
                    contents,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                # Update instance batch size so future calls start from this known-good value
                self._batch_size = batch_size
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    # Free VRAM and retry with half the batch size
                    try:
                        import torch as _t
                        _t.cuda.empty_cache()
                    except Exception:
                        pass
                    batch_size = max(1, batch_size // 2)
                    last_err = e
                    continue
                raise RuntimeError(f"Failed to batch-encode {len(memories)} memories: {e}") from e
        if all_embeddings is None:
            raise RuntimeError(
                f"Failed to batch-encode {len(memories)} memories after OOM retries: {last_err}"
            )

        for i, mem_id in enumerate(ids):
            self._embeddings[mem_id] = all_embeddings[i]

        self._embedding_matrix = np.array(all_embeddings, dtype=np.float32)
        self._embedding_ids = ids          # live reference — may grow via incremental appends
        self._norms = np.ones((len(ids), 1), dtype=np.float32)

        _evict_index_cache_if_needed()
        # Store a COPY of ids in the cache — incremental appends on self._embedding_ids
        # must never mutate the cached list (which would corrupt every future cache hit).
        _INDEX_CACHE[_index_key] = (self._embedding_matrix, list(ids), self._norms)
        _INDEX_CACHE_ORDER.append(_index_key)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve using semantic similarity.

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, score) tuples.

        Raises:
            RuntimeError: If query encoding fails.
        """
        if not self._memories or self._embedding_matrix is None:
            return []

        # OPTIMIZATION: Check query embedding cache first (20-30% speedup if repeated)
        # Use hashlib.md5 — Python's hash() is non-collision-safe and unstable across runs
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if self._cache_enabled and query_hash in self._query_embedding_cache:
            query_embedding = self._query_embedding_cache[query_hash]
        else:
            # Encode query — normalize so cosine = dot product (matches index)
            try:
                query_embedding = self._model.encode(
                    query, convert_to_tensor=False, normalize_embeddings=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to encode query '{query}': {e}") from e

            # Cache this query embedding for reuse
            if self._cache_enabled:
                self._query_embedding_cache[query_hash] = query_embedding

        # Cosine similarity = dot product (both sides are unit vectors)
        query_vec = np.array(query_embedding, dtype=np.float32)
        similarities = (self._embedding_matrix @ query_vec
        )
        # Clamp negatives to 0
        np.maximum(similarities, 0.0, out=similarities)

        # Apply user filter via boolean mask — cached per user_id, built once per index()
        if user_id:
            if user_id not in self._user_mask_cache:
                self._user_mask_cache[user_id] = np.array(
                    [self._memories[mid].user_id == user_id for mid in self._embedding_ids],
                    dtype=bool,
                )
            similarities[~self._user_mask_cache[user_id]] = -1.0

        # Use argpartition for O(N) top-k selection instead of O(N log N) full sort
        if top_k < len(similarities):
            # Get indices of top-k largest values
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort only those k elements
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                break
            results.append((self._embedding_ids[idx], score))

        return results[:top_k]

    def name(self) -> str:
        """Return strategy name."""
        return "embeddings"

    @classmethod
    def known_model_size_mb(cls, model_name: str) -> int | None:
        return cls._MODEL_SIZE_MB.get(model_name)

    def clear(self) -> None:
        """Clear all indexed data and caches."""
        self._memories.clear()
        self._embeddings.clear()
        self._embedding_matrix = None
        self._embedding_ids = []
        self._norms = None
        self._query_embedding_cache.clear()
        self._user_mask_cache.clear()

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch encode multiple texts at once.

        OPTIMIZATION (Phase 2): Process multiple queries in one pass
        instead of encoding them individually. Expected: 40% speedup.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embeddings (one per text).

        Raises:
            RuntimeError: If batch encoding fails.
        """
        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Batch encoding failed for {len(texts)} texts: {e}") from e

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Batch retrieve for multiple queries using pre-computed embeddings.

        OPTIMIZATION (Phase 2): Encode all queries at once, then compute
        similarities for all queries. Expected: 40% speedup.

        Args:
            queries: List of query strings.
            top_k: Number of top results per query.
            user_id: Filter results to specific user (optional).

        Returns:
            List of results per query: each is list of (memory_id, score) tuples.
        """
        if not self._memories or self._embedding_matrix is None:
            return [[] for _ in queries]

        # Batch encode all queries at once
        query_embeddings = self.encode_batch(queries)

        results = []
        for query_embedding in query_embeddings:
            # Vectorized cosine similarity
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            query_norm = np.linalg.norm(query_vec) + 1e-8

            similarities = (self._embedding_matrix @ query_vec.T).flatten() / (
                self._norms.flatten() * query_norm
            )
            np.maximum(similarities, 0.0, out=similarities)

            # Apply user filter if needed
            if user_id:
                mask = np.array(
                    [self._memories[mid].user_id == user_id for mid in self._embedding_ids],
                    dtype=bool,
                )
                similarities[~mask] = -1.0

            # Get top-k results
            if top_k < len(similarities):
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]

            top_results = [
                (self._embedding_ids[idx], float(similarities[idx]))
                for idx in top_indices
                if similarities[idx] >= 0.0
            ]
            results.append(top_results[:top_k])

        return results

    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers is installed."""
        return SentenceTransformer is not None
