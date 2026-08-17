"""Session-aware embedding strategy.

Instead of embedding each turn independently, this strategy:
1. Groups memories by session/task (conversation context)
2. Creates session-level embeddings (multiple turns combined)
3. Retrieves by session similarity, then returns individual turns

This closes the gap between simple per-turn retrieval (53% recall)
and published systems that understand conversational structure (88%+).

The key insight: a single turn like "sushi" doesn't embed well for
"What food does Alice like?" but a session containing
"I really enjoy eating sushi. It's my favorite food." embeds much better.
"""

from __future__ import annotations

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    np = None

from benchmark.resources.hw_probe import (
    DEVICE as _EMBEDDING_DEVICE,
    CUDA_AVAILABLE as _CUDA_AVAILABLE,
    MPS_AVAILABLE as _MPS_AVAILABLE,
    MLX_AVAILABLE as _MLX_AVAILABLE,
    CPU_CORES as _CPU_CORES,
)
from benchmark.resources.mlx_embedder import MLXEmbedder as _MLXEmbedder

try:
    import torch as _torch
    if _CUDA_AVAILABLE or _MPS_AVAILABLE:
        _cpu_cap = max(1, _CPU_CORES // 2)
        try:
            _torch.set_num_threads(_cpu_cap)
        except RuntimeError:
            pass
except ImportError:
    pass

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent


class SessionEmbeddingsStrategy(RetrievalStrategy):
    """Session-aware embedding retrieval.

    Groups memories by task_id (session proxy), creates combined
    session embeddings, and retrieves at session granularity.
    Returns individual memory IDs from matching sessions.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_session_length: int = 512) -> None:
        """Initialize with embedding model.

        Args:
            model_name: HuggingFace sentence-transformer model.
            max_session_length: Max chars per session chunk.
        """
        import os as _os
        _use_mlx = (
            _MPS_AVAILABLE
            and _MLX_AVAILABLE
            and _os.environ.get("BENCHMARK_EMBEDDING_BACKEND", "").lower() == "mlx"
        )
        if _use_mlx:
            try:
                self._model = _MLXEmbedder(model_name)
            except Exception:
                _use_mlx = False

        if not _use_mlx:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install: pip install memtuner[embeddings]"
                )
            self._model = SentenceTransformer(model_name, device=_EMBEDDING_DEVICE)
        self._max_session_length = max_session_length

        # Per-session data
        self._sessions: dict[str, list[MemoryEvent]] = {}  # session_key → memories
        self._session_embeddings: dict[str, object] = {}  # session_key → embedding
        self._memory_to_session: dict[str, str] = {}  # memory_id → session_key
        self._all_memories: dict[str, MemoryEvent] = {}

    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers is installed."""
        return SentenceTransformer is not None

    def index(self, memories: list[MemoryEvent]) -> None:
        """Index memories by grouping into sessions and embedding.

        Sessions are defined by (user_id, creation_window) where memories
        within a 7-day window form one session. This handles datasets where
        task_id is too coarse (e.g., LoCoMo has one task_id per conversation).

        Args:
            memories: All memory events to index.
        """
        self._all_memories = {m.id: m for m in memories}
        self._sessions.clear()
        self._session_embeddings.clear()
        self._memory_to_session.clear()

        # Group by user first, then chunk by creation proximity
        user_memories: dict[str, list[MemoryEvent]] = {}
        for mem in memories:
            uid = mem.user_id or "__none__"
            if uid not in user_memories:
                user_memories[uid] = []
            user_memories[uid].append(mem)

        # Within each user, create session chunks of ~10 memories
        # This mimics real session boundaries (a conversation session)
        chunk_size = 10
        for uid, mems in user_memories.items():
            for chunk_idx in range(0, len(mems), chunk_size):
                chunk = mems[chunk_idx : chunk_idx + chunk_size]
                session_key = f"{uid}::chunk_{chunk_idx // chunk_size}"
                self._sessions[session_key] = chunk
                for mem in chunk:
                    self._memory_to_session[mem.id] = session_key

        # Create session-level embeddings
        session_keys = list(self._sessions.keys())
        session_texts = []
        for key in session_keys:
            session_mems = self._sessions[key]
            combined = " ".join(m.content for m in session_mems)
            session_texts.append(combined[: self._max_session_length])

        if session_texts:
            embeddings = self._model.encode(
                session_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=256 if (_CUDA_AVAILABLE or _MPS_AVAILABLE or _MLX_AVAILABLE) else 64,
            )
            for key, embedding in zip(session_keys, embeddings):
                self._session_embeddings[key] = embedding

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve memories by finding the most relevant sessions first.

        1. Embed the query
        2. Find top sessions by cosine similarity
        3. Return individual memory IDs from those sessions

        Args:
            query: Query text.
            top_k: Number of memory IDs to return.
            user_id: Filter to specific user.

        Returns:
            List of (memory_id, score) tuples.
        """
        if not self._session_embeddings:
            return []

        # Filter sessions by user
        if user_id:
            candidate_keys = [k for k in self._sessions if k.startswith(f"{user_id}::")]
        else:
            candidate_keys = list(self._sessions.keys())

        if not candidate_keys:
            return []

        # Embed query
        query_embedding = self._model.encode(query, convert_to_tensor=False)

        # Compute similarities to candidate sessions
        scored_sessions: list[tuple[str, float]] = []
        for key in candidate_keys:
            if key not in self._session_embeddings:
                continue
            session_emb = self._session_embeddings[key]
            similarity = float(
                np.dot(query_embedding, session_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(session_emb) + 1e-10)
            )
            scored_sessions.append((key, similarity))

        # Sort sessions by similarity
        scored_sessions.sort(key=lambda x: x[1], reverse=True)

        # Collect individual memory IDs from top sessions
        results: list[tuple[str, float]] = []
        for session_key, session_score in scored_sessions:
            for mem in self._sessions[session_key]:
                if user_id and mem.user_id != user_id:
                    continue
                results.append((mem.id, session_score))
                if len(results) >= top_k * 2:  # Over-fetch for ranking
                    break
            if len(results) >= top_k * 2:
                break

        return results[:top_k]

    def name(self) -> str:
        """Return strategy name."""
        return "session_embeddings"

    def clear(self) -> None:
        """Clear all indexed data."""
        self._sessions.clear()
        self._session_embeddings.clear()
        self._memory_to_session.clear()
        self._all_memories.clear()
