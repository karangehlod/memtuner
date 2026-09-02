"""BM25 keyword-based retrieval strategy.

Fast, deterministic keyword matching using BM25 (TF-IDF variant).
No external LLM calls or embeddings needed.

Latency: <10ms | Cost: Free | Accuracy: Good | Setup: 30 min
"""

import hashlib

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent

# BM25 corpus cache — avoids re-tokenising the same memory set for every cell.
# Phase 5 runs 57 cells on the same episodic corpus; this reduces 57 tokenisations to 1.
# Key: "bm25:<corpus_hash>"  Value: (BM25Okapi, id_list, user_index dict)
_BM25_CORPUS_CACHE: dict[str, tuple] = {}
_BM25_CACHE_MAX = 4


class BM25Strategy(RetrievalStrategy):
    """BM25 (keyword matching) retrieval strategy."""

    def __init__(self) -> None:
        """Initialize BM25 strategy."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed. Install: pip install rank-bm25")
        self._memories: dict[str, MemoryEvent] = {}
        self._bm25: BM25Okapi | None = None
        self._id_list: list[str] = []  # Ordered list for index-based access
        self._user_index: dict[str, set] = {}  # user_id → set of indices (pre-built)
        # Boolean mask cache: True at positions belonging to the given user_id.
        # Built once per (user_id, corpus) pair; invalidated on re-index.
        self._user_mask_cache: dict[str, np.ndarray] = {}

    def index(self, memories: list[MemoryEvent]) -> None:
        """Index memories for BM25 retrieval.

        Args:
            memories: List of memories to index.
        """
        self._memories = {mem.id: mem for mem in memories}

        if not memories:
            self._id_list = []
            self._user_index = {}
            self._bm25 = None
            return

        # Corpus cache: same memory set → skip tokenisation and BM25Okapi rebuild
        _corpus_hash = hashlib.md5(
            "|".join(sorted(m.id for m in memories)).encode()
        ).hexdigest()[:16]
        _cache_key = f"bm25:{_corpus_hash}"

        if _cache_key in _BM25_CORPUS_CACHE:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "[BM25_CACHE] Hit — skipping tokenisation of %d memories (hash=%s)", len(memories), _corpus_hash
            )
            self._bm25, self._id_list, self._user_index = _BM25_CORPUS_CACHE[_cache_key]
            self._user_mask_cache = {}
            return

        self._id_list = [mem.id for mem in memories]
        self._user_mask_cache = {}  # invalidate on corpus change

        # Build user index for O(1) user filtering
        self._user_index = {}
        for i, mem in enumerate(memories):
            uid = mem.user_id or "__none__"
            if uid not in self._user_index:
                self._user_index[uid] = set()
            self._user_index[uid].add(i)

        # Tokenize and build BM25 index
        token_lists = [mem.content.lower().split() for mem in memories]
        self._bm25 = BM25Okapi(token_lists) if token_lists else None

        # Evict oldest if at capacity, then store
        if len(_BM25_CORPUS_CACHE) >= _BM25_CACHE_MAX:
            oldest = next(iter(_BM25_CORPUS_CACHE))
            del _BM25_CORPUS_CACHE[oldest]
        _BM25_CORPUS_CACHE[_cache_key] = (self._bm25, self._id_list, self._user_index)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve using BM25 ranking.

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, score) tuples.
        """
        if not self._bm25 or not self._memories:
            return []

        # BM25 scoring — returns scores array aligned with index order
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Apply user filter via cached boolean mask — built once per (user_id, corpus)
        if user_id:
            valid_indices = self._user_index.get(user_id)
            if not valid_indices:
                return []
            if user_id not in self._user_mask_cache:
                mask = np.zeros(len(self._id_list), dtype=bool)
                for idx in valid_indices:
                    mask[idx] = True
                self._user_mask_cache[user_id] = mask
            scores_arr = np.array(scores, dtype=np.float32)
            scores_arr[~self._user_mask_cache[user_id]] = 0.0
        else:
            scores_arr = np.array(scores, dtype=np.float32)

        # O(N) argpartition for top-k instead of O(N log N) full sort
        if top_k < len(scores_arr):
            top_indices = np.argpartition(scores_arr, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores_arr[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores_arr)[::-1]

        results = []
        for idx in top_indices:
            score = float(scores_arr[idx])
            if score <= 0:
                break
            results.append((self._id_list[idx], score))

        return results[:top_k]

    def name(self) -> str:
        """Return strategy name."""
        return "bm25"

    def clear(self) -> None:
        """Clear all indexed data.

        Rebinds to fresh containers rather than mutating in place — index()
        stores these exact objects (uncopied) in the module-level corpus
        cache, so an in-place .clear() would corrupt that cache for every
        other instance sharing the same cached corpus.
        """
        self._memories = {}
        self._id_list = []
        self._user_index = {}
        self._user_mask_cache = {}
        self._bm25 = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if rank-bm25 is installed."""
        return BM25Okapi is not None
