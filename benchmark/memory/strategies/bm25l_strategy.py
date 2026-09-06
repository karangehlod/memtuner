"""BM25L retrieval strategy — BM25 variant optimised for longer documents.

BM25L (Lv & Zhai 2011) improves on BM25 Okapi by replacing the term-frequency
component with a lower-bounded version:
    tf_L(t,d) = (k1+1) * c(t,d) / (k1 * (1-b + b*|d|/avdl) + c(t,d))
where c(t,d) is the normalised tf with a delta floor that prevents over-penalising
long documents. This gives measurably better recall on long-form conversational
memory (LoCoMo, LongMemEval) compared to BM25 Okapi.
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy

if TYPE_CHECKING:
    from benchmark.models.memory_event import MemoryEvent

try:
    from rank_bm25 import BM25L as _BM25L
except ImportError:
    _BM25L = None

_CACHE: dict[str, tuple] = {}
_CACHE_MAX = 16


class BM25LStrategy(RetrievalStrategy):
    """BM25L (lower-bounded TF normalisation) retrieval strategy.

    Drop-in replacement for BM25Strategy with better long-document recall.
    Requires rank-bm25 >= 0.2.2 (BM25L is included in the same package).
    """

    def __init__(self) -> None:
        if _BM25L is None:
            raise ImportError(
                "rank-bm25 not installed or too old (need >= 0.2.2). "
                "Install: pip install rank-bm25"
            )
        self._memories: dict[str, MemoryEvent] = {}
        self._bm25: _BM25L | None = None
        self._id_list: list[str] = []
        self._user_index: dict[str, set] = {}
        # Boolean mask cache: same pattern as BM25Strategy — built once per (user_id, corpus).
        self._user_mask_cache: dict[str, np.ndarray] = {}

    def index(self, memories: list[MemoryEvent]) -> None:
        self._memories = {m.id: m for m in memories}
        if not memories:
            self._bm25 = None
            self._id_list = []
            self._user_index = {}
            self._user_mask_cache = {}
            return

        _hash = hashlib.md5(
            "|".join(sorted(m.id for m in memories)).encode()
        ).hexdigest()[:16]
        _key = f"bm25l:{_hash}"

        if _key in _CACHE:
            self._bm25, self._id_list, self._user_index = _CACHE[_key]
            self._user_mask_cache = {}  # invalidate mask cache on corpus change
            return

        self._id_list = [m.id for m in memories]
        self._user_mask_cache = {}  # invalidate on corpus change
        self._user_index = {}
        for i, m in enumerate(memories):
            uid = m.user_id or "__none__"
            self._user_index.setdefault(uid, set()).add(i)

        token_lists = [_PUNCT_RE.sub(" ", m.content.lower()).split() for m in memories]
        self._bm25 = _BM25L(token_lists) if token_lists else None

        if len(_CACHE) >= _CACHE_MAX:
            del _CACHE[next(iter(_CACHE))]
        _CACHE[_key] = (self._bm25, self._id_list, self._user_index)

    @property
    def name(self) -> str:
        return "bm25l"

    def clear(self) -> None:
        self._memories = {}
        self._bm25 = None
        self._id_list = []
        self._user_index = {}
        self._user_mask_cache = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._id_list:
            return []

        tokens = _PUNCT_RE.sub(" ", query.lower()).split()
        scores = self._bm25.get_scores(tokens)

        scores_arr = np.array(scores, dtype=np.float32)

        # Apply user filter via cached boolean mask — built once per (user_id, corpus).
        if user_id:
            if user_id not in self._user_mask_cache:
                valid = (
                    self._user_index.get(user_id, set())
                    | self._user_index.get("__none__", set())
                )
                mask = np.zeros(len(self._id_list), dtype=bool)
                for idx in valid:
                    if idx < len(mask):
                        mask[idx] = True
                self._user_mask_cache[user_id] = mask
            scores_arr[~self._user_mask_cache[user_id]] = 0.0

        # heapq.nlargest is O(N log K) vs O(N log N) for a full sort; K is typically 5-10.
        if top_k < len(scores_arr):
            top_indices = np.argpartition(scores_arr, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores_arr[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores_arr)[::-1]

        return [
            (self._id_list[idx], float(scores_arr[idx]))
            for idx in top_indices
            if scores_arr[idx] > 0
        ][:top_k]
