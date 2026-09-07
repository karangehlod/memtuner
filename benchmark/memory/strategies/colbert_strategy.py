"""ColBERT-style token-level retrieval strategy.

ColBERT (Khattab & Zaharia 2020) scores a (query, document) pair via MaxSim:
    score(q,d) = sum_{t in q} max_{t' in d} cos(E_t, E_{t'})

where E_t are the contextual token embeddings from a transformer encoder.
Unlike bi-encoder (dense) retrieval — which compresses the entire text into one
vector — ColBERT preserves token-level detail and is especially strong at:
  - exact phrase recall
  - multi-hop reasoning queries
  - queries that reference specific named entities

Implementation note: true ColBERT requires a model fine-tuned with the late-
interaction objective (e.g., colbert-ir/colbertv2.0). This strategy uses
all-MiniLM-L6-v2 as the encoder and applies the MaxSim scoring formula,
giving a ColBERT-style signal without the full training overhead. Swap the
model name via BENCHMARK_COLBERT_MODEL to use a proper ColBERT checkpoint.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy

if TYPE_CHECKING:
    from benchmark.models.memory_event import MemoryEvent

try:
    from sentence_transformers import SentenceTransformer as _ST  # noqa: N814
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from benchmark.resources.hw_probe import DEVICE as _DEVICE

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, _ST] = {}

# ColBERT corpus cache — stores pre-computed chunk embeddings per corpus+model.
# Phase 2 runs 2 memory_type cells for ColBERT (episodic + preference); without this
# cache the second cell re-encodes all 5,879 memories × 10 chunks = ~86s wasted.
# Key: "model_name:device:corpus_hash"  Value: (doc_vecs dict, user_index dict)
import hashlib as _hashlib
from collections import deque as _deque

_COLBERT_INDEX_CACHE: dict[str, tuple[dict, dict]] = {}
_COLBERT_INDEX_CACHE_ORDER: _deque[str] = _deque()
_COLBERT_INDEX_CACHE_MAX = 4  # doc_vecs per entry can be 50-200 MB

# Number of tokens to split each text into for MaxSim scoring.
# Larger = more accurate but slower. 32 is a good balance.
_CHUNK_TOKENS = int(os.environ.get("BENCHMARK_COLBERT_CHUNK", "32"))


def _get_model(name: str) -> _ST:
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = _ST(name, device=_DEVICE)
    return _MODEL_CACHE[name]


def _chunk_text(text: str, n: int) -> list[str]:
    """Split text into overlapping n-word windows for token-level encoding."""
    words = text.split()
    if len(words) <= n:
        return [text]
    return [
        " ".join(words[i : i + n])
        for i in range(0, max(1, len(words) - n + 1), n // 2)
    ]


def _maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT MaxSim: sum of per-query-token maximum cosine similarity."""
    # query_vecs: (Q, D)   doc_vecs: (K, D)
    q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sim = q_norm @ d_norm.T          # (Q, K)
    return float(sim.max(axis=1).sum())


class ColBERTStrategy(RetrievalStrategy):
    """Token-level MaxSim retrieval (ColBERT-style).

    Indexes each memory as a set of overlapping token-window embeddings.
    Scores each (query, memory) pair via MaxSim aggregation.
    """

    def __init__(self, model_name: str | None = None) -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install: pip install sentence-transformers"
            )
        # Priority: explicit model_name arg (from StudyCell config) > env var > hardcoded default
        _model_name = model_name or os.environ.get("BENCHMARK_COLBERT_MODEL", _DEFAULT_MODEL)
        self._model = _get_model(_model_name)
        self._memories: dict[str, MemoryEvent] = {}
        # memory_id → stacked token embeddings (K, D)
        self._doc_vecs: dict[str, np.ndarray] = {}
        self._user_index: dict[str, list[str]] = {}

    def name(self) -> str:
        return "colbert"

    def clear(self) -> None:
        self._memories = {}
        self._doc_vecs = {}
        self._user_index = {}

    def index(self, memories: list[MemoryEvent]) -> None:
        self._memories = {m.id: m for m in memories}
        self._doc_vecs = {}
        self._user_index = {}

        if not memories:
            return

        # Check corpus cache — prevents re-encoding 5879 mems × 10 chunks (~86s)
        # when the second memory-type cell has the same corpus+model as the first.
        _corpus_hash = _hashlib.md5(
            "|".join(sorted(m.id for m in memories)).encode()
        ).hexdigest()[:16]
        _model_key = getattr(self._model, "model_card_data", {}).get("model_name", "") or str(type(self._model))
        _cache_key = f"colbert:{_model_key}:{_corpus_hash}"

        if _cache_key in _COLBERT_INDEX_CACHE:
            self._doc_vecs, self._user_index = _COLBERT_INDEX_CACHE[_cache_key]
            return

        for m in memories:
            chunks = _chunk_text(m.content, _CHUNK_TOKENS)
            vecs = self._model.encode(chunks, batch_size=256, show_progress_bar=False)
            self._doc_vecs[m.id] = np.array(vecs)
            uid = m.user_id or "__none__"
            self._user_index.setdefault(uid, []).append(m.id)
            self._user_index.setdefault("__none__", [])

        # Store in cache
        while len(_COLBERT_INDEX_CACHE) >= _COLBERT_INDEX_CACHE_MAX:
            _COLBERT_INDEX_CACHE.pop(_COLBERT_INDEX_CACHE_ORDER.popleft(), None)
        _COLBERT_INDEX_CACHE[_cache_key] = (self._doc_vecs, self._user_index)
        _COLBERT_INDEX_CACHE_ORDER.append(_cache_key)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if not self._doc_vecs:
            return []

        q_chunks = _chunk_text(query, _CHUNK_TOKENS)
        q_vecs = np.array(
            self._model.encode(q_chunks, batch_size=256, show_progress_bar=False)
        )

        if user_id:
            candidates = (
                self._user_index.get(user_id, [])
                + self._user_index.get("__none__", [])
            )
            candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order
        else:
            candidates = list(self._doc_vecs.keys())

        scored = [
            (mid, _maxsim(q_vecs, self._doc_vecs[mid]))
            for mid in candidates
            if mid in self._doc_vecs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
