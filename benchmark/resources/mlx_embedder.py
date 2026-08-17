"""MLX-accelerated embedding backend for Apple Silicon.

Drop-in SentenceTransformer replacement using Apple's MLX framework.
Uses the Neural Engine and unified memory — consistently faster than
torch+MPS for inference-only workloads on M-series chips.

Install: pip install mlx mlx-embeddings

Model names: use mlx-community quantized models, or pass a standard
HuggingFace name and let the resolver find the mlx-community equivalent.

    Benchmark model support
    -----------------------
    Model                       MLX                                     Notes
    ─────────────────────────── ──────────────────────────────────────  ─────────────────────────
    all-MiniLM-L6-v2          → mlx-community/all-MiniLM-L6-v2-4bit    BERT, ✅ supported
    BAAI/bge-base-en-v1.5     → (no mlx-community version)             BERT, falls back to ST/MPS
    BAAI/bge-m3               → mlx-community/bge-m3-mlx-4bit          XLM-RoBERTa, ✅ supported
    Qwen/Qwen3-Embedding-0.6B → mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ  ✅ supported
    Qwen/Qwen3-Embedding-4B   → mlx-community/Qwen3-Embedding-4B-4bit-DWQ    ✅ supported
    google/embeddinggemma-300m → mlx-community/embeddinggemma-300m-4bit  Gemma3, ✅ supported

    To use a model not in the mapping, pass the mlx-community name directly:
    e.g. "mlx-community/bge-m3-mlx-8bit"
"""

from __future__ import annotations

import numpy as np

_MLX_AVAILABLE = False
_mlx_load = None

try:
    import mlx.core  # type: ignore  # noqa: F401
    from mlx_embeddings.utils import load as _mlx_load  # type: ignore
    _MLX_AVAILABLE = True
except ImportError:
    pass

# ── HuggingFace name → mlx-community quantized name ─────────────────────────
# Only models that have a verified mlx-community conversion are listed here.
# Models not present fall back to sentence-transformers via the caller's
# except-branch in embeddings_strategy.py.
_HF_TO_MLX: dict[str, str] = {
    # BERT (all-MiniLM-L6-v2)
    "all-MiniLM-L6-v2":                       "mlx-community/all-MiniLM-L6-v2-4bit",
    "sentence-transformers/all-MiniLM-L6-v2": "mlx-community/all-MiniLM-L6-v2-4bit",
    # XLM-RoBERTa (bge-m3)
    "BAAI/bge-m3":                             "mlx-community/bge-m3-mlx-4bit",
    # Qwen3 Embedding
    "Qwen/Qwen3-Embedding-0.6B":              "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "Qwen/Qwen3-Embedding-4B":                "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    # Gemma3 (embeddinggemma-300m)
    "google/embeddinggemma-300m":             "mlx-community/embeddinggemma-300m-4bit",
    # Note: BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5, and
    # Qwen/Qwen3-Embedding-1.7B have no mlx-community conversion yet.
    # They will fall back to sentence-transformers automatically.
}


def resolve_mlx_model_name(model_name: str) -> str:
    """Return the mlx-community name for a given HuggingFace model name.

    If the name already starts with 'mlx-community/' it is returned unchanged.
    Unknown names are returned as-is and will fail at load time with a clear error.
    """
    if model_name.startswith("mlx-community/") or model_name.startswith("mlx/"):
        return model_name
    return _HF_TO_MLX.get(model_name, model_name)


def is_available() -> bool:
    """Return True if mlx and mlx-embeddings are installed."""
    return _MLX_AVAILABLE


class MLXEmbedder:
    """SentenceTransformer-compatible embedder backed by Apple MLX.

    Provides the same .encode() interface so it can replace SentenceTransformer
    transparently in all strategy files. Uses the Neural Engine and the full
    unified memory pool on M-series chips.

    Usage
    -----
    embedder = MLXEmbedder("all-MiniLM-L6-v2")
    vecs = embedder.encode(["hello", "world"], normalize_embeddings=True)
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        if not _MLX_AVAILABLE:
            raise ImportError(
                "mlx-embeddings is not installed.\n"
                "Run:  pip install mlx mlx-embeddings\n"
                "Then restart the benchmark."
            )
        self._hf_name = model_name
        self._mlx_name = resolve_mlx_model_name(model_name)
        try:
            self._model, self._tokenizer = _mlx_load(self._mlx_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MLX model '{self._mlx_name}' "
                f"(resolved from '{model_name}').\n"
                f"Verify the model exists at https://huggingface.co/{self._mlx_name}\n"
                f"Error: {exc}"
            ) from exc
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Public interface — mirrors SentenceTransformer.encode()
    # ------------------------------------------------------------------

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 128,
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode sentences to float32 numpy embeddings using MLX.

        Args:
            sentences: A single string or a list of strings.
            normalize_embeddings: L2-normalize each output vector (default True).
            batch_size: Sentences per forward pass. MLX can handle very large
                batches given Apple Silicon's unified memory — 128 is a good
                starting point; raise to 256–512 for large corpora.
            convert_to_tensor: Ignored (always returns numpy for compatibility).
            show_progress_bar: Ignored.

        Returns:
            float32 ndarray of shape (dim,) for a single string,
            or (n, dim) for a list.
        """
        is_single = isinstance(sentences, str)
        texts = [sentences] if is_single else list(sentences)

        if not texts:
            dim = self.get_embedding_dimension()
            empty = np.zeros((0, dim), dtype=np.float32)
            return empty[0] if is_single else empty

        parts: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer.batch_encode_plus(
                batch,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = self._model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # text_embeds: mean-pooled embeddings from the model
            embs = np.array(outputs.text_embeds, dtype=np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                embs = embs / norms
            parts.append(embs)

        result = np.vstack(parts)
        if self._dim is None:
            self._dim = result.shape[1]

        return result[0] if is_single else result

    # ------------------------------------------------------------------
    # Dimension helpers — same as SentenceTransformer
    # ------------------------------------------------------------------

    def get_embedding_dimension(self) -> int:
        if self._dim is None:
            probe = self.encode(["probe"], normalize_embeddings=False)
            self._dim = int(probe.shape[-1])
        return self._dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.get_embedding_dimension()

    def __repr__(self) -> str:
        return f"MLXEmbedder(model='{self._mlx_name}')"
