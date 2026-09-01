"""Hardware probe — detects CPU cores, GPU backend, VRAM, and SM count at import time.

Supports three GPU backends in priority order:
  1. CUDA  (NVIDIA)  — Linux, Windows, WSL
  2. MPS   (Apple)   — macOS Apple Silicon (M1/M2/M3/M4)
  3. CPU             — fallback on any platform

Results cached as module-level constants — every caller gets the same
values without repeated syscalls or GPU queries.

Public API
----------
CPU_CORES        : int   — logical CPU cores available to the process
GPU_VRAM_MB      : int   — VRAM of the primary GPU in MiB (0 if CPU-only)
GPU_SM_COUNT     : int   — SM count for CUDA; 0 for MPS/CPU
CUDA_AVAILABLE   : bool  — NVIDIA CUDA available
MPS_AVAILABLE    : bool  — Apple MPS available
DEVICE           : str   — "cuda" | "mps" | "cpu"  (best available)
embed_batch_size(model_dim: int) -> int
    Optimal SentenceTransformers batch size for detected hardware.
cpu_worker_count(fraction: float = 0.5) -> int
    Recommended parallel worker count (default: 50% of cores).
"""

from __future__ import annotations

import os

# ── CPU ───────────────────────────────────────────────────────────────────────
CPU_CORES: int = os.cpu_count() or 2

# ── GPU backend detection ─────────────────────────────────────────────────────
CUDA_AVAILABLE: bool = False
MPS_AVAILABLE:  bool = False
GPU_VRAM_MB:    int  = 0
GPU_SM_COUNT:   int  = 0
DEVICE:         str  = "cpu"

try:
    import torch as _torch

    # Priority 1 — NVIDIA CUDA (Linux, Windows, WSL)
    if _torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE = "cuda"
        _props = _torch.cuda.get_device_properties(0)
        GPU_VRAM_MB   = _props.total_memory // (1024 * 1024)
        GPU_SM_COUNT  = _props.multi_processor_count

    # Priority 2 — Apple MPS (macOS Apple Silicon only)
    # Apple Silicon uses unified memory — the GPU and CPU share the same physical
    # RAM pool. There is no separate VRAM; the full system RAM is the GPU budget.
    # We use 75% to leave headroom for the OS, kernel drivers, and the process heap.
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEVICE = "mps"
        try:
            import psutil as _psutil
            _ram_mb = _psutil.virtual_memory().total // (1024 * 1024)
            GPU_VRAM_MB = int(_ram_mb * 0.75)
        except ImportError:
            GPU_VRAM_MB = 16384  # safe default for M-series without psutil

except Exception:
    pass  # torch absent — stay at CPU defaults

# ── MLX detection (Apple MLX framework — faster than torch MPS for inference) ─
MLX_AVAILABLE: bool = False
try:
    import mlx.core as _mlx_core  # type: ignore  # noqa: F401
    from mlx_embeddings.utils import load as _mlx_load  # type: ignore  # noqa: F401
    MLX_AVAILABLE = True
except ImportError:
    pass


def embed_batch_size(model_dim: int = 384) -> int:
    """Return optimal encode batch_size for the detected hardware.

    Targets 50 % of free GPU memory for activations (CUDA/MPS), or a fixed
    512 MB budget for CPU.  Result is rounded down to the nearest power-of-two
    for aligned kernel launches.

    Formula chain (bytes per sample)
    ---------------------------------
    bytes_per_sample = ACTIVATION_FACTOR × seq_len × model_dim × float32_bytes
                     = 12  ×  256  ×  model_dim  ×  4

    Where:
      seq_len           = 256   (default SentenceTransformers max sequence length)
      float32_bytes     = 4     (fp32 element size in bytes)
      ACTIVATION_FACTOR = 12    — accounts for all tensors held live during a
                                  forward pass under torch.no_grad():
                                    Q, K, V projections      (3×)
                                    Attention score matrix   (1×)
                                    FFN up-projection        (~4× model_dim)
                                    FFN down-projection      (~4× model_dim)
                                  The factor rolls up per-layer overhead so the
                                  estimate stays conservative across architectures.

    Budget selection
    ----------------
    CUDA : torch.cuda.mem_get_info()[0]  ×  0.5
           (free VRAM bytes queried after model weights are loaded)
    MPS  : GPU_VRAM_MB  ×  1 024 × 1 024  ×  0.5
           (GPU_VRAM_MB was set to 75 % of total unified RAM at import time)
    CPU  : fixed 512 MB budget (512 × 1 024 × 1 024 bytes)

    Output clamps
    -------------
    GPU (CUDA/MPS) : [64,  2048]
    CPU            : [32,   512]

    Caller
    ------
    Called in benchmark/memory/strategies/embeddings_strategy.py __init__()
    at approximately line 178, AFTER the SentenceTransformer model is loaded,
    so CUDA free memory already reflects the actual weight footprint.
    """
    # 12× multiplier accounts for multi-head attention scores (heads × seq × seq),
    # FFN intermediate (4 × dim), and hidden states kept across layers under
    # no_grad(). Undercounting causes OOM; overcounting wastes nothing.
    ACTIVATION_FACTOR = 12
    bytes_per_sample = 256 * model_dim * 4 * ACTIVATION_FACTOR

    if DEVICE in ("cuda", "mps") and GPU_VRAM_MB > 0:
        # For CUDA: prefer free memory over total memory so model weights already
        # loaded don't inflate the budget. Fall back to 50% of total when unavailable.
        if CUDA_AVAILABLE:
            try:
                import torch as _t
                free_bytes, _ = _t.cuda.mem_get_info()
                budget_bytes = int(free_bytes * 0.5)
            except Exception:
                budget_bytes = int(GPU_VRAM_MB * 1024 * 1024 * 0.50)
        else:
            # MPS (Apple Silicon): psutil total is unified memory; use 50%.
            budget_bytes = int(GPU_VRAM_MB * 1024 * 1024 * 0.50)
        raw  = budget_bytes // bytes_per_sample
        size = max(64, min(2048, _floor_pow2(raw)))
    else:
        budget_bytes = 512 * 1024 * 1024  # 512 MB RAM target
        raw  = budget_bytes // bytes_per_sample
        size = max(32, min(512, _floor_pow2(raw)))

    return size


def cpu_worker_count(fraction: float = 1.0) -> int:
    """Return recommended parallel worker count.

    Defaults to cpu_count - 1 (all cores except one for the OS scheduler).
    BM25/recency cells are CPU-bound and do not use the GPU, so there is no
    reason to reserve cores for GPU headroom on those phases.
    Minimum 1, maximum CPU_CORES - 1.
    """
    if fraction >= 1.0:
        return max(1, CPU_CORES - 1)
    return max(1, min(CPU_CORES - 1, int(CPU_CORES * fraction)))


def _floor_pow2(n: int) -> int:
    """Return the largest power of two ≤ n."""
    if n <= 0:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ── Human-readable summary (shown when BENCHMARK_HW_DEBUG=1) ──────────────────
if os.environ.get("BENCHMARK_HW_DEBUG"):
    _vram_str = f"{GPU_VRAM_MB} MiB" if GPU_VRAM_MB else "shared/unknown"
    _sm_str   = f", {GPU_SM_COUNT} SMs" if GPU_SM_COUNT else ""
    print(
        f"[hw_probe] device={DEVICE}  VRAM={_vram_str}{_sm_str}  "
        f"MLX={MLX_AVAILABLE}  "
        f"CPU_CORES={CPU_CORES}  "
        f"batch(384)={embed_batch_size(384)}  "
        f"batch(768)={embed_batch_size(768)}  "
        f"workers={cpu_worker_count()}"
    )
