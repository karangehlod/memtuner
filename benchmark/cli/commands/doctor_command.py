"""``benchmark doctor`` — system capability check and run recommendation.

Reads actual hardware (CPU cores, RAM, GPU), checks which Python packages
are installed, then prints hardware-specific information:

  - What mode/phases this machine can run
  - Which embedding models fit in VRAM
  - Estimated wall-clock time for each mode
  - The exact command to copy-paste and run immediately

No configuration required. Run this first to know what to expect.
"""

from __future__ import annotations

import os
from pathlib import Path


def _section(title: str) -> None:
    print(f"\n  {'─' * 56}")
    print(f"  {title}")
    print(f"  {'─' * 56}")


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✓  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ⚠  {label}{suffix}")


def _info(label: str, detail: str = "") -> None:
    suffix = f"  — {detail}" if detail else ""
    print(f"     {label}{suffix}")


def _check_import(module: str) -> bool:
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _write_env(config: dict[str, str], env_path: Path) -> None:
    """Write or update benchmark keys in .env without touching user-set values.

    Rules:
    - Lines for keys in `config` are replaced with new values.
    - Existing lines for other keys are preserved unchanged.
    - New keys are appended at the end under a doctor-managed block.
    - Keys set to empty string are removed (uncommented or deleted).
    """
    MANAGED_HEADER = "# --- auto-configured by benchmark doctor ---"
    MANAGED_FOOTER = "# --- end doctor config ---"

    # Read existing file
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    # Remove previous doctor-managed block entirely
    cleaned: list[str] = []
    in_managed = False
    for line in existing_lines:
        if MANAGED_HEADER in line:
            in_managed = True
            continue
        if MANAGED_FOOTER in line:
            in_managed = False
            continue
        if not in_managed:
            cleaned.append(line)

    # Build new managed block
    managed: list[str] = []
    for key, value in config.items():
        if value:  # empty value = remove
            managed.append(f"{key}={value}")

    # Write back
    result = cleaned[:]
    # Strip trailing blank lines before appending
    while result and result[-1].strip() == "":
        result.pop()

    if managed:
        result += ["", MANAGED_HEADER]
        result += managed
        result += [MANAGED_FOOTER, ""]

    env_path.write_text("\n".join(result) + "\n", encoding="utf-8")


def run_doctor(verbose: bool = False, apply: bool = False) -> None:
    """Run the full system capability check and print hardware analysis.

    If apply=True, writes the hardware-derived configuration directly to .env
    so every subsequent run automatically uses the correct settings without flags.
    """

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║             MemTuner — Doctor                        ║")
    print("  ║     Hardware analysis + ready-to-run commands         ║")
    print("  ╚══════════════════════════════════════════════════════╝")

    # ── 1. Hardware detection ─────────────────────────────────────────────────
    _section("Hardware")

    cpu_cores = os.cpu_count() or 2
    _ok(f"CPU: {cpu_cores} logical cores")

    ram_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_gb = vm.total / 1024**3
        ram_avail_gb = vm.available / 1024**3
        _ok(f"RAM: {ram_gb:.1f} GB total  ({ram_avail_gb:.1f} GB available now)")
    except ImportError:
        _warn("RAM: cannot detect (psutil not installed)")
        ram_gb = 8.0  # conservative fallback

    from benchmark.resources.hw_probe import (
        CUDA_AVAILABLE,
        DEVICE,
        GPU_SM_COUNT,
        GPU_VRAM_MB,
        MPS_AVAILABLE,
    )

    if CUDA_AVAILABLE:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            _ok(f"GPU: NVIDIA CUDA — {gpu_name}  ({GPU_VRAM_MB} MB VRAM, {GPU_SM_COUNT} SMs)")
        except Exception:
            _ok(f"GPU: NVIDIA CUDA  ({GPU_VRAM_MB} MB VRAM)")
    elif MPS_AVAILABLE:
        _ok(f"GPU: Apple MPS (Metal)  (~{GPU_VRAM_MB} MB shared budget)")
    else:
        _warn("GPU: not detected — embeddings will run on CPU")
        _info("Install CUDA PyTorch:  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        _info("Apple Silicon MPS:    pip install --upgrade torch  (MPS included in PyTorch ≥ 2.0)")

    # ── 2. Dependency check ───────────────────────────────────────────────────
    _section("Dependencies")

    has_bm25  = _check_import("rank_bm25")
    has_st    = _check_import("sentence_transformers")
    has_torch = _check_import("torch")
    _check_import("numpy")
    has_mpl   = _check_import("matplotlib")
    has_httpx = _check_import("httpx")
    _check_import("psutil")

    if has_bm25:
        _ok("rank-bm25", "Phase 1 (BM25 baseline) available")
    else:
        _warn("rank-bm25 missing", "pip install rank-bm25")

    if has_st and has_torch:
        _ok("sentence-transformers + torch", "Phases 2–4 (embedding/reranker) available")
    elif has_torch and not has_st:
        _warn("sentence-transformers missing", "pip install sentence-transformers")
    elif not has_torch:
        _warn("torch missing", "see GPU section above for install command")

    if has_mpl:
        _ok("matplotlib", "chart generation enabled")
    else:
        _warn("matplotlib missing — charts disabled")
        _info("Fix: pip install matplotlib")

    if has_httpx:
        _ok("httpx", "Ollama LLM judge available")
    else:
        _info("httpx not installed — LLM judge disabled (optional)")

    # ── 3. Model VRAM fit ─────────────────────────────────────────────────────
    _section("Model VRAM Fit")

    models = [
        ("all-MiniLM-L6-v2",          90,    "384-dim, fast baseline"),
        ("BAAI/bge-base-en-v1.5",      210,   "768-dim, balanced English"),
        ("BAAI/bge-m3",                1100,  "1024-dim, multilingual"),
        ("Qwen/Qwen3-Embedding-0.6B",  1200,  "1024-dim, reasoning tasks"),
        ("Qwen/Qwen3-Embedding-4B",    7600,  "2560-dim, highest quality"),
    ]
    rerankers = [
        ("ms-marco-MiniLM-L6-v2",      90,   "fast cross-encoder"),
        ("bge-reranker-base",           210,  "stronger cross-encoder"),
    ]

    gpu_effective = GPU_VRAM_MB if DEVICE != "cpu" else 0
    worst_reranker_mb = max(r[1] for r in rerankers)

    for name, size_mb, note in models:
        short = name.split("/")[-1]
        fits_alone     = gpu_effective == 0 or size_mb <= gpu_effective * 0.85
        fits_w_rerank  = gpu_effective == 0 or (size_mb + worst_reranker_mb) <= gpu_effective * 0.90

        if DEVICE == "cpu":
            _info(f"{short:<35s} {size_mb:5d} MB  — runs on CPU (slow)", note)
        elif fits_w_rerank:
            _ok(f"{short:<35s} {size_mb:5d} MB  fits + reranker", note)
        elif fits_alone:
            _warn(f"{short:<35s} {size_mb:5d} MB  fits alone (no reranker)", note)
        else:
            _warn(f"{short:<35s} {size_mb:5d} MB  exceeds VRAM — skip or use CPU",
                  f"use --skip-models {short}")

    # ── 4. Capability matrix ──────────────────────────────────────────────────
    _section("What This Machine Can Run")

    phases_available = []
    has_gpu = DEVICE != "cpu"

    if has_bm25:
        phases_available.append(1)
        _ok("Phase 1 — BM25 + Recency baseline", "no GPU needed")

    if has_st and has_torch:
        phases_available += [2, 3, 4]
        if has_gpu:
            _ok("Phase 2 — Embedding model sweep", "GPU accelerated")
            _ok("Phase 3 — Hybrid weight sweep",   "GPU accelerated")
            _ok("Phase 4 — Reranker comparison",   "GPU accelerated")
        else:
            _warn("Phase 2–4 — Embedding/reranker", "CPU only — ~10× slower, expect long runtimes")

    if has_bm25:
        phases_available.append(5)
        _ok("Phase 5 — Decay sweep", "CPU + BM25 (no GPU needed for this phase)")

    if not has_bm25:
        _warn("Phase 1 skipped — rank-bm25 not installed")
    if not (has_st and has_torch):
        _warn("Phases 2–4 skipped — sentence-transformers not installed")

    # ── 5. Time estimates ─────────────────────────────────────────────────────
    # ── Build hardware-derived config (needed for time estimates too) ──────────
    workers = max(1, cpu_cores - 1)  # use all cores except 1 for OS

    # Determine which models exceed VRAM and must be skipped
    skip_models: list[str] = []
    if GPU_VRAM_MB > 0 and GPU_VRAM_MB < 10000:
        skip_models.append("Qwen/Qwen3-Embedding-4B")

    _section("Estimated Run Times (LoCoMo dataset, 1,977 queries)")

    # BM25 scales well with workers; embedding is sequential (GPU)
    bm25_time_s = max(30, int(200 / workers))
    # Time per embedding model varies significantly by GPU backend:
    #   CUDA (NVIDIA): ~90s  — batched CUDA kernels, optimal
    #   MPS  (Apple):  ~300s — Metal kernels, ~3x slower than CUDA for this workload
    #   CPU:           ~1800s — ~20x slower than CUDA
    if CUDA_AVAILABLE:
        embed_time_per_model_s = 90
        reranker_time_s = 60
        decay_time_s = 120
    elif MPS_AVAILABLE:
        embed_time_per_model_s = 300
        reranker_time_s = 180
        decay_time_s = 200
    else:
        embed_time_per_model_s = 1800
        reranker_time_s = 600
        decay_time_s = 300

    # Count only models that will actually run (skip_models excluded)
    from benchmark.workload.study_matrix import EMBEDDING_MODELS_LOCAL
    n_local_models = max(1, len([m for m in EMBEDDING_MODELS_LOCAL
                                  if m.split("/")[-1] not in skip_models
                                  and m not in skip_models]))

    times = {
        "quick  (Phase 1 only, BM25 + Recency)":         bm25_time_s,
        "default (Phases 1–3, adds embeddings + hybrid)": bm25_time_s + n_local_models * embed_time_per_model_s + 60,
        "full   (all 5 phases)":                          bm25_time_s + n_local_models * embed_time_per_model_s + reranker_time_s + decay_time_s,
    }
    for label, secs in times.items():
        mins = secs // 60
        if mins < 2:
            t_str = f"~{secs}s"
        elif mins < 60:
            t_str = f"~{mins} min"
        else:
            t_str = f"~{mins // 60}h {mins % 60}m"
        _info(f"--mode {label}", t_str)

    # env var config derived from hardware (no editorial judgment — pure measurement)
    env_config: dict[str, str] = {
        "BENCHMARK_WORKERS":     str(workers),
        "BENCHMARK_SKIP_MODELS": " ".join(skip_models) if skip_models else "",
    }

    # ── 6. Commands / Apply ───────────────────────────────────────────────────
    if not has_bm25:
        print()
        print("  ⚠  Install required packages first, then re-run doctor:")
        print("     pip install rank-bm25")
        if not has_st:
            print("     pip install sentence-transformers")
        return

    if apply:
        # Write config to .env and tell the user what changed
        env_path = Path(__file__).resolve().parents[3] / ".env"
        _write_env(env_config, env_path)

        _section("Configuration written to .env")
        _ok(f".env updated: {env_path}")
        print()

        for key, value in env_config.items():
            if value:
                print(f"  {key}={value}")
            else:
                print(f"  {key}  (removed — no restriction for this hardware)")

        print()
        print("  These settings take effect on every subsequent run automatically.")
        print("  You no longer need to pass --workers or --skip-models flags.")
        print()
        print("  Run the benchmark now (settings from .env apply automatically):")
        if has_st and has_torch:
            print("    python study_runner.py --mode full")
        else:
            print("    python study_runner.py --mode full --phases 1 5")

    else:
        _section("Commands for This Machine — Copy & Run")
        print()
        print("  These commands are tailored to your hardware.")
        print("  To apply this config automatically to every run:")
        print("    python study_runner.py --doctor --apply")
        print("    benchmark doctor --apply")
        print()

        def _cmd(label: str, cmd: str) -> None:
            print(f"  # {label}")
            print(f"  {cmd}")
            print()

        ds_arg = "--gold-dataset data/input/locomo10.json"

        # Always available
        _cmd("Sanity check — BM25 + Recency only, ~45s, no GPU needed",
             f"python study_runner.py {ds_arg} --mode quick --workers {workers}")

        if has_st and has_torch:
            skip_arg = f"--skip-models {' '.join(skip_models)}" if skip_models else ""
            full_cmd = f"python study_runner.py {ds_arg} --mode full --workers {workers} {skip_arg}".strip()
            _cmd("Full study — all phases", full_cmd)
            _cmd("Statistical run — 3 seeds for 95% bootstrap CIs (3× longer)",
                 f"python study_runner.py {ds_arg} --mode full --seeds 42 123 456 --workers {workers} {skip_arg}".strip())
        else:
            _cmd("Full study — BM25 + decay only (no embedding models)",
                 f"python study_runner.py {ds_arg} --mode full --phases 1 5 --workers {workers}")
        print()

    print(f"  {'─' * 56}")
    print("  Run with --apply to write this config to .env automatically.")
    print(f"  {'─' * 56}")
    print()
