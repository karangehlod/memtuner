"""Comprehensive Study Runner — LoCoMo (and other datasets) memory benchmark.

Runs a structured study across four phases, each isolating one dimension:

  Phase 1 — BM25 baseline (no embedding model required)
  Phase 2 — Semantic embedding model comparison (local + Ollama)
  Phase 3 — Hybrid BM25 weight sweep (keyword vs semantic balance)
  Phase 4 — Reranker comparison (cross-encoder models)
  Phase 5 — Decay × lambda sweep on the winning strategy

After each phase the study uses the best result to seed the next phase,
so later phases automatically build on the winner of the prior phase.

Usage:
    python study_runner.py --gold-dataset data/input/locomo10.json
    python study_runner.py --gold-dataset data/input/locomo10.json --phases 1 2 3
    python study_runner.py --gold-dataset data/input/locomo10.json --mode quick
    python study_runner.py --gold-dataset data/input/locomo10.json --mode full \\
        --ollama-url http://localhost:11434/v1

Outputs (inside --output-dir):
    study_<run_id>/
        study_*_summary.json      — machine-readable full summary + recommendations
        study_*_grid.csv          — flat table, one row per cell
        study_*_report.txt        — human-readable ranked report
        phase1_bm25_baseline.png
        phase2_embedding_comparison.png
        phase3_hybrid_weight.png
        phase4_reranker_comparison.png
        phase5_decay_heatmap.png
        phase6_leaderboard.png
        study_report.png           — all panels combined
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

# Invalidate stale .pyc bytecode for the benchmark package on every run.
# On Windows, copying files (e.g. via zip extract) preserves the source mtime
# but not the pyc mtime, so Python may silently run old bytecode that doesn't
# reflect recent source changes (e.g. EMBEDDING_MODELS_OLLAMA = []).
def _purge_pycache(root: str) -> None:
    import glob, os as _os
    for pyc in glob.glob(f"{root}/**/__pycache__/*.pyc", recursive=True):
        try:
            _os.remove(pyc)
        except OSError:
            pass

_purge_pycache(str(Path(__file__).parent / "benchmark"))

# ── CPU thread cap ──────────────────────────────────────────────────────────
# HuggingFace tokenizers and numpy/OpenBLAS use ALL cores by default, pinning
# CPU at 100% even when the GPU is doing the heavy work.  Cap each library at
# half of available cores so Ollama and the OS retain breathing room.
# Must happen before any torch / sentence-transformers / numpy import.
_cpu_cores = os.cpu_count() or 4
_thread_cap = str(max(1, _cpu_cores // 2))
os.environ.setdefault("OMP_NUM_THREADS", _thread_cap)
os.environ.setdefault("MKL_NUM_THREADS", _thread_cap)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _thread_cap)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # disables HF tokenizer fork warning + excess threads


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic Memory Benchmark — Comprehensive Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
First time? Run the system check first:
  python study_runner.py --doctor
  benchmark doctor              (if installed via pip install -e .)

This reads your CPU, RAM, and GPU, then prints the exact command
to run for your hardware — no configuration needed.

Modes:
  quick   — BM25 baseline only (phase 1). Fast sanity check.
  default — Phases 1–3: baseline + embedding models + hybrid sweep.
  full    — All 5 phases including reranker and decay sweep.
  custom  — Specify --phases explicitly.

Study phases:
  1  BM25 baseline         (no model required)
  2  Embedding comparison  (local: MiniLM, bge-base; Ollama if --ollama-url set)
  3  Hybrid weight sweep   (bm25_weight ∈ {0.2, 0.5, 0.8})
  4  Reranker comparison   (n-gram baseline vs cross-encoder models)
  5  Decay × lambda sweep  (exponential / logarithmic / linear × 6 lambda values)

Dataset behaviour:
  No --gold-dataset  → auto-downloads all available datasets, runs all, merges results
  One path           → runs only that dataset, writes one report
  Multiple paths     → runs each, then merges into one unified report

Examples:
  # Run everything — download datasets if needed, benchmark all, merge report
  python study_runner.py --mode full --ollama-url http://localhost:11434/v1

  # Run only LoCoMo
  python study_runner.py --gold-dataset data/input/locomo10.json --mode full \\
      --ollama-url http://localhost:11434/v1

  # Run two specific datasets and get a merged report
  python study_runner.py --gold-dataset data/input/locomo10.json data/input/squad_gold.json --mode full

  # Quick sanity check on LoCoMo only
  python study_runner.py --gold-dataset data/input/locomo10.json --mode quick
""",
    )

    parser.add_argument(
        "--gold-dataset",
        nargs="+",
        default=None,
        help=(
            "One or more gold dataset paths. "
            "If omitted, all available datasets are auto-downloaded and run, "
            "then results are merged into one unified report. "
            "Examples:\n"
            "  data/input/locomo10.json                    (single dataset)\n"
            "  data/input/locomo10.json data/input/squad_gold.json  (specific pair)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Parent directory for study results (default: data/output)",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "default", "full", "custom"],
        default="default",
        help="Study mode (default: default = phases 1-3)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Specific phases to run (overrides --mode)",
    )
    parser.add_argument(
        "--memory-types",
        nargs="+",
        default=["episodic", "semantic", "preference"],
        help="Memory types to include (default: episodic semantic preference)",
    )
    parser.add_argument(
        "--workload",
        default="medium_qpd",
        choices=["low_qpd", "medium_qpd", "high_qpd"],
        help="Workload profile (default: medium_qpd)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: cpu_count // 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Run each cell multiple times with different seeds for statistical significance. "
            "Results are pooled and 95%% bootstrap CIs are reported. "
            "Example: --seeds 42 123 456   (3 runs per cell, ~3x wall time). "
            "Omit for a single run (default)."
        ),
    )
    parser.add_argument(
        "--evaluation-horizon",
        type=int,
        default=None,
        help="Override evaluation horizon (number of dataset days to process)",
    )
    parser.add_argument(
        "--ollama-url",
        default="",
        help="Ollama base URL — used only for the LLM judge (not embeddings). "
             "Example: http://localhost:11434/v1",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Phase 5 early-stopping patience (default: 3). "
            "If composite score does not improve for N consecutive λ steps on a "
            "given decay policy, remaining steps are skipped. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help=(
            "Ollama model to use as LLM judge for answer quality evaluation. "
            "Recommended: nvidia/nemotron-mini-4b-instruct or gemma3:4b. "
            "Requires --ollama-url. When set, each query is scored 1-5 by the judge "
            "after retrieval — adds ~2-5s per query but measures real answer quality, "
            "not just retrieval ID matching. Omit to skip judge evaluation."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization (matplotlib not required)",
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        default=[],
        metavar="MODEL",
        help=(
            "Embedding models to exclude from Phase 2. "
            "Use the short name (last path component) or full HuggingFace ID. "
            "Example: --skip-models Qwen3-Embedding-4B bge-m3 "
            "(removes large/slow models you don't want to wait for)"
        ),
    )
    parser.add_argument(
        "--skip-rerankers",
        nargs="+",
        default=[],
        metavar="RERANKER",
        help=(
            "Reranker models to exclude from Phase 4. "
            "Example: --skip-rerankers bge-reranker-base "
            "(keeps only ms-marco-MiniLM and none)"
        ),
    )
    parser.add_argument(
        "--only-models",
        nargs="+",
        default=[],
        metavar="MODEL",
        help=(
            "Run Phase 2 with ONLY these embedding models (whitelist). "
            "Example: --only-models all-MiniLM-L6-v2 bge-base-en-v1.5 "
            "(fastest Phase 2 run)"
        ),
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help=(
            "Run hardware capability check. Shows what this machine can run "
            "and the exact command to use. Use with --apply to write config to .env."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Used with --doctor: write hardware-derived settings (BENCHMARK_WORKERS, "
            "BENCHMARK_SKIP_MODELS) to .env so every subsequent run uses them automatically."
        ),
    )
    parser.add_argument(
        "--study-config",
        default="configs/study_defaults.yaml",
        metavar="YAML",
        help=(
            "YAML file controlling model lists and benchmark settings. "
            "Edit this instead of touching code to change which models are tested. "
            "Default: configs/study_defaults.yaml"
        ),
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="CSV",
        help=(
            "Merge one or more previously written grid CSVs into a single unified report. "
            "Example: --merge data/output/study_A/study_*_grid.csv data/output/study_B/study_*_grid.csv"
        ),
    )

    args = parser.parse_args()

    # ── Merge mode: combine existing CSVs and exit ───────────────────────────
    if getattr(args, "doctor", False):
        from benchmark.cli.commands.doctor_command import run_doctor
        run_doctor(verbose=True, apply=getattr(args, "apply", False))
        return

    if args.merge:
        _run_merge(args)
        return

    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Load .env if present
    _load_dotenv(project_root)

    # ── Load study config YAML (model lists, benchmark settings) ────────────────
    _study_cfg: dict = {}
    _study_config_path = getattr(args, "study_config", "configs/study_defaults.yaml") or ""
    if _study_config_path:
        import yaml as _yaml
        _cfg_file = Path(project_root) / _study_config_path
        if _cfg_file.exists():
            with open(_cfg_file) as _f:
                _study_cfg = _yaml.safe_load(_f) or {}
        else:
            print(f"  [warn] --study-config not found: {_cfg_file}  (using code defaults)")

    # Attach YAML-driven model lists to args so run_study_for_dataset can read them.
    # CLI flags (--skip-models, --only-models) always override YAML.
    _embed_cfg = _study_cfg.get("embedding", {})
    args._yaml_local_models = _embed_cfg.get("local_models", None)
    args._yaml_api_models   = _embed_cfg.get("api_models", [])
    args._yaml_skip_models  = _embed_cfg.get("skip", [])
    args._yaml_rerankers    = _study_cfg.get("rerankers", None)
    _bench_cfg = _study_cfg.get("benchmark", {})
    if _bench_cfg.get("seed") and args.seed == 42:
        args.seed = int(_bench_cfg["seed"])

    # ── Apply env-var overrides from .env (written by `doctor --apply`) ────────
    # BENCHMARK_WORKERS and BENCHMARK_SKIP_MODELS are written by the doctor
    # command and take effect here, before any CLI flag defaults are applied.
    # CLI flags always win over env vars — explicit > auto-configured.
    if args.workers is None:
        _env_workers = os.environ.get("BENCHMARK_WORKERS", "").strip()
        if _env_workers and _env_workers.isdigit():
            args.workers = int(_env_workers)

    _env_skip = os.environ.get("BENCHMARK_SKIP_MODELS", "").strip()
    if _env_skip and not getattr(args, "skip_models", []):
        # Parse space-separated model names from env var
        args.skip_models = [m.strip() for m in _env_skip.split() if m.strip()]
        print(f"  [env] BENCHMARK_SKIP_MODELS={_env_skip}")

    # Propagate judge CLI flags into env vars so the study cell worker (which
    # runs in-process) picks them up via benchmark.judge.llm_client.get_judge_config().
    _judge_model = getattr(args, "judge_model", "")
    if _judge_model and args.ollama_url:
        os.environ["BENCHMARK_JUDGE_MODEL"] = _judge_model
        os.environ.setdefault("BENCHMARK_JUDGE_BASE_URL", args.ollama_url)
        os.environ.setdefault("BENCHMARK_LLM_BASE_URL", args.ollama_url)
        print(f"  LLM judge enabled: {_judge_model} @ {args.ollama_url}")

    # ── Resolve dataset list ─────────────────────────────────────────────────
    data_dir = Path(project_root) / "data"

    if args.gold_dataset:
        # Specific dataset(s) requested — use exactly those, no auto-download
        gold_paths = [Path(p) for p in args.gold_dataset]
        missing = [p for p in gold_paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"ERROR: Dataset not found: {p}", file=sys.stderr)
            sys.exit(1)
    else:
        # No dataset specified — auto-download missing ones, then run all available
        print("\nNo dataset specified — auto-discovering and downloading all available datasets...")
        _ensure_all_datasets(data_dir, project_root)

        input_dir = data_dir / "input"
        search_dirs = [input_dir, data_dir] if input_dir.is_dir() else [data_dir]
        gold_paths = sorted(
            p for d in search_dirs
            for p in list(d.glob("locomo*.json")) + list(d.glob("*_gold.json"))
        )
        if not gold_paths:
            print("ERROR: No datasets available under data/ even after download attempt.", file=sys.stderr)
            sys.exit(1)
        print(f"  Found {len(gold_paths)} dataset(s): {[p.name for p in gold_paths]}\n")

    from benchmark.workload.profile import get_profile
    from benchmark.workload.study_matrix import StudyExpander, DEFAULT_EMBEDDING_MODEL
    from benchmark.workload.study_scheduler import StudyScheduler
    from benchmark.workload.study_aggregator import StudyAggregator, StudyReporter

    # Resolve phases
    mode_phases = {
        "quick": [1],
        "default": [1, 2, 3],
        "full": [1, 2, 3, 4, 5],
        "custom": args.phases or [1],
    }
    phases = args.phases or mode_phases[args.mode]

    # ── Multi-dataset: run each, then auto-merge ─────────────────────────────
    if len(gold_paths) > 1:
        _run_multi_dataset(args, gold_paths, phases, project_root)
        return

    gold_path = gold_paths[0]

    profile = get_profile(args.workload)
    evaluation_horizon = args.evaluation_horizon or profile.evaluation_horizon

    run_id = uuid.uuid4().hex[:12]
    output_dir = Path(args.output_dir) / f"study_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Hardware / dependency diagnostics ────────────────────────────────────
    from benchmark.resources.hw_probe import DEVICE as _HW_DEVICE, GPU_VRAM_MB as _HW_VRAM
    try:
        import torch as _t
        _torch_threads = _t.get_num_threads()
    except Exception:
        _torch_threads = "n/a"

    _gpu_line = _describe_gpu(_HW_DEVICE, _HW_VRAM)

    # Check which optional dependencies are present
    _st_ok      = _check_import("sentence_transformers", "sentence-transformers")
    _rank_ok    = _check_import("rank_bm25",             "rank-bm25")
    _httpx_ok   = _check_import("httpx",                 "httpx")

    print(f"\nAgentic Memory Benchmark — Comprehensive Study Runner")
    print(f"{'=' * 60}")
    print(f"  Run ID:         {run_id}")
    print(f"  Mode:           {args.mode}")
    print(f"  Phases:         {phases}")
    print(f"  Gold dataset:   {gold_path} ({gold_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Workload:       {profile.label}")
    print(f"  Evaluation horizon: {evaluation_horizon}")
    print(f"  Memory types:   {args.memory_types}")
    print(f"  Ollama URL:     {args.ollama_url or '(not configured — Ollama judge skipped)'}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Live progress:  {output_dir}/progress_{run_id}.csv")
    print(f"  GPU:            {_gpu_line}")
    _cpu_count = os.cpu_count() or 4
    _max_safe = max(1, _cpu_count - 1)
    _requested_workers = args.workers or _max_safe
    _effective_workers = min(_requested_workers, _max_safe)
    _workers_note = f"  (capped from {_requested_workers})" if _requested_workers > _max_safe else ""
    print(f"  CPU workers:    {_effective_workers} / {_cpu_count} cores{_workers_note}  (OMP/MKL threads capped at {_thread_cap})")
    print(f"  Dependencies:   rank-bm25={'✓' if _rank_ok else '✗ missing (pip install rank-bm25)'}  "
          f"sentence-transformers={'✓' if _st_ok else '⚠ not installed (Phase 2-4 will use CPU fallback)'}  "
          f"httpx={'✓' if _httpx_ok else '⚠ not installed (Ollama judge disabled)'}")

    # Warn before spending time on cells that will immediately fail
    if 2 in phases and not _st_ok:
        print(f"  [warn] Phase 2-4 (embedding/reranker) skipped — sentence-transformers not installed.")
        print(f"         Install: pip install sentence-transformers")
        phases = [p for p in phases if p == 1 or p == 5]
    print()

    # Single-dataset path
    _run_single_dataset(args, gold_path, phases, project_root)


def _print_phase_desc(desc: dict):
    if desc.get("embedding_models"):
        print(f"  Embedding models:   {desc['embedding_models']}")
    if desc.get("embedding_backends"):
        print(f"  Embedding backends: {desc['embedding_backends']}")
    if desc.get("bm25_weights"):
        print(f"  BM25 weights:       {desc['bm25_weights']}")
    if desc.get("reranker_models"):
        print(f"  Reranker models:    {desc['reranker_models']}")


def _dataset_label_from_csv(csv_path: Path) -> str:
    """Derive a short dataset label from a grid CSV path.

    The CSV lives at: data/output/study_<run_id>/study_<ts>_<run_id>_grid.csv
    We reconstruct the dataset name from the gold path stored in the CSV or fall
    back to the parent folder name.
    """
    try:
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                # Look for a dataset name in any column
                for col in ("dataset_name", "dataset", "gold_dataset"):
                    if col in row and row[col]:
                        import re
                        name = row[col].lower()
                        name = re.sub(r".*[/\\]", "", name)
                        name = re.sub(r"(_gold|_oracle|_dev.*)?\.json$", "", name)
                        name = re.sub(r"\d+$", "", name).rstrip("_-")
                        return name or csv_path.parent.name
                break
    except Exception:
        pass
    # Fall back to folder name
    return csv_path.parent.name


def _print_narrative_summary(per_dataset_summaries: dict) -> None:
    """Print a concise winner-per-dataset table to stdout."""
    print()
    print(f"    {'Dataset':<22}  {'Best strategy':<18}  {'Recall@K':>8}  Why it wins")
    print("    " + "─" * 78)
    from benchmark.gold.dataset_profiles import get_profile
    for ds_name, summary in sorted(per_dataset_summaries.items()):
        recs = summary.get("recommendations", {})
        best_strat = recs.get("best_retrieval_strategy", "—")
        strat_ranks = summary.get("retrieval_strategy_ranking", [])
        recall = strat_ranks[0]["avg_recall"] if strat_ranks else 0.0
        profile = get_profile(ds_name)
        why = profile.character if profile else "—"
        print(f"    {ds_name:<22}  {best_strat:<18}  {recall:>8.4f}  {why}")
    print()


def _describe_gpu(device: str, vram_mb: int) -> str:
    """Return a human-readable GPU description for the startup banner."""
    if device == "cuda":
        try:
            import torch as _t
            name = _t.cuda.get_device_name(0)
            return f"✓ CUDA — {name} ({vram_mb} MB VRAM)"
        except Exception:
            return f"✓ CUDA ({vram_mb} MB VRAM)"
    elif device == "mps":
        return f"✓ MPS — Apple Silicon (~{vram_mb} MB shared)"
    else:
        return "○ CPU only — embeddings run on CPU (~10–50× slower than GPU)"


def _check_import(module: str, pip_name: str) -> bool:
    """Return True if a Python module can be imported."""
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _rename_csv_descriptively(csv_path: Path) -> Path:
    """Rename a grid CSV to a human-readable name that encodes dataset + phases.

    Before: study_20260818_032941_b4853b73731c_grid.csv
    After:  squad_11873q_p1-baselines_p2-embeddings_p3-bm25_p4-decay_20260818_b4853b73_grid.csv

    Phase abbreviations and dataset names come from configs/benchmark_config.yaml.
    """
    import csv as _csv
    from datetime import datetime as _dt
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).parent / "scripts"))
        from config import cfg as _cfg
        _PHASE_SHORT = _cfg.phase_abbreviations
        _DS_NAMES    = {k: v.lower().replace(" ", "_") for k, v in _cfg.datasets.query_count_to_name.items()}
        _max_tokens  = _cfg.reporting.max_phase_tokens_in_filename
    except Exception:
        # Fallback if config isn't loadable
        _PHASE_SHORT = {
            "phase1_baselines": "p1-baselines", "phase2_embedding_comparison": "p2-embeddings",
            "phase3_hybrid_broad": "p3-bm25-broad", "phase4_decay_broad": "p4-decay-broad",
            "phase5_reranker_comparison": "p5-reranker",
        }
        _DS_NAMES   = {11873: "squad", 7983: "coqa", 1977: "locomo", 470: "longmemeval", 200: "synthetic"}
        _max_tokens = 4

    try:
        rows: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            return csv_path

        nq      = int(rows[0].get("total_queries", 0))
        ds      = _DS_NAMES.get(nq, f"ds{nq}")
        phases  = sorted({r.get("study_phase", "") for r in rows if r.get("study_phase")})
        p_short = [_PHASE_SHORT.get(p, p.replace("phase", "p")) for p in phases]
        phase_str = "_".join(p_short[:_max_tokens])
        if len(p_short) > _max_tokens:
            phase_str += f"_p{len(p_short)}total"

        date_str  = _dt.now().strftime("%Y%m%d")
        short_id  = csv_path.parent.name.replace("study_", "")[:8]  # first 8 chars of run_id

        new_name = f"{ds}_{nq}q_{phase_str}_{date_str}_{short_id}_grid.csv"
        new_path = csv_path.parent / new_name
        csv_path.rename(new_path)
        return new_path
    except Exception:
        return csv_path  # leave original name on any error


def _trigger_report_generation(project_root: str) -> None:
    """Regenerate master CSV, formula doc, and reports_data.js after a run."""
    try:
        import importlib.util, sys as _sys
        script = Path(project_root) / "scripts" / "generate_reports.py"
        if not script.exists():
            return
        spec = importlib.util.spec_from_file_location("generate_reports", script)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.generate(Path(project_root))
    except Exception as _e:
        print(f"  [warn] Report generation skipped: {_e}")


def _write_leaderboards_json(agg, dataset_label: str, run_id: str) -> "Path | None":
    """Write benchmark_results/leaderboards.json from study_aggregator output.

    This is the canonical output file — always reflects the most recent real run.
    Unlike benchmark/memory/leaderboards.py (which uses score-estimated metrics),
    all values here are gold-grounded: recall, MRR, and nDCG are computed against
    the gold memory IDs from the dataset via ScenarioRunner / GoldOracle.

    Input
    -----
    agg : StudyAggregator
        Populated from real ScenarioRunner gold-grounded results, typically via
        StudyAggregator(all_results) or StudyAggregator.from_csv().

    Output
    ------
    benchmark_results/leaderboards.json  (created if the directory is missing).
    Returns the Path on success, or None if writing fails.

    Schema — top-level keys
    -----------------------
    run_metadata : dict
        run_id        (str)  — hex run identifier
        dataset       (str)  — dataset stem / label
        timestamp     (str)  — ISO-8601 timestamp of write
        gold_grounded (bool) — always True for this writer
        ci_method     (str)  — "bootstrap_percentile_1000"
        ci_level      (float)— 0.95
        note          (str)  — plain-English provenance note

    accuracy_leaderboard : dict
        title   (str)
        metric  (str)
        entries (list) — one dict per retrieval strategy, sorted by avg_recall desc:
            rank               (int)   — 1-based
            retrieval_strategy (str)
            avg_recall_at_k    (float) — from rank_by_retrieval_strategy() key "avg_recall"
            avg_mrr            (float) — from key "avg_mrr"
            avg_ndcg           (float) — from key "avg_ndcg"
            avg_precision      (float) — from key "avg_precision"
            recall_ci_low      (float|None) — lower bound of 95% bootstrap CI
                                              from bootstrap_ci(metric="recall_at_k")
            recall_ci_high     (float|None) — upper bound of 95% bootstrap CI
            mrr_ci_low         (float|None) — lower bound of 95% bootstrap CI for MRR
            mrr_ci_high        (float|None) — upper bound of 95% bootstrap CI for MRR
            sig_vs_next        (bool)  — True when this strategy's recall_ci_low
                                         exceeds the next-lower strategy's recall_ci_high,
                                         i.e. the difference is statistically significant
                                         (from significance_table(), p < 0.05 approx.)
            runs               (int)   — number of seed runs contributing, from key "runs"

    efficiency_leaderboard : dict
        title   (str)
        metric  (str)
        entries (list) — one dict per strategy, sorted by avg_latency_p50_ms asc:
            rank               (int)
            retrieval_strategy (str)
            avg_latency_p50_ms (float) — from rank_by_retrieval_strategy() key "avg_latency_p50_ms"
            recall_per_ms      (float) — avg_recall / max(avg_latency_p50_ms, 0.001);
                                         from key "recall_per_ms" (higher = better tradeoff)

    embedding_model_leaderboard : dict
        title   (str)
        metric  (str)
        entries (list) — one dict per embedding model, sorted by avg_recall desc:
            rank               (int)
            embedding_model    (str)
            embedding_backend  (str|None)
            avg_recall         (float) — from rank_by_embedding_model() key "avg_recall"
            avg_mrr            (float)
            avg_ndcg           (float)
            recall_ci_low      (float|None) — from bootstrap_ci(metric="recall_at_k",
                                              group_by="embedding_model")
            recall_ci_high     (float|None)
            avg_latency_p50_ms (float)
            recall_per_ms      (float)
            runs               (int)

    reranker_leaderboard : dict
        Raw output of StudyAggregator.rank_by_reranker() — a list of dicts
        with at minimum: reranker_model, avg_recall, recall_lift_vs_none.

    per_dataset_breakdown : dict
        Raw output of StudyAggregator.study_summary()["per_dataset"] — keyed
        by dataset name, each value contains best_strategy, best_embedding,
        best_strategy_recall, and recommendations.

    cost_vs_quality : dict
        title   (str)
        entries (list) — strategies sorted by recall_per_ms desc:
            retrieval_strategy (str)
            avg_recall         (float)
            avg_latency_p50_ms (float)
            recall_per_ms      (float)

    top_recommendations : dict
        Raw output of StudyAggregator.study_summary()["top_ranked"] — contains
        top_retrieval_strategy, top_embedding_model, top_embedding_backend,
        top_bm25_weight, top_reranker.

    Key names from rank_by_retrieval_strategy()
    --------------------------------------------
    The function consumes the following keys from each row dict returned by
    StudyAggregator.rank_by_retrieval_strategy():
      avg_recall          — mean Recall@K across all runs for this strategy
      avg_mrr             — mean MRR
      avg_ndcg            — mean nDCG
      avg_latency_p50_ms  — mean p50 query latency in milliseconds
      recall_per_ms       — avg_recall / max(avg_latency_p50_ms, 0.001)
      avg_precision       — mean precision
      runs                — number of seed runs aggregated
      sig_vs_next         — bool from significance_table(); True when
                            ci_low of this strategy > ci_high of next-lower strategy

    CI key names (from bootstrap_ci())
    ------------------------------------
    bootstrap_ci(metric="recall_at_k", group_by="retrieval_strategy") returns a
    dict keyed by strategy name; each value has:
      recall_ci_low   — 2.5th percentile of the bootstrap distribution
      recall_ci_high  — 97.5th percentile of the bootstrap distribution
    (These are also referred to as ci_low / ci_high inside significance_table().)
    """
    from pathlib import Path as _Path
    import json as _json
    from datetime import datetime as _dt

    summary = agg.study_summary()
    strat_ranks = agg.rank_by_retrieval_strategy()
    embed_ranks = agg.rank_by_embedding_model()
    reranker_ranks = agg.rank_by_reranker()
    per_dataset = summary.get("per_dataset", {})

    # Bootstrap CIs across strategies and embedding models
    ci_strategy = agg.bootstrap_ci(metric="recall_at_k", group_by="retrieval_strategy")
    ci_mrr      = agg.bootstrap_ci(metric="mrr",         group_by="retrieval_strategy")
    ci_embed    = agg.bootstrap_ci(metric="recall_at_k", group_by="embedding_model")

    # Accuracy leaderboard — retrieval strategies ranked by recall@K
    accuracy_entries = []
    for rank_i, row in enumerate(strat_ranks, 1):
        strategy = row["retrieval_strategy"]
        ci = ci_strategy.get(strategy, {})
        ci_m = ci_mrr.get(strategy, {})
        accuracy_entries.append({
            "rank": rank_i,
            "retrieval_strategy": strategy,
            "avg_recall_at_k": row["avg_recall"],
            "avg_mrr": row.get("avg_mrr", 0.0),
            "avg_ndcg": row.get("avg_ndcg", 0.0),
            "avg_precision": row.get("avg_precision", 0.0),
            "recall_ci_low":  ci.get("ci_low"),
            "recall_ci_high": ci.get("ci_high"),
            "mrr_ci_low":     ci_m.get("ci_low"),
            "mrr_ci_high":    ci_m.get("ci_high"),
            "sig_vs_next":    row.get("sig_vs_next", False),
            "runs": row.get("runs", 0),
        })

    # Efficiency leaderboard — strategies by latency.
    # rank_by_retrieval_strategy() emits "avg_latency_p50_ms" and "recall_per_ms".
    efficiency_entries = []
    for rank_i, row in enumerate(
        sorted(strat_ranks, key=lambda x: x.get("avg_latency_p50_ms", 9999)), 1
    ):
        lat = row.get("avg_latency_p50_ms", 0.0)
        recall = row.get("avg_recall", 0.0)
        efficiency_entries.append({
            "rank": rank_i,
            "retrieval_strategy": row["retrieval_strategy"],
            "avg_latency_p50_ms": lat,
            "recall_per_ms": recall / max(lat, 0.001),
        })

    # Embedding model leaderboard
    embed_entries = []
    for rank_i, row in enumerate(embed_ranks, 1):
        ci_e = ci_embed.get(row["embedding_model"], {})
        embed_entries.append({
            "rank": rank_i,
            "embedding_model": row["embedding_model"],
            "embedding_backend": row.get("embedding_backend"),
            "avg_recall": row["avg_recall"],
            "avg_mrr": row.get("avg_mrr", 0.0),
            "avg_ndcg": row.get("avg_ndcg", 0.0),
            "recall_ci_low":  ci_e.get("ci_low"),
            "recall_ci_high": ci_e.get("ci_high"),
            "avg_latency_p50_ms": row.get("avg_latency_p50_ms", 0.0),
            "recall_per_ms": (
                row.get("avg_recall", 0.0) / max(row.get("avg_latency_p50_ms", 0.001), 0.001)
            ),
            "runs": row.get("runs", 0),
        })

    # Cost-vs-quality: recall / latency tradeoff per strategy
    cost_entries = [
        {
            "retrieval_strategy": r["retrieval_strategy"],
            "avg_recall": r["avg_recall"],
            "avg_latency_p50_ms": r.get("avg_latency_p50_ms", 0.0),
            "recall_per_ms": (
                r.get("avg_recall", 0.0) / max(r.get("avg_latency_p50_ms", 0.001), 0.001)
            ),
        }
        for r in sorted(
            strat_ranks,
            key=lambda x: x.get("avg_recall", 0.0) / max(x.get("avg_latency_p50_ms", 0.001), 0.001),
            reverse=True,
        )
    ]

    leaderboard = {
        "run_metadata": {
            "run_id": run_id,
            "dataset": dataset_label,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
            "gold_grounded": True,
            "ci_method": "bootstrap_percentile_1000",
            "ci_level": 0.95,
            "note": (
                "Recall, MRR, nDCG computed against gold memory IDs from the dataset. "
                "CIs are non-parametric 95% bootstrap percentile intervals (Sakai 2006). "
                "Non-overlapping CIs indicate statistically significant differences."
            ),
        },
        "accuracy_leaderboard": {
            "title": "Retrieval Strategy Ranking",
            "metric": "Recall@K + MRR + nDCG (gold-grounded)",
            "entries": accuracy_entries,
        },
        "efficiency_leaderboard": {
            "title": "Latency Ranking (lower = faster)",
            "metric": "Mean query latency (ms)",
            "entries": efficiency_entries,
        },
        "embedding_model_leaderboard": {
            "title": "Embedding Model Ranking",
            "metric": "Recall@K (gold-grounded, with 95% CI)",
            "entries": embed_entries,
        },
        "reranker_leaderboard": {
            "title": "Reranker Ranking",
            "metric": "Recall@K lift vs no-reranking",
            "entries": reranker_ranks,
        },
        "per_dataset_breakdown": per_dataset,
        "cost_vs_quality": {
            "title": "Cost-vs-Quality (recall per ms)",
            "entries": cost_entries,
        },
        "top_recommendations": summary.get("top_ranked", {}),
    }

    out_dir = _Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "leaderboards.json"
    with open(out_path, "w", encoding="utf-8") as _f:
        _json.dump(leaderboard, _f, indent=2, default=str)
    print(f"\n  benchmark_results/leaderboards.json updated (gold-grounded, run_id={run_id})")
    return out_path


def _print_summary(agg):
    import os as _os
    _k = int(_os.environ.get("BENCHMARK_RECALL_K", "10"))
    print()
    print(f"  STRATEGY RANKING  (Recall@{_k}):")
    for i, row in enumerate(agg.rank_by_retrieval_strategy(), 1):
        print(f"    {i}. {row['retrieval_strategy']:20s}  "
              f"recall@{_k}={row['avg_recall']:.4f}  composite={row['avg_composite']:.4f}")

    print()
    print("  EMBEDDING MODEL RANKING:")
    for i, row in enumerate(agg.rank_by_embedding_model(), 1):
        print(f"    {i}. {row['embedding_model']:35s}  "
              f"recall={row['avg_recall']:.4f}  lat={row['avg_latency_ms']:.1f}ms")

    bw = agg.rank_by_bm25_weight()
    if bw:
        print()
        print("  HYBRID BM25 WEIGHT:")
        for i, row in enumerate(bw, 1):
            print(f"    {i}. bm25_weight={row['bm25_weight']:.2f}  "
                  f"recall={row['avg_recall']:.4f}  mrr={row['avg_mrr']:.4f}")

    rr = agg.rank_by_reranker()
    if rr:
        print()
        print("  RERANKER RANKING:")
        for i, row in enumerate(rr, 1):
            print(f"    {i}. {row['reranker_model']:40s}  "
                  f"recall={row['avg_recall']:.4f}  lift={row['recall_lift_vs_none']:+.4f}")

    summary = agg.study_summary()
    top = summary.get("top_ranked", {})
    print()
    print(f"  TOP-RANKED  (each axis ranked independently from its own phase, Recall@{_k}):")
    print(f"    Retrieval strategy:  {top.get('top_retrieval_strategy', '—')}")
    print(f"    Embedding model:     {top.get('top_embedding_model', '—')}")
    print(f"    Embedding backend:   {top.get('top_embedding_backend', '—')}")
    print(f"    BM25 weight:         {top.get('top_bm25_weight', '—')}")
    print(f"    Reranker:            {top.get('top_reranker', '—')}")

    per_ds = summary.get("per_dataset", {})
    if per_ds:
        print()
        print("  PER-DATASET TOP-RANKED CONFIGURATION:")
        for ds_name, info in sorted(per_ds.items()):
            strat  = info.get("best_strategy") or "—"
            embed  = (info.get("best_embedding") or "—").split("/")[-1]
            recall = info.get("best_strategy_recall", 0.0)
            print(f"    {ds_name:<35s}  strategy={strat:<18s}  "
                  f"embed={embed:<25s}  recall={recall:.4f}")

    # Statistical significance table — only shown when multiple seeds were used.
    # Reports 95% bootstrap CI for Recall@K per strategy.
    n_results = len(agg._study_results) if hasattr(agg, "_study_results") else 0
    n_strategies = len({r.retrieval_strategy for r in agg._study_results}) if hasattr(agg, "_study_results") else 0
    if n_strategies > 1 and n_results >= n_strategies * 2:
        print()
        print("  STATISTICAL SIGNIFICANCE (95% Bootstrap CI, Recall@K):")
        print(f"    {'Strategy':<22s}  {'Mean':>6s}  {'95% CI':>17s}  {'Std':>6s}  {'N':>4s}  Sig")
        print("    " + "-" * 68)
        sig_table = agg.significance_table(metric="recall_at_k", n_bootstrap=1000)
        for row in sig_table:
            sig_marker = "★" if row["sig_vs_next"] else " "
            print(
                f"    {row['group']:<22s}  {row['mean']:>6.4f}  "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
                f"{row['std']:>6.4f}  {row['n']:>4d}  {sig_marker}"
            )
        print("    ★ = CI does not overlap with next-lower group (p < 0.05 approx.)")


def _detect_memory_types(gold_path: Path, requested: list[str]) -> list[str]:
    """Return only the memory types that actually have data in this dataset.

    Benchmarking semantic_store against a dataset whose memories are all episodic
    produces recall=0 by construction — not a retrieval failure, just an empty store.
    This function checks the gold dataset and filters to only non-empty types.
    Falls back to the full requested list if detection fails.
    """
    try:
        from benchmark.gold.oracle import GoldOracle
        oracle = GoldOracle()
        ds = oracle.load_dataset(gold_path, scenario_name="type_detect")
        type_counts: dict[str, int] = {}
        for day in ds.events:
            for event in day.memory_events:
                t = event.type.value if hasattr(event.type, "value") else str(event.type)
                type_counts[t] = type_counts.get(t, 0) + 1

        present = {t for t, n in type_counts.items() if n > 0}
        filtered = [t for t in requested if t in present]

        if not filtered:
            # Dataset has memories but none match requested types — fall back
            return requested

        if set(filtered) != set(requested):
            skipped = [t for t in requested if t not in filtered]
            print(f"  [auto] Skipping {skipped} — no memories of those types in {gold_path.name}")
            print(f"  [auto] Memory type counts: {type_counts}")
            print(f"  [auto] Benchmarking types: {filtered}")
        return filtered
    except Exception:
        return requested


def _ensure_all_datasets(data_dir: Path, project_root: str) -> None:
    """Download and convert any missing auto-downloadable datasets, then generate synthetic."""
    import sys as _sys
    _sys.path.insert(0, str(project_root))

    # Reuse prepare_datasets logic
    scripts_dir = Path(project_root) / "scripts"
    _sys.path.insert(0, str(scripts_dir))
    try:
        import prepare_datasets as _pd
        # Patch DATA_DIR to match project root
        _pd.DATA_DIR = data_dir
        for dest, url, desc in _pd.DOWNLOADS:
            dest = data_dir / dest.relative_to(_pd.project_root / "data")
            if dest.exists():
                print(f"  ✓ Already present: {dest.name}")
                continue
            print(f"  Downloading {desc}...")
            dest.parent.mkdir(parents=True, exist_ok=True)
            _pd._download_file(url, dest, desc)

        for name, src_rel, converter_fn, out_rel in _pd.CONVERSIONS:
            out = data_dir / out_rel.relative_to(_pd.project_root / "data")
            if out.exists():
                print(f"  ✓ Already converted: {out.name}")
                continue
            if src_rel is not None:
                src = data_dir / src_rel.relative_to(_pd.project_root / "data")
                if not src.exists():
                    print(f"  ✗ {name}: source not downloaded, skipping")
                    continue
            print(f"  Converting {name}...")
            try:
                # Temporarily patch DATA_DIR so converters resolve paths correctly
                orig = _pd.DATA_DIR
                _pd.DATA_DIR = data_dir
                result = converter_fn()
                _pd.DATA_DIR = orig
                print(f"  ✓ {name}: {Path(result).name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
    except Exception as e:
        print(f"  [warn] Auto-download failed: {e}")
        print(f"  Run manually: python scripts/prepare_datasets.py --download --convert")


def _run_multi_dataset(args, gold_paths: list, phases: list, project_root: str) -> None:
    """Run benchmark on multiple datasets sequentially, then auto-merge into one report."""
    import glob as _glob
    from datetime import datetime
    from benchmark.workload.study_aggregator import StudyAggregator, StudyReporter

    t0 = time.monotonic()
    n = len(gold_paths)
    written_csvs: list[Path] = []

    print(f"\nAgentic Memory Benchmark — Multi-Dataset Run")
    print(f"{'=' * 60}")
    print(f"  Datasets ({n}):")
    for i, p in enumerate(gold_paths, 1):
        print(f"    {i}. {p.name}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Mode:    {args.mode}  |  Phases: {phases}")
    from benchmark.resources.hw_probe import DEVICE as _d, GPU_VRAM_MB as _v
    print(f"  GPU:     {_describe_gpu(_d, _v)}")
    _st_ok   = _check_import("sentence_transformers", "sentence-transformers")
    _rank_ok = _check_import("rank_bm25", "rank-bm25")
    print(f"  Deps:    rank-bm25={'✓' if _rank_ok else '✗'}  "
          f"sentence-transformers={'✓' if _st_ok else '⚠ not installed — Phases 2-4 skipped'}")
    _judge_info = f"{args.judge_model} @ {args.ollama_url}" if getattr(args, "judge_model", "") and args.ollama_url else "disabled"
    print(f"  Judge:   {_judge_info}")
    print()

    for ds_idx, gold_path in enumerate(gold_paths, 1):
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  DATASET {ds_idx}/{n}: {gold_path.name}")
        print(sep)

        # Patch args temporarily so _run_single_dataset can use it
        args.gold_dataset = [str(gold_path)]
        csv_path = _run_single_dataset(args, gold_path, phases, project_root)
        if csv_path:
            written_csvs.append(csv_path)

        elapsed = time.monotonic() - t0
        remaining_est = (elapsed / ds_idx) * (n - ds_idx)
        print(f"\n  [{ds_idx}/{n} done | elapsed {elapsed:.0f}s | ETA ~{remaining_est:.0f}s remaining]")

    if len(written_csvs) > 1:
        print(f"\n{'=' * 60}")
        print(f"  Auto-merging {len(written_csvs)} dataset runs into unified report...")
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"merged_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        per_dataset_summaries: dict = {}
        for csv_path in written_csvs:
            src_id = csv_path.parent.name
            agg = StudyAggregator.from_csv(csv_path, source_run_id=src_id)
            all_results.extend(agg._results)
            # Derive dataset name from folder name (study_<run_id>)
            ds_label = _dataset_label_from_csv(csv_path)
            per_dataset_summaries[ds_label] = agg.study_summary()

        merged_agg = StudyAggregator(all_results)
        reporter = StudyReporter(output_dir)
        paths = reporter.write_all(
            merged_agg, f"merged_{ts}",
            skip_plots=args.no_plots,
            all_results=all_results,
        )

        # ── Cross-dataset heatmap ──────────────────────────────────────────
        if not args.no_plots and len(per_dataset_summaries) > 1:
            try:
                from benchmark.reporting.study_visualizer import StudyVisualizer
                heatmap_path = output_dir / "cross_dataset_heatmap.png"
                StudyVisualizer.plot_cross_dataset_heatmap(
                    per_dataset_summaries, heatmap_path
                )
                paths["cross_dataset_heatmap"] = str(heatmap_path)
            except Exception as _e:
                paths["viz_error"] = str(_e)

        # ── Narrative report ───────────────────────────────────────────────
        try:
            from benchmark.reporting.narrative_report import NarrativeReportGenerator
            narrative_path = output_dir / "narrative_report.txt"
            NarrativeReportGenerator().generate(per_dataset_summaries, narrative_path)
            paths["narrative_report"] = str(narrative_path)
            print(f"\n  NARRATIVE REPORT — cross-dataset story:")
            # Print a condensed version of the winner matrix
            _print_narrative_summary(per_dataset_summaries)
        except Exception as _e:
            print(f"  [warn] Narrative report: {_e}")

        total_elapsed = time.monotonic() - t0
        print(f"\n  UNIFIED REPORT — {len(all_results)} cells across {len(written_csvs)} datasets")
        print(f"  Total time: {total_elapsed:.1f}s")
        _print_summary(merged_agg)
        print(f"\n  Reports written to: {output_dir}")
        for name, path in sorted(paths.items()):
            if path and name != "viz_error":
                print(f"    {name:20s}: {path}")
        if "viz_error" in paths:
            print(f"  [warn] Visualization: {paths['viz_error']}")
        _trigger_report_generation(project_root)


def _run_single_dataset(args, gold_path: Path, phases: list, project_root: str) -> "Path | None":
    """Run benchmark on one dataset and return path to the written grid CSV."""
    from benchmark.workload.profile import get_profile
    from benchmark.workload.study_matrix import StudyExpander, DEFAULT_EMBEDDING_MODEL
    from benchmark.workload.study_scheduler import StudyScheduler
    from benchmark.workload.study_aggregator import StudyAggregator, StudyReporter

    profile = get_profile(args.workload)
    evaluation_horizon = args.evaluation_horizon or profile.evaluation_horizon

    run_id = uuid.uuid4().hex[:12]
    output_dir = Path(args.output_dir) / f"study_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler = StudyScheduler(max_workers=args.workers, output_dir=str(output_dir), run_id=run_id)

    # Validate that the dataset is parseable before spending time on cell setup.
    # Empty or malformed files (e.g. squad_gold.json at 0 bytes) cause every cell
    # to fail with GoldDatasetError — catch this once and skip the whole dataset.
    try:
        from benchmark.gold.oracle import GoldOracle as _GoldOracle
        _preflight_ds = _GoldOracle().load_dataset(gold_path, scenario_name="preflight")
    except Exception as _e:
        print(f"\n  [skip] {gold_path.name}: cannot load dataset — {_e}")
        print(f"  [skip] Delete {gold_path} and re-run to regenerate it.")
        return None

    # ── Leakage check ────────────────────────────────────────────────────────
    # Detect whether any evaluation query appears verbatim in the memory corpus.
    # > 1% leakage means the benchmark is measuring memorisation, not retrieval.
    try:
        from benchmark.evaluation.leakage_checker import LeakageChecker as _LeakageChecker
        _mem_contents = {
            ev.id: ev.content
            for day in _preflight_ds.events
            for ev in day.memory_events
        }
        _queries = [gq.query for gq in _preflight_ds.queries]
        _leakage = _LeakageChecker(min_overlap_chars=15).check(_queries, _mem_contents)
        _leakage_symbol = "CLEAN" if _leakage.is_clean else "WARNING"
        print(f"  [leakage] {_leakage_symbol}: {_leakage.leaked_count}/{_leakage.total_queries} "
              f"queries overlap corpus ({_leakage.leakage_rate*100:.1f}%)")
        if not _leakage.is_clean:
            print(f"  [leakage] > 1% leakage — results may reflect memorisation, not retrieval quality.")
    except Exception as _le:
        print(f"  [leakage] check skipped: {_le}")

    # Auto-detect which memory types actually exist in this dataset.
    # Running semantic_store/preference_store cells against a dataset that has zero
    # semantic/preference memories produces 0 recall — correct but misleading.
    # Only benchmark the types the dataset actually populated.
    effective_memory_types = _detect_memory_types(gold_path, args.memory_types)

    # ── Model filtering: --skip-models / --only-models / --skip-rerankers ────
    # All model lists come from YAML (configs/study_defaults.yaml).
    # Code has no hardcoded fallback lists — edit the YAML to change models.
    _base_local_models = getattr(args, "_yaml_local_models", None) or []
    _base_api_models   = getattr(args, "_yaml_api_models",   None) or []
    _base_rerankers    = getattr(args, "_yaml_rerankers",    None) or []

    if not _base_local_models and not _base_api_models:
        print(
            "  [warn] No embedding models configured. "
            "Add models to configs/study_defaults.yaml under embedding.local_models "
            "or use --study-config to point at your config file."
        )

    # YAML skip list is pre-merged into the effective skip set (CLI --skip-models appends)
    _yaml_skip  = [m.lower() for m in getattr(args, "_yaml_skip_models", [])]
    _skip_models = list({m.lower() for m in getattr(args, "skip_models", [])} | set(_yaml_skip))
    _only_models    = [m.lower() for m in getattr(args, "only_models", [])]
    _skip_rerankers = [r.lower() for r in getattr(args, "skip_rerankers", [])]

    def _filter_models(models: list[str]) -> list[str]:
        """Apply --skip-models and --only-models filters."""
        if _only_models:
            return [m for m in models if m.lower() in _only_models
                    or m.split("/")[-1].lower() in _only_models]
        if _skip_models:
            return [m for m in models if m.lower() not in _skip_models
                    and m.split("/")[-1].lower() not in _skip_models]
        return models

    def _filter_rerankers(rerankers: list[str]) -> list[str]:
        if _skip_rerankers:
            return [r for r in rerankers if r.lower() not in _skip_rerankers
                    and r.split("/")[-1].lower() not in _skip_rerankers]
        return rerankers

    effective_local_models  = _filter_models(_base_local_models)

    # api_models require BENCHMARK_OPENAI_BASE_URL — skip early with a clear message rather
    # than letting each cell fail individually with a RuntimeError.
    _api_base_url = os.environ.get("BENCHMARK_OPENAI_BASE_URL", "").strip()
    _raw_api_models = _filter_models(_base_api_models)
    if _raw_api_models and not _api_base_url:
        print(
            f"  [skip] api_models ({_raw_api_models}) skipped — "
            "BENCHMARK_OPENAI_BASE_URL not set in .env. "
            "Set it to https://api.openai.com/v1 or http://localhost:11434/v1 to enable."
        )
        effective_api_models = []
    else:
        effective_api_models = _raw_api_models

    effective_ollama_models = []   # Ollama connects via api_models + BENCHMARK_OPENAI_BASE_URL
    effective_rerankers     = _filter_rerankers(_base_rerankers)

    # CrossEncoder reranking requires CUDA — MPS and CPU hang on macOS ARM.
    # Auto-remove neural rerankers when CUDA is unavailable; keep "none" baseline.
    try:
        from benchmark.memory.strategies.llm_rerank_strategy import _CE_AVAILABLE as _ce_ok
        from benchmark.resources.hw_probe import DEVICE as _ce_device
    except Exception:
        _ce_ok, _ce_device = False, "unknown"
    if not _ce_ok:
        _neural = [r for r in effective_rerankers if r.lower() != "none"]
        if _neural:
            print(f"  [skip] CrossEncoder rerankers {_neural} require CUDA — not available on {_ce_device}.")
            print(f"         Phase 5 will run the 'none' baseline only.")
            effective_rerankers = [r for r in effective_rerankers if r.lower() == "none"]

    all_effective = effective_local_models + effective_api_models
    if all_effective != _base_local_models + _base_api_models:
        print(f"  [filter] Embedding models: {all_effective}")
    if effective_rerankers != _base_rerankers:
        print(f"  [filter] Reranker models:  {effective_rerankers}")

    # ── Multi-seed support ───────────────────────────────────────────────────
    seeds = getattr(args, "seeds", None) or [getattr(args, "seed", 42)]
    if len(seeds) > 1:
        print(f"  Statistical mode: {len(seeds)} seeds → {len(seeds)}x cells per phase")
        print(f"  Seeds: {seeds}")

    # Shared expander used for phase headers (seed doesn't affect phase structure)
    expander = StudyExpander(
        memory_types=effective_memory_types,
        ollama_base_url=args.ollama_url or "",
        workload_profile=args.workload,
        seed=getattr(args, "seed", 42),
    )

    all_results = []
    # Phase seeds: updated after each phase from actual results.
    # strategy and embedding_model+backend always come from the same row —
    # never mixed from separate rankings.
    from benchmark.workload.study_matrix import DEFAULT_EMBEDDING_BACKEND
    best_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    best_embedding_backend: str = DEFAULT_EMBEDDING_BACKEND
    best_bm25_weight: float | None = None   # unknown until Phase 3 runs
    best_strategy: str = "bm25"             # valid: Phase 1 sweeps bm25
    t_total = time.monotonic()

    for phase_num in phases:
        _print_phase_header(phase_num, expander, best_embedding_model, best_embedding_backend, best_strategy)

        phase_results = []
        for seed_i, seed_val in enumerate(seeds):
            seed_expander = StudyExpander(
                memory_types=effective_memory_types,
                ollama_base_url=args.ollama_url or "",
                workload_profile=args.workload,
                seed=seed_val,
            )

            if phase_num == 3:
                # Phase 3: broad hybrid weight sweep → find best region → fine zoom
                if len(seeds) > 1:
                    print(f"  Seed {seed_val} ({seed_i+1}/{len(seeds)})")
                seed_results = _run_phase3_two_stage(
                    seed_expander, scheduler,
                    best_embedding_model, best_embedding_backend,
                    gold_path, evaluation_horizon,
                )
                phase_results.extend(seed_results)
                continue

            if phase_num == 4:
                # Phase 4: broad λ sweep → show response curve → fine zoom around optimum
                if len(seeds) > 1:
                    print(f"  Seed {seed_val} ({seed_i+1}/{len(seeds)})")
                if best_bm25_weight is not None:
                    print(f"  Seeding Phase 4 with bm25_weight={best_bm25_weight:.2f} from Phase 3")
                seed_results = _run_phase4_two_stage(
                    seed_expander, scheduler, best_strategy,
                    best_embedding_model, best_embedding_backend,
                    gold_path, evaluation_horizon,
                    best_bm25_weight=best_bm25_weight,
                )
                phase_results.extend(seed_results)
                continue

            cells = _get_phase_cells(
                phase_num, seed_expander,
                best_embedding_model, best_embedding_backend,
                best_bm25_weight, best_strategy,
                local_models=effective_local_models,
                ollama_models=effective_ollama_models,
                api_models=effective_api_models,
                reranker_models=effective_rerankers,
            )
            if not cells:
                print(f"  [skip] No cells generated for phase {phase_num}.")
                continue

            if len(seeds) > 1:
                desc = seed_expander.describe(cells)
                print(f"  Seed {seed_val} ({seed_i+1}/{len(seeds)}): {desc['total_cells']} cells")
            else:
                desc = seed_expander.describe(cells)
                print(f"  Cells: {desc['total_cells']}")
                _print_phase_desc(desc)

            seed_results = scheduler.run(
                cells=cells,
                gold_dataset_path=str(gold_path),
                evaluation_horizon=evaluation_horizon,
            )
            phase_results.extend(seed_results)

        if not phase_results:
            continue
        elapsed = time.monotonic() - t_total
        ok = sum(1 for r in phase_results if r.success)

        try:
            from benchmark.memory.strategies.embeddings_strategy import _INDEX_CACHE
            from benchmark.memory.strategies.bm25_strategy import _BM25_CORPUS_CACHE
            cache_info = (f"  embed-cache={len(_INDEX_CACHE)} entries  "
                          f"bm25-cache={len(_BM25_CORPUS_CACHE)} entries")
        except Exception:
            cache_info = ""

        print(f"\n  Phase {phase_num} done: {ok}/{len(phase_results)} success in {elapsed:.1f}s{cache_info}")

        all_results.extend(phase_results)
        phase_agg = StudyAggregator(all_results, dataset_name=gold_path.stem)
        recs = phase_agg.study_summary().get("recommendations", {})
        best_embedding_model  = recs.get("best_embedding_model",  best_embedding_model)
        best_embedding_backend = recs.get("best_embedding_backend", best_embedding_backend)
        best_bm25_weight = recs.get("best_bm25_weight", best_bm25_weight)
        best_strategy = recs.get("best_retrieval_strategy", best_strategy)

        _bw_str = f"{best_bm25_weight:.2f}" if best_bm25_weight is not None else "—"
        _em_str = best_embedding_model.split('/')[-1] if best_embedding_model else "—"
        _eb_str = best_embedding_backend[:2] if best_embedding_backend else "—"
        print(f"  → Next phase seed: strategy={best_strategy}  "
              f"embed={_em_str}({_eb_str})  "
              f"bm25w={_bw_str}")

    total_elapsed = time.monotonic() - t_total
    print(f"\n{'=' * 60}")
    print(f"  {gold_path.name}: {len(all_results)} cells in {total_elapsed:.1f}s  "
          f"(success: {sum(1 for r in all_results if r.success)})")

    agg = StudyAggregator(all_results, dataset_name=gold_path.stem)
    reporter = StudyReporter(output_dir)
    paths = reporter.write_all(agg, run_id, skip_plots=args.no_plots, all_results=all_results)
    _print_summary(agg)

    # ── Export benchmark_results/leaderboards.json ────────────────────────────
    # Converts study_aggregator output into the canonical leaderboard format so
    # benchmark_results/ always reflects the most recent real run.
    try:
        _lb_path = _write_leaderboards_json(agg, gold_path.stem, run_id)
        if _lb_path:
            paths["leaderboards_json"] = str(_lb_path)
    except Exception as _lb_err:
        print(f"  [warn] leaderboards.json export failed: {_lb_err}")

    print(f"\n  Reports written to: {output_dir}")
    for name, path in sorted(paths.items()):
        if path and name != "viz_error":
            print(f"    {name:20s}: {path}")
    if "viz_error" in paths:
        print(f"  [warn] Visualization failed: {paths['viz_error']}")

    csv_path = paths.get("grid_csv")
    if csv_path:
        csv_path = str(_rename_csv_descriptively(Path(csv_path)))
    _trigger_report_generation(project_root)
    return Path(csv_path) if csv_path else None


def _print_phase_header(phase_num: int, expander, best_embed: str, best_backend: str, best_strategy: str) -> None:
    sep = "─" * 60
    _em = (best_embed or "—").split('/')[-1]
    _eb = (best_backend or "—")[:2]
    short = f"{_em}({_eb})"
    labels = {
        1: "PHASE 1 — BM25 + Recency Baselines",
        2: "PHASE 2 — Semantic Embedding-Model Comparison",
        3: f"PHASE 3 — Hybrid Weight Sweep  (model: {short})",
        4: f"PHASE 4 — Temporal/Decay Sweep  (strategy: {best_strategy}, model: {short})",
        5: f"PHASE 5 — Reranker Comparison  (model: {short})",
    }
    print(f"\n{sep}")
    print(f"  {labels.get(phase_num, f'PHASE {phase_num}')}")
    print(sep)


def _get_phase_cells(
    phase_num: int,
    expander,
    best_embed: str,
    best_backend: str,
    best_bm25w: "float | None",
    best_strategy: str,
    local_models: list | None = None,
    ollama_models: list | None = None,
    api_models: list | None = None,
    reranker_models: list | None = None,
):
    if phase_num == 1:
        return expander.phase_bm25_baseline()
    elif phase_num == 2:
        return expander.phase_embedding_model_comparison(
            local_models=local_models,
            ollama_models=ollama_models,
            api_models=api_models,
        )
    elif phase_num == 3:
        return expander.phase_hybrid_weight_sweep(
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
        )
    elif phase_num == 4:
        return expander.phase_decay_lambda_sweep(
            best_strategy=best_strategy,
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
            bm25_weight=best_bm25w,
        )
    elif phase_num == 5:
        return expander.phase_reranker_comparison(
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
            reranker_models=reranker_models,
        )
    return []


def _response_curve(
    label: str,
    param_name: str,
    points: "list[tuple[float, float]]",
    per_type: "dict[str, list[tuple[float, float]]] | None" = None,
) -> None:
    """Print an ASCII response curve for a parameter sweep.

    points:   list of (param_value, mean_recall_across_types) — the aggregate curve.
    per_type: optional dict of memory_type → [(param_value, recall)] for per-type breakdown.
    """
    if not points:
        return
    recalls = [r for _, r in points]
    lo, hi = min(recalls), max(recalls)
    span = hi - lo or 1e-9
    height = 5
    width = len(points)

    print(f"\n  {label} — response curve ({param_name})  [mean recall across all memory types]")
    # Print rows top→bottom
    for row in range(height, -1, -1):
        threshold = lo + (row / height) * span
        if row == height:
            print(f"  {hi:.4f} |", end="")
        elif row == 0:
            print(f"  {lo:.4f} |", end="")
        else:
            print(f"         |", end="")
        for _, recall in points:
            dot = "●" if recall >= threshold - span / height / 2 else " "
            print(f" {dot}", end="")
        print()
    print("         +" + "──" * width)
    # x-axis labels (every other point to avoid crowding)
    print("          ", end="")
    for i, (pv, _) in enumerate(points):
        if i % 2 == 0:
            print(f"{pv:<4.4g}", end=" ")
        else:
            print("     ", end="")
    print()

    best_pv, best_mean = max(points, key=lambda x: x[1])
    print(f"  Best: {param_name}={best_pv}  mean_recall={best_mean:.4f}  (averaged across memory types)")

    # Per-type breakdown at the best param value — shows the real per-type numbers
    if per_type:
        print(f"  Per-type recall at {param_name}={best_pv}:")
        for mem_type, type_points in sorted(per_type.items()):
            at_best = {round(pv, 4): r for pv, r in type_points}
            r = at_best.get(round(best_pv, 4))
            if r is not None:
                print(f"    {mem_type:15s}  recall={r:.4f}")


def _best_param_from_results(results: list, param_attr: str) -> "float | None":
    """Return the param value of the single highest-recall successful result."""
    ok = [r for r in results if r.success]
    if not ok:
        return None
    best = max(ok, key=lambda r: r.recall_at_k)
    return getattr(best, param_attr, None)


def _run_phase3_two_stage(
    expander,
    scheduler,
    best_embed: str,
    best_backend: str,
    gold_path,
    evaluation_horizon: int,
) -> list:
    """Phase 3: broad BM25-weight sweep (0.0…1.0 step 0.1) → fine zoom.

    Stage 1: 11-point sweep reveals the full BM25/semantic tradeoff curve.
    Stage 2: 9-point 0.05-step zoom ±0.20 around the Stage-1 optimum.
    After each stage the response curve is printed so the researcher can see
    whether the optimum is sharp or flat.
    """
    all_results = []

    # ── Stage 1: broad ───────────────────────────────────────────────────────
    print(f"  Stage 1 (broad): BM25 weight 0.0…1.0 step 0.1")
    broad_cells = expander.phase_hybrid_weight_sweep(
        best_embedding_model=best_embed,
        best_embedding_backend=best_backend,
        stage="broad",
    )
    broad_results = scheduler.run(
        cells=broad_cells,
        gold_dataset_path=str(gold_path),
        evaluation_horizon=evaluation_horizon,
    )
    all_results.extend(broad_results)

    # Aggregate by bm25_weight → mean recall for the response curve
    from collections import defaultdict
    import statistics as _stats
    by_w: dict[float, list[float]] = defaultdict(list)
    by_w_per_type: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in broad_results:
        if r.success:
            w = round(r.bm25_weight, 2)
            by_w[w].append(r.recall_at_k)
            by_w_per_type[r.memory_type][w].append(r.recall_at_k)
    broad_curve = sorted(
        [(w, _stats.mean(recalls)) for w, recalls in by_w.items()],
        key=lambda x: x[0],
    )
    broad_per_type = {
        mt: sorted([(w, _stats.mean(rs)) for w, rs in wmap.items()])
        for mt, wmap in by_w_per_type.items()
    }
    _response_curve("Phase 3 broad", "bm25_weight", broad_curve, per_type=broad_per_type)

    best_broad_w = max(broad_curve, key=lambda x: x[1])[0] if broad_curve else 0.5

    # ── Stage 2: fine zoom ───────────────────────────────────────────────────
    print(f"\n  Stage 2 (fine): ±0.20 around bm25_weight={best_broad_w:.2f} step 0.05")
    fine_cells = expander.phase_hybrid_weight_sweep(
        best_embedding_model=best_embed,
        best_embedding_backend=best_backend,
        stage="fine",
        fine_around=best_broad_w,
    )
    fine_results = scheduler.run(
        cells=fine_cells,
        gold_dataset_path=str(gold_path),
        evaluation_horizon=evaluation_horizon,
    )
    all_results.extend(fine_results)

    by_w2: dict[float, list[float]] = defaultdict(list)
    by_w2_per_type: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in fine_results:
        if r.success:
            w = round(r.bm25_weight, 2)
            by_w2[w].append(r.recall_at_k)
            by_w2_per_type[r.memory_type][w].append(r.recall_at_k)
    fine_curve = sorted(
        [(w, _stats.mean(recalls)) for w, recalls in by_w2.items()],
        key=lambda x: x[0],
    )
    fine_per_type = {
        mt: sorted([(w, _stats.mean(rs)) for w, rs in wmap.items()])
        for mt, wmap in by_w2_per_type.items()
    }
    _response_curve("Phase 3 fine", "bm25_weight", fine_curve, per_type=fine_per_type)

    return all_results


def _run_phase4_two_stage(
    expander,
    scheduler,
    best_strategy: str,
    best_embed: str,
    best_backend: str,
    gold_path,
    evaluation_horizon: int,
    best_bm25_weight: "float | None" = None,
) -> list:
    """Phase 4: broad λ sweep (0.0005…0.10, 8 points) → response curve → fine zoom.

    Stage 1: 8-point log-scale sweep reveals the full decay landscape.
             Includes the no-decay baseline for context.
    Stage 2: 7-point log-scale zoom ±0.25 orders of magnitude around the
             Stage-1 optimum. Skipped if Stage 1 finds that no decay helps.

    Both stages print the response curve so the researcher can see:
      - where the curve peaks (best λ)
      - how sharp/flat the optimum is (sensitivity)
      - whether adding decay helps at all vs no-decay
    """
    from collections import defaultdict
    import statistics as _stats
    all_results = []

    policies = ["exponential", "logarithmic", "linear", "tiered"]

    for policy in policies:
        print(f"\n  [{policy}] Stage 1 (broad): λ ∈ [0.0005, 0.10]")

        broad_cells = expander.phase_decay_lambda_sweep(
            best_strategy=best_strategy,
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
            decay_policies=[policy],
            stage="broad",
            include_no_decay_baseline=True,
            bm25_weight=best_bm25_weight,
        )
        broad_results = scheduler.run(
            cells=broad_cells,
            gold_dataset_path=str(gold_path),
            evaluation_horizon=evaluation_horizon,
        )
        all_results.extend(broad_results)

        # Response curve: λ → mean recall across memory types
        by_lam: dict[float, list[float]] = defaultdict(list)
        by_lam_per_type: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        no_decay_recalls: list[float] = []
        no_decay_per_type: dict[str, list[float]] = defaultdict(list)
        for r in broad_results:
            if r.success:
                lam = r.lambda_value
                if lam == 0.0:
                    no_decay_recalls.append(r.recall_at_k)
                    no_decay_per_type[r.memory_type].append(r.recall_at_k)
                else:
                    by_lam[lam].append(r.recall_at_k)
                    by_lam_per_type[r.memory_type][lam].append(r.recall_at_k)

        broad_curve = sorted(
            [(lam, _stats.mean(recalls)) for lam, recalls in by_lam.items()],
            key=lambda x: x[0],
        )
        broad_per_type = {
            mt: sorted([(lam, _stats.mean(rs)) for lam, rs in lmap.items()])
            for mt, lmap in by_lam_per_type.items()
        }
        _response_curve(f"Phase 4 [{policy}] broad", "lambda", broad_curve, per_type=broad_per_type)

        if no_decay_recalls:
            no_decay_mean = _stats.mean(no_decay_recalls)
            per_type_str = "  ".join(
                f"{mt}={_stats.mean(rs):.4f}" for mt, rs in sorted(no_decay_per_type.items())
            )
            print(f"  No-decay baseline — mean={no_decay_mean:.4f}  [{per_type_str}]")

        if not broad_curve:
            print(f"  [skip] No successful broad results for {policy}")
            continue

        best_broad_lam = max(broad_curve, key=lambda x: x[1])[0]
        best_broad_recall = max(broad_curve, key=lambda x: x[1])[1]

        # Skip fine zoom if decay provides no benefit over no-decay baseline
        if no_decay_recalls and best_broad_recall <= _stats.mean(no_decay_recalls):
            print(f"  [{policy}] Decay does not improve over no-decay — skipping fine zoom.")
            continue

        print(f"\n  [{policy}] Stage 2 (fine): log-zoom around λ={best_broad_lam:.4f}")
        fine_cells = expander.phase_decay_lambda_sweep(
            best_strategy=best_strategy,
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
            decay_policies=[policy],
            stage="fine",
            fine_around=best_broad_lam,
            include_no_decay_baseline=False,
            bm25_weight=best_bm25_weight,
        )
        fine_results = scheduler.run(
            cells=fine_cells,
            gold_dataset_path=str(gold_path),
            evaluation_horizon=evaluation_horizon,
        )
        all_results.extend(fine_results)

        by_lam2: dict[float, list[float]] = defaultdict(list)
        by_lam2_per_type: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in fine_results:
            if r.success and r.lambda_value > 0:
                by_lam2[r.lambda_value].append(r.recall_at_k)
                by_lam2_per_type[r.memory_type][r.lambda_value].append(r.recall_at_k)
        fine_curve = sorted(
            [(lam, _stats.mean(recalls)) for lam, recalls in by_lam2.items()],
            key=lambda x: x[0],
        )
        fine_per_type = {
            mt: sorted([(lam, _stats.mean(rs)) for lam, rs in lmap.items()])
            for mt, lmap in by_lam2_per_type.items()
        }
        _response_curve(f"Phase 4 [{policy}] fine", "lambda", fine_curve, per_type=fine_per_type)

    # ── Phase 4b: Archival floor sweep ───────────────────────────────────────
    # Uses the best policy + best λ from the broad sweep above.
    # Varies archival_floor across [0.0, 0.25, 0.50, 0.65, 0.75, 0.90] to test
    # whether the 0.65 default distorts comparisons of old-memory retrieval.
    ok_results = [r for r in all_results if r.success and r.lambda_value > 0]
    if ok_results:
        best_4b = max(ok_results, key=lambda r: r.recall_at_k)
        best_policy_4b = best_4b.decay_policy
        best_lam_4b = best_4b.lambda_value
        print(f"\n  Phase 4b (archival floor): policy={best_policy_4b} λ={best_lam_4b:.4f}")
        print(f"  Floors: [0.0, 0.25, 0.50, 0.65, 0.75, 0.90]")

        floor_cells = expander.phase_archival_floor_sweep(
            best_strategy=best_strategy,
            best_embedding_model=best_embed,
            best_embedding_backend=best_backend,
            best_policy=best_policy_4b,
            best_lambda=best_lam_4b,
            bm25_weight=best_bm25_weight,
        )
        floor_results = scheduler.run(
            cells=floor_cells,
            gold_dataset_path=str(gold_path),
            evaluation_horizon=evaluation_horizon,
        )
        all_results.extend(floor_results)

        by_floor: dict[float | None, list[float]] = defaultdict(list)
        by_floor_per_type: dict[str, dict[float | None, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in floor_results:
            if r.success:
                fl = getattr(r, "archival_floor", 0.65)
                by_floor[fl].append(r.recall_at_k)
                by_floor_per_type[r.memory_type][fl].append(r.recall_at_k)
        floor_curve = sorted(
            [(f, _stats.mean(recalls)) for f, recalls in by_floor.items() if recalls],
            key=lambda x: (x[0] is None, x[0] or 0.0),
        )
        floor_per_type = {
            mt: sorted([(f, _stats.mean(rs)) for f, rs in fmap.items()],
                       key=lambda x: (x[0] is None, x[0] or 0.0))
            for mt, fmap in by_floor_per_type.items()
        }
        _response_curve("Phase 4b archival_floor", "archival_floor", floor_curve, per_type=floor_per_type)

    return all_results


def _run_merge(args) -> None:
    """Merge multiple grid CSVs into one unified report."""
    import glob as _glob
    from datetime import datetime
    from benchmark.workload.study_aggregator import StudyAggregator, StudyReporter

    # Expand globs (Windows PowerShell doesn't expand them automatically)
    csv_paths: list[Path] = []
    for pattern in args.merge:
        expanded = _glob.glob(pattern)
        if expanded:
            csv_paths.extend(Path(p) for p in expanded)
        else:
            p = Path(pattern)
            if p.exists():
                csv_paths.append(p)

    if not csv_paths:
        print("ERROR: No CSV files found for merging.", file=sys.stderr)
        sys.exit(1)

    print(f"\nMerging {len(csv_paths)} CSV file(s):")
    all_results = []
    for csv_path in csv_paths:
        # Use the parent directory name as the source run ID for provenance
        source_id = csv_path.parent.name
        agg = StudyAggregator.from_csv(csv_path, source_run_id=source_id)
        n = len(agg._results)
        print(f"  {csv_path}  ({n} rows)")
        all_results.extend(agg._results)

    if not all_results:
        print("ERROR: No results loaded from CSVs.", file=sys.stderr)
        sys.exit(1)

    merged_agg = StudyAggregator(all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = args.output_dir if hasattr(args, "output_dir") and args.output_dir else "data/output"
    output_dir = Path(out_base) / f"merged_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"merged_{ts}"
    reporter = StudyReporter(output_dir)
    no_plots = getattr(args, "no_plots", False)
    paths = reporter.write_all(merged_agg, run_id, skip_plots=no_plots, all_results=all_results)

    total = len(all_results)
    success = sum(1 for r in all_results if r.success)
    print(f"\nMerged {total} cells ({success} successful) from {len(csv_paths)} runs")
    _print_summary(merged_agg)
    print(f"\n  Reports written to: {output_dir}")
    for name, path in sorted(paths.items()):
        if path and name != "viz_error":
            print(f"    {name:20s}: {path}")
    if "viz_error" in paths:
        print(f"  [warn] Visualization: {paths['viz_error']}")


def _load_dotenv(project_root: str):
    """Load .env file if present — no external dependency required."""
    env_path = Path(project_root) / ".env"
    if not env_path.exists():
        return
    import os
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required on Windows for ProcessPoolExecutor
    main()
