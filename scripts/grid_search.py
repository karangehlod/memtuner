#!/usr/bin/env python3
"""
MemTuner — Full Grid Search

Runs ALL memory types × ALL retrieval strategies × ALL decay policies on the
SAME dataset. Lambda sweeps from 0.05 → 0.30 in steps of 0.05.

Every cell uses identical data so results are directly comparable.

Usage:
    # Quick test (27 cells, lite dataset, ~5 min)
    python grid_search.py --dataset data/generated/lite/production_gold_dataset.json

    # Full grid on lite dataset (all 4 memory types × 3 strategies × 6 lambda steps)
    python grid_search.py --dataset data/generated/lite/production_gold_dataset.json --mode full

    # Production: 100K queries, all strategies, all types
    python grid_search.py --dataset data/generated/production/production_gold_dataset.json \\
        --mode full --workers 8 --days 90

    # Custom lambda range (default: 0.05 to 0.30 step 0.05)
    python grid_search.py --dataset ... --lambda-min 0.05 --lambda-max 0.30 --lambda-step 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


def build_lambda_steps(lam_min: float, lam_max: float, lam_step: float) -> list[float]:
    """Build lambda list from min/max/step, inclusive of both ends."""
    steps = []
    val = round(lam_min, 10)
    while val <= lam_max + 1e-9:
        steps.append(round(val, 10))
        val = round(val + lam_step, 10)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full grid search — ALL memory types × ALL strategies × ALL decay policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Grid modes:
  quick   — 3 memory types × 3 strategies × 3 decay policies × 1 lambda = 27 cells (~5 min)
  full    — 4 types × 3 strategies × 3 decay policies × N lambda steps (default: 6)
  all     — 4 types × 5 strategies × 5 policies × 6 lambda steps (large)

Lambda sweep (default: 0.05 → 0.30, step 0.05):
  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  Maps to half-lives: [14d, 7d, 5d, 3d, 3d, 2d]

Examples:
  python grid_search.py --dataset data/generated/lite/production_gold_dataset.json
  python grid_search.py --dataset data/generated/lite/production_gold_dataset.json --mode full
  python grid_search.py --dataset data/generated/production/production_gold_dataset.json \\
      --mode full --workers 8 --days 30
""",
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to gold dataset JSON (same for ALL grid cells)")
    parser.add_argument("--output-dir", default="data/output",
                        help="Directory for results (default: data/output)")
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="quick",
                        help="Grid expansion mode (default: quick)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: cpu_count - 1)")
    parser.add_argument("--days", type=int, default=None,
                        help="Simulated days override (default: from workload profile)")
    parser.add_argument("--workload", default="medium_qpd",
                        choices=["low_qpd", "medium_qpd", "high_qpd", "production"],
                        help="Workload profile (default: medium_qpd)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # Lambda range
    parser.add_argument("--lambda-min", type=float, default=0.001,
                        help="Lambda start (default: 0.001)")
    parser.add_argument("--lambda-max", type=float, default=0.10,
                        help="Lambda end inclusive (default: 0.10)")
    parser.add_argument("--lambda-step", type=float, default=0.02,
                        help="Lambda step size (default: 0.02)")
    # Backward-compat flags (ignored, kept so old callers don't error)
    parser.add_argument("--benchmark-dataset", dest="benchmark_dataset_compat",
                        help=argparse.SUPPRESS)
    parser.add_argument("--production", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--all-strategies", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Support old --benchmark-dataset flag
    dataset_path_str = args.dataset or args.benchmark_dataset_compat
    if not dataset_path_str:
        parser.error("--dataset is required")

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmark.workload.matrix import MatrixExpander, LAMBDA_STEPS
    from benchmark.workload.profile import get_profile
    from benchmark.workload.scheduler import MatrixScheduler
    from benchmark.workload.aggregator import MatrixAggregator, MatrixReporter
    from benchmark.recommendation.ranker import MatrixRanker, QualityThresholds

    dataset_path = Path(dataset_path_str)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        print("Generate one with:", file=sys.stderr)
        print("  python generate_production_dataset.py --lite --output-dir data/generated/lite",
              file=sys.stderr)
        sys.exit(1)

    # Build lambda list
    lambda_steps = build_lambda_steps(args.lambda_min, args.lambda_max, args.lambda_step)

    profile = get_profile(args.workload)
    evaluation_horizon = args.days or profile.evaluation_horizon

    run_id = uuid.uuid4().hex[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grid_{ts}_{run_id}"

    print("\nMemTuner — Full Grid Search")
    print("=" * 64)
    print(f"  Run ID:          {run_id}")
    print(f"  Dataset:         {dataset_path}")
    print(f"  Dataset size:    {dataset_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Mode:            {args.mode}")
    print(f"  Workload:        {profile.label}")
    print(f"  Simulated days:  {evaluation_horizon}")
    print(f"  Lambda range:    {lambda_steps[0]} → {lambda_steps[-1]} (step {args.lambda_step})")
    print(f"  Lambda steps:    {lambda_steps}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output dir:      {output_dir}")
    print()
    print("  IMPORTANT: ALL cells run against the SAME dataset.")
    print("  Results are directly comparable across memory types and strategies.")
    print()

    expander = MatrixExpander()

    if args.mode == "quick":
        cells = expander.expand_core_3x3(workload_profile=args.workload, seed=args.seed)
    elif args.mode == "full":
        # SemanticStore is excluded: LoCoMo is an episodic-dialog dataset.
        # SemanticStore only accepts MemoryType.SEMANTIC events — always empty here.
        # Include it via --mode all if your dataset has semantic memory entries.
        cells = expander.expand_full(
            workload_profile=args.workload,
            seed=args.seed,
            memory_types=["episodic", "preference", "entity"],
            strategies=["bm25", "embeddings", "hybrid"],
            decay_policies=["none", "exponential", "linear"],
            lambda_steps=lambda_steps,
        )
    else:  # all
        cells = expander.expand_full(
            workload_profile=args.workload,
            seed=args.seed,
            lambda_steps=lambda_steps,
        )

    desc = expander.describe(cells)
    print(f"  Grid dimensions:")
    print(f"    Memory types:         {desc['memory_types']}")
    print(f"    Retrieval strategies: {desc['retrieval_strategies']}")
    print(f"    Decay policies:       {desc['decay_policies']}")
    print(f"    Lambda values:        {desc['lambda_values']}")
    print(f"    Total cells:          {desc['total_cells']}")
    print()

    # Estimate time (smoke test baseline: ~22s/cell for 3 days)
    est_secs_per_cell = max(10, evaluation_horizon * 7)
    est_total_min = (desc["total_cells"] * est_secs_per_cell) / max(1, args.workers or 1) / 60
    print(f"  Estimated runtime: ~{est_total_min:.0f} min (rough estimate)")
    print()

    scheduler = MatrixScheduler(max_workers=args.workers, output_dir=str(output_dir))

    # Streaming progress log — one JSON line per cell, written immediately on completion.
    # Never holds all results in memory; safe for very long runs.
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / f"progress_{ts}_{run_id}.jsonl"
    _log_file = open(progress_log_path, "w", buffering=1, encoding="utf-8")  # line-buffered

    total_cells = len(cells)
    _t_first_cell = [None]  # track ETA after first cell

    def _progress(completed: int, total: int, result) -> None:
        elapsed_so_far = time.monotonic() - t0
        if completed == 1:
            _t_first_cell[0] = elapsed_so_far
        eta = ""
        if _t_first_cell[0] and completed > 0:
            secs_per_cell = elapsed_so_far / completed
            remaining = (total - completed) * secs_per_cell / max(1, args.workers or 1)
            eta = f"  ETA ~{remaining/60:.1f} min"
        status = "OK" if result.success else "FAIL"
        composite = result.composite_score() if result.success else float("nan")
        line = (
            f"  [{completed:3d}/{total}] {status}  "
            f"{result.memory_type:12s} × {result.retrieval_strategy:10s} "
            f"× {result.decay_policy:12s}  λ={result.lambda_value:.2f}  "
            f"composite={composite:+.4f}{eta}"
        )
        print(line, flush=True)
        # Stream to log file immediately (line-buffered, no memory accumulation)
        log_entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "completed": completed,
            "total": total,
            "status": status,
            "memory_type": result.memory_type,
            "retrieval_strategy": result.retrieval_strategy,
            "decay_policy": result.decay_policy,
            "lambda": result.lambda_value,
            "recall_at_k": round(result.recall_at_k, 4),
            "correct_recalls": result.correct_recalls,
            "total_queries": result.total_queries,
            "precision_at_k": round(result.precision_at_k, 4),
            "noise_ratio": round(result.contamination_rate, 4),
            "composite": round(composite, 4) if result.success else None,
            "mrr": round(result.mrr, 4),
            "ndcg": round(result.ndcg, 4),
            "latency_p50_ms": round(result.latency_p50_ms, 2),
            "latency_p99_ms": round(result.latency_p99_ms, 2),
            "peak_ram_mb": round(result.peak_ram_mb, 1),
            "peak_cpu_pct": round(result.peak_cpu_percent, 1),
            "duration_s": round(result.duration_seconds, 1),
            "error": result.error_message[:200] if result.error_message else None,
        }
        _log_file.write(json.dumps(log_entry) + "\n")

    t0 = time.monotonic()
    results = scheduler.run(
        cells=cells,
        gold_dataset_path=str(dataset_path),
        evaluation_horizon=evaluation_horizon,
        progress_callback=_progress,
    )
    elapsed = time.monotonic() - t0
    _log_file.close()

    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)

    print(f"\n{'=' * 64}")
    print(f"  Completed {len(results)} cells in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"  Successful: {success_count}   Failed: {fail_count}")

    if fail_count > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if not r.success:
                print(f"    [{r.cell_id}] {r.memory_type} × {r.retrieval_strategy} "
                      f"× {r.decay_policy}: {r.error_message[:120]}")

    # ── Invariance checks ────────────────────────────────────────────────
    # These are hard guards against silent wiring bugs.  A wiring bug causes
    # multiple logically distinct configurations to produce identical numbers.
    # ─────────────────────────────────────────────────────────────────────
    successes = [r for r in results if r.success]
    invariance_failures: list[str] = []

    if successes:
        # 1. No two different retrieval strategies should share the exact same
        #    average composite score (rounded to 4 dp) — indicates both hit the
        #    same fallback code path.
        from collections import defaultdict
        strategy_composites: dict[str, list[float]] = defaultdict(list)
        for r in successes:
            strategy_composites[r.retrieval_strategy].append(r.composite_score())

        strategy_avgs = {
            s: round(sum(v) / len(v), 4)
            for s, v in strategy_composites.items()
            if v
        }
        seen_avgs: dict[float, str] = {}
        for strategy, avg in strategy_avgs.items():
            if avg in seen_avgs:
                invariance_failures.append(
                    f"STRATEGY COLLISION: '{strategy}' and '{seen_avgs[avg]}' share "
                    f"identical average composite={avg:.4f}. Both may be hitting the "
                    f"same fallback retrieval path. Check resolved_retriever_class in "
                    f"the results JSON."
                )
            seen_avgs[avg] = strategy

        # 3. Resource metrics should not all be zero for successful cells.
        zero_ram_cells = sum(1 for r in successes if r.peak_ram_mb == 0.0)
        if zero_ram_cells == len(successes):
            invariance_failures.append(
                f"RESOURCE ZERO: peak_ram_mb=0.0 for all {len(successes)} successful "
                f"cells. psutil may not be installed or the tracker is not attached. "
                f"Run: python -m pip install psutil"
            )

        # 4. Log per-cell diagnostic (first 10 cells)
        print(f"\n{'=' * 64}")
        print("  PER-CELL DIAGNOSTICS (resolved_retriever):")
        diag_cells = successes[:10]
        for r in diag_cells:
            print(
                f"    [{r.cell_id}] {r.memory_type:10s} × {r.retrieval_strategy:10s} "
                f"retriever={r.resolved_retriever_class}"
            )
        if len(successes) > 10:
            print(f"    ... ({len(successes) - 10} more cells, see progress JSONL)")

    if invariance_failures:
        print(f"\n{'=' * 64}")
        print("  INVARIANCE CHECK FAILURES:")
        for msg in invariance_failures:
            print(f"    ⚠  {msg}")
        print()
        # Hard exit with non-zero code so CI pipelines catch this
        import sys as _sys
        print("  Benchmark completed but invariance checks failed — see above.")
        # (Don't sys.exit here so reports are still written)
    else:
        print(f"\n  ✓ All invariance checks passed.")
    # ─────────────────────────────────────────────────────────────────────

    # Aggregate and report
    aggregator = MatrixAggregator(results)
    reporter = MatrixReporter(output_dir)
    paths = reporter.write_all(aggregator, run_id)

    # Append timestamp-normalization metadata to the summary JSON so it is
    # recorded alongside best_config and rankings for offline analysis.
    norm_meta = scheduler.normalization_meta
    if norm_meta:
        summary_json_path = Path(paths["summary_json"])
        try:
            with open(summary_json_path) as _f:
                summary_data = json.load(_f)
            summary_data["timestamp_normalization"] = norm_meta
            with open(summary_json_path, "w") as _f:
                json.dump(summary_data, _f, indent=2)
        except OSError:
            pass  # Non-critical — results still written without this block

    # Recommendation
    ranker = MatrixRanker(QualityThresholds())
    best = ranker.best_production_config(results)

    print(f"\n{'=' * 64}")
    print("BEST PRODUCTION CONFIG (meets all quality thresholds):")
    if best:
        print(f"  Memory type:        {best.memory_type}")
        print(f"  Retrieval strategy: {best.retrieval_strategy}")
        print(f"  Decay policy:       {best.decay_policy} (λ={best.lambda_value:.2f})")
        print(f"  Recall@K:           {best.recall_at_k:.4f}")
        print(f"  Noise Ratio:        {best.contamination_rate:.4f}  (fraction of retrieved items that are irrelevant)")
        print(f"  Temporal Accuracy:  {best.temporal_accuracy:.4f}")
        print(f"  Composite Score:    {best.composite_score:.4f}")
        print(f"  Peak RAM:           {best.peak_ram_mb:.1f} MB")
        print(f"  Explanation:        {best.explanation}")
    else:
        print("  No configuration met all quality thresholds.")
        print("  Check the CSV for the closest configs.")

    print(f"\n  Memory type ranking:")
    for row in aggregator.rank_by_memory_type():
        print(f"    {row['memory_type']:12s}  composite={row['avg_composite']:.4f}"
              f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}")

    print(f"\n  Strategy ranking:")
    for row in aggregator.rank_by_retrieval_strategy():
        print(f"    {row['retrieval_strategy']:12s}  composite={row['avg_composite']:.4f}"
              f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}")

    print(f"\n  Decay policy ranking:")
    for row in aggregator.rank_by_decay_policy():
        print(f"    {row['decay_policy']:12s}  composite={row['avg_composite']:.4f}"
              f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}")

    print(f"\n  Reports written to: {output_dir}")
    for name, path in paths.items():
        size_kb = Path(path).stat().st_size / 1024
        print(f"    {name}: {path}  ({size_kb:.1f} KB)")
    prog_kb = progress_log_path.stat().st_size / 1024
    print(f"    progress_log: {progress_log_path}  ({prog_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    # Required for Windows multiprocessing spawn context
    main()
