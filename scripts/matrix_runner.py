"""Matrix benchmark runner — entry point for the full 3D grid.

Usage:
    python matrix_runner.py --mode core3x3 --gold-dataset data/generated/lite/production_gold_dataset.json
    python matrix_runner.py --mode full --gold-dataset data/generated/production/production_gold_dataset.json
    python matrix_runner.py --mode lambda-sweep --memory-type episodic --strategy bm25 --decay exponential

Works on macOS, Linux, and Windows.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MemTuner — Matrix Runner (Memory × Strategy × Decay)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Grid Modes:
  core3x3      — 3×3 core grid: episodic/semantic/preference × bm25/embeddings/hybrid × none/exp/log
                 27 cells × 1 lambda = 27 runs (fastest — good for CI / quick comparison)

  lambda-sweep — Sweep all 7 lambda values for a fixed memory+strategy+decay combination
                 7 runs (shows how lambda affects recall/FPR)

  full         — Full 4×5×5×7 Cartesian grid = 700 combinations (production, takes hours)

  custom       — Provide explicit --memory-types, --strategies, --decay-policies, --lambdas

Examples:
  # Quick core 3x3 comparison (recommended starting point)
  python matrix_runner.py --mode core3x3 \\
    --gold-dataset data/generated/lite/production_gold_dataset.json

  # Lambda sweep: how does decay rate affect episodic + bm25?
  python matrix_runner.py --mode lambda-sweep \\
    --memory-type episodic --strategy bm25 --decay exponential \\
    --gold-dataset data/generated/lite/production_gold_dataset.json

  # Full production matrix
  python matrix_runner.py --mode full \\
    --gold-dataset data/generated/production/production_gold_dataset.json \\
    --workers 8 --workload high_qpd
""",
    )

    parser.add_argument(
        "--mode",
        choices=["core3x3", "lambda-sweep", "full", "custom"],
        default="core3x3",
        help="Grid expansion mode (default: core3x3)",
    )
    parser.add_argument(
        "--gold-dataset",
        required=True,
        help="Path to gold dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for output files (default: data/output)",
    )
    parser.add_argument(
        "--workload",
        default="medium_qpd",
        choices=["low_qpd", "medium_qpd", "high_qpd", "production"],
        help="Workload profile (default: medium_qpd)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--evaluation-horizon",
        type=int,
        default=None,
        help="Override evaluation horizon (number of dataset days to process)",
    )

    # Core3x3 / custom options
    parser.add_argument("--memory-types", nargs="+", default=None,
                        help="Memory types to include (custom/core mode)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Retrieval strategies to include")
    parser.add_argument("--decay-policies", nargs="+", default=None,
                        help="Decay policies to include")
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help="Lambda values to sweep")

    # Lambda sweep options
    parser.add_argument("--memory-type", default="episodic",
                        help="Memory type for lambda-sweep mode")
    parser.add_argument("--strategy", default="bm25",
                        help="Retrieval strategy for lambda-sweep mode")
    parser.add_argument("--decay", default="exponential",
                        help="Decay policy for lambda-sweep mode")

    args = parser.parse_args()

    # Ensure project root in path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from benchmark.recommendation.ranker import MatrixRanker, QualityThresholds
    from benchmark.workload.aggregator import MatrixAggregator, MatrixReporter
    from benchmark.workload.matrix import MatrixExpander
    from benchmark.workload.profile import get_profile
    from benchmark.workload.scheduler import MatrixScheduler

    # Validate dataset
    gold_path = Path(args.gold_dataset)
    if not gold_path.exists():
        print(f"ERROR: Gold dataset not found: {gold_path}", file=sys.stderr)
        print("Generate one with: python generate_production_dataset.py --lite", file=sys.stderr)
        sys.exit(1)

    # Resolve profile
    profile = get_profile(args.workload)
    evaluation_horizon = args.evaluation_horizon or profile.evaluation_horizon

    run_id = uuid.uuid4().hex[:12]
    output_dir = Path(args.output_dir) / f"matrix_{run_id}"

    print("\nMemTuner — Matrix Runner")
    print(f"{'='*56}")
    print(f"  Run ID:          {run_id}")
    print(f"  Mode:            {args.mode}")
    print(f"  Gold dataset:    {gold_path} ({gold_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Workload:        {profile.label}")
    print(f"  Evaluation horizon: {evaluation_horizon}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Platform:        {sys.platform}")

    # Expand matrix
    expander = MatrixExpander()

    if args.mode == "core3x3":
        cells = expander.expand_core_3x3(
            workload_profile=args.workload,
            seed=args.seed,
        )
    elif args.mode == "lambda-sweep":
        cells = expander.expand_lambda_sweep(
            memory_type=args.memory_type,
            strategy=args.strategy,
            decay_policy=args.decay,
            workload_profile=args.workload,
            seed=args.seed,
        )
    elif args.mode == "full":
        cells = expander.expand_full(
            workload_profile=args.workload,
            seed=args.seed,
        )
    else:  # custom
        cells = expander.expand_full(
            workload_profile=args.workload,
            seed=args.seed,
            strategies=args.strategies,
            memory_types=args.memory_types,
            decay_policies=args.decay_policies,
            lambda_steps=args.lambdas,
        )

    desc = expander.describe(cells)
    print("\n  Matrix dimensions:")
    print(f"    Memory types:        {desc['memory_types']}")
    print(f"    Retrieval strategies:{desc['retrieval_strategies']}")
    print(f"    Decay policies:      {desc['decay_policies']}")
    print(f"    Lambda values:       {desc['lambda_values']}")
    print(f"    Total cells:         {desc['total_cells']}")
    print()

    # Confirm if large run
    if len(cells) > 50:
        print(f"  WARNING: {len(cells)} cells — this will take significant time.")
        print("  Use --mode core3x3 for a quick 27-cell comparison first.")
        print()

    # Run
    scheduler = MatrixScheduler(max_workers=args.workers, output_dir=str(output_dir))

    t0 = time.monotonic()
    results = scheduler.run(
        cells=cells,
        gold_dataset_path=str(gold_path),
        evaluation_horizon=evaluation_horizon,
    )
    elapsed = time.monotonic() - t0

    print(f"\n  Completed {len(results)} runs in {elapsed:.1f}s")
    print(f"  Successful: {sum(1 for r in results if r.success)}")
    print(f"  Failed:     {sum(1 for r in results if not r.success)}")

    # Aggregate and report
    aggregator = MatrixAggregator(results)
    reporter = MatrixReporter(output_dir)
    paths = reporter.write_all(aggregator, run_id)

    # Recommendation
    ranker = MatrixRanker(QualityThresholds())
    best = ranker.best_production_config(results)

    print(f"\n{'='*56}")
    print("BEST PRODUCTION CONFIG:")
    if best:
        print(f"  Memory type:        {best.memory_type}")
        print(f"  Retrieval strategy: {best.retrieval_strategy}")
        print(f"  Decay policy:       {best.decay_policy} (λ={best.lambda_value:.2f})")
        print(f"  Recall@K:           {best.recall_at_k:.4f}")
        print(f"  Noise Ratio:        {best.contamination_rate:.4f}")
        print(f"  Composite Score:    {best.composite_score:.4f}")
        print(f"  Peak RAM:           {best.peak_ram_mb:.1f} MB")
        print(f"  Explanation:        {best.explanation}")
    else:
        print("  No configuration met all quality thresholds.")
        print("  Review noise ratio — try higher lambda values or lower pruning thresholds.")

    print(f"\n  Reports written to: {output_dir}")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    print()


if __name__ == "__main__":
    # Required for Windows multiprocessing 'spawn' context
    # Must be inside if __name__ == '__main__' guard
    main()
