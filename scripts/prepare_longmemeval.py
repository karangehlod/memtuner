#!/usr/bin/env python3
"""Convert LongMemEval dataset and prepare for benchmarking.

Usage:
    python scripts/prepare_longmemeval.py [--subset-only] [--questions-per-type 5]

This script:
1. Converts longmemeval_oracle.json → gold format for full benchmark
2. Creates a small test subset for fast validation
3. Shows dataset statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.gold.longmemeval_adapter import (
    LongMemEvalAdapter,
    convert_longmemeval_to_gold,
    create_test_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongMemEval data for benchmarking")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory containing LongMemEval JSON files",
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only create the test subset (faster)",
    )
    parser.add_argument(
        "--questions-per-type",
        type=int,
        default=5,
        help="Number of questions per type in test subset",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    oracle_path = data_dir / "longmemeval_oracle.json"
    nested_oracle_path = data_dir / "longmemeval" / "longmemeval_oracle.json"

    if not oracle_path.exists() and nested_oracle_path.exists():
        oracle_path = nested_oracle_path

    if not oracle_path.exists():
        print(f"ERROR: {oracle_path} not found.")
        print("Please download the LongMemEval dataset first:")
        print(f"  cd {data_dir / 'longmemeval'}")
        print("  curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json")
        print("  curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")
        print("  curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json")
        sys.exit(1)

    # Show stats
    print("=" * 60)
    print("LongMemEval Dataset Preparation")
    print("=" * 60)

    import json
    with oracle_path.open() as fh:
        raw_data = json.load(fh)

    print(f"\nSource: {oracle_path.name}")
    print(f"Total instances: {len(raw_data)}")

    adapter = LongMemEvalAdapter()
    difficulty = adapter.get_difficulty_distribution(raw_data)
    print("\nDifficulty distribution:")
    for level, count in sorted(difficulty.items()):
        print(f"  {level:10s}: {count:4d}")

    # Question type breakdown
    by_type: dict[str, int] = {}
    for instance in raw_data:
        qtype = instance.get("question_type", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1

    print("\nQuestion types:")
    for qtype, count in sorted(by_type.items()):
        print(f"  {qtype:35s}: {count:4d}")

    # 1. Create test subset (always)
    print(f"\n{'=' * 60}")
    print(f"Creating test subset ({args.questions_per_type} per type)...")
    subset_path = data_dir / "longmemeval_test_subset.json"
    subset = create_test_subset(oracle_path, subset_path, args.questions_per_type)
    print(f"  Queries:  {len(subset.queries)}")
    print(f"  Days:     {len(subset.events)}")
    print(f"  Memories: {subset.total_conversation_turns}")
    print(f"  Output:   {subset_path}")

    # 2. Convert full dataset (unless --subset-only)
    if not args.subset_only:
        print(f"\n{'=' * 60}")
        print("Converting FULL dataset to gold format...")
        full_path = data_dir / "longmemeval_oracle_gold.json"
        full = convert_longmemeval_to_gold(oracle_path, full_path, "longmemeval_full")
        print(f"  Queries:  {len(full.queries)}")
        print(f"  Days:     {len(full.events)}")
        print(f"  Memories: {full.total_conversation_turns}")
        print(f"  Users:    {len(full.user_ids)}")
        print(f"  Output:   {full_path}")

        # Also convert longmemeval_s if available
        s_path = data_dir / "longmemeval_s_cleaned.json"
        if s_path.exists():
            print(f"\n{'=' * 60}")
            print("Converting LongMemEval-S (115k tokens) to gold format...")
            s_gold_path = data_dir / "longmemeval_s_gold.json"
            s_gold = convert_longmemeval_to_gold(s_path, s_gold_path, "longmemeval_s")
            print(f"  Queries:  {len(s_gold.queries)}")
            print(f"  Days:     {len(s_gold.events)}")
            print(f"  Memories: {s_gold.total_conversation_turns}")
            print(f"  Output:   {s_gold_path}")

        # Also convert longmemeval_m if available
        m_path = data_dir / "longmemeval_m_cleaned.json"
        if m_path.exists():
            print(f"\n{'=' * 60}")
            print("Converting LongMemEval-M (500 sessions) to gold format...")
            m_gold_path = data_dir / "longmemeval_m_gold.json"
            m_gold = convert_longmemeval_to_gold(m_path, m_gold_path, "longmemeval_m")
            print(f"  Queries:  {len(m_gold.queries)}")
            print(f"  Days:     {len(m_gold.events)}")
            print(f"  Memories: {m_gold.total_conversation_turns}")
            print(f"  Output:   {m_gold_path}")

    print(f"\n{'=' * 60}")
    print("DONE!")
    print("\nNext steps:")
    print("  1. Quick validation:  python -m pytest tests/ -x -q")
    print("  2. Benchmark run:     benchmark run --config configs/longmemeval.yaml")
    print("  3. Grid search:       python grid_search.py --gold data/longmemeval_oracle_gold.json")


if __name__ == "__main__":
    main()
