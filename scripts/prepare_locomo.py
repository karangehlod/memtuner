#!/usr/bin/env python3
"""Download and prepare LoCoMo dataset for benchmarking.

Usage:
    python scripts/prepare_locomo.py [--subset-only] [--max-conversations 3]

This script:
1. Downloads locomo10.json from the GitHub repository
2. Converts to our gold format for full benchmark
3. Creates a small test subset for fast validation
4. Shows dataset statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def download_locomo(data_dir: Path) -> Path:
    """Download the LoCoMo dataset from GitHub."""
    output_path = data_dir / "locomo10.json"

    if output_path.exists():
        print(f"  Found existing: {output_path}")
        return output_path

    print(f"  Downloading from: {LOCOMO_URL}")
    print(f"  Saving to: {output_path}")

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(LOCOMO_URL, str(output_path))
    except Exception as e:
        print(f"\n  ERROR: Failed to download: {e}")
        print("  Please download manually:")
        print(f"    curl -Lo {output_path} {LOCOMO_URL}")
        sys.exit(1)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LoCoMo data for benchmarking")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory to store LoCoMo data",
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only create the test subset (faster)",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=3,
        help="Number of conversations in test subset",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=10,
        help="Max sessions per conversation in test subset",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    print("=" * 60)
    print("LoCoMo Dataset Preparation")
    print("=" * 60)

    # 1. Download
    print("\n[1/4] Downloading LoCoMo dataset...")
    locomo_path = download_locomo(data_dir)

    # 2. Show stats
    print("\n[2/4] Analyzing dataset...")
    with locomo_path.open() as fh:
        raw_data = json.load(fh)

    if isinstance(raw_data, dict):
        if "data" in raw_data:
            raw_data = raw_data["data"]
        else:
            raw_data = [raw_data]

    print(f"  Conversations: {len(raw_data)}")

    total_sessions = 0
    total_turns = 0
    total_qa = 0
    for sample in raw_data:
        conv = sample.get("conversation", {})
        sessions = [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")]
        total_sessions += len(sessions)
        for sk in sessions:
            turns = conv.get(sk, [])
            if isinstance(turns, list):
                total_turns += len(turns)
        total_qa += len(sample.get("qa", []))

    print(f"  Sessions: {total_sessions}")
    print(f"  Turns: {total_turns}")
    print(f"  QA pairs: {total_qa}")

    from benchmark.gold.locomo_loader import LoCoMoLoader

    adapter = LoCoMoLoader()
    categories = adapter.get_category_distribution(raw_data)
    print("\n  QA categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:25s}: {count:4d}")

    difficulty = adapter.get_difficulty_distribution(raw_data)
    print("\n  Difficulty distribution:")
    for level, count in sorted(difficulty.items()):
        print(f"    {level:10s}: {count:4d}")

    # 3. Create test subset
    from benchmark.gold.locomo_loader import convert_locomo_to_gold, create_test_subset

    print(f"\n[3/4] Creating test subset ({args.max_conversations} conversations, "
          f"{args.max_sessions} sessions each)...")
    subset_path = data_dir / "locomo_test_subset.json"
    subset = create_test_subset(
        locomo_path, subset_path,
        max_conversations=args.max_conversations,
        max_sessions=args.max_sessions,
    )
    print(f"  Queries:  {len(subset.queries)}")
    print(f"  Days:     {len(subset.events)}")
    print(f"  Memories: {subset.total_conversation_turns}")
    print(f"  Output:   {subset_path}")

    # 4. Convert full dataset
    if not args.subset_only:
        print("\n[4/4] Converting FULL dataset to gold format...")
        full_path = data_dir / "locomo_gold.json"
        full = convert_locomo_to_gold(locomo_path, full_path, "locomo_full")
        print(f"  Queries:  {len(full.queries)}")
        print(f"  Days:     {len(full.events)}")
        print(f"  Memories: {full.total_conversation_turns}")
        print(f"  Users:    {len(full.user_ids)}")
        print(f"  Output:   {full_path}")
    else:
        print("\n[4/4] Skipped full conversion (--subset-only)")

    print(f"\n{'=' * 60}")
    print("DONE!")
    print("\nNext steps:")
    print("  1. Quick validation:  python -m pytest tests/unit/test_locomo_adapter.py -v")
    print("  2. Benchmark run:     benchmark run --config configs/locomo.yaml")
    print("  3. Grid search:       python grid_search.py --gold data/locomo_gold.json")


if __name__ == "__main__":
    main()
