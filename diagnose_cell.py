#!/usr/bin/env python3
"""Run a single benchmark cell and print the full error if it fails.

Usage:
    python diagnose_cell.py --gold-dataset data/locomo10.json
"""
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.workload.study_scheduler import _run_study_cell_worker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dataset", required=True)
    parser.add_argument("--memory-type", default="episodic")
    parser.add_argument("--strategy", default="bm25")
    args = parser.parse_args()

    cell_dict = {
        "memory_type": args.memory_type,
        "retrieval_strategy": args.strategy,
        "embedding_model": "all-MiniLM-L6-v2",
        "bm25_weight": 1.0,
        "reranker_model": "none",
        "decay_policy": "none",
        "decay_lambda": 0.0,
        "decay_pruning_threshold": 0.15,
        "workload_profile": "medium_qpd",
        "seed": 42,
        "study_phase": "diagnostic",
        "ollama_base_url": "",
    }

    print(f"Running single cell: {args.memory_type} × {args.strategy}")
    print(f"Gold dataset: {args.gold_dataset}\n")

    try:
        result = _run_study_cell_worker(
            cell_dict,
            gold_dataset_path=args.gold_dataset,
            output_dir="data/diagnostic_output",
            evaluation_horizon=50,
        )
        if result.get("success"):
            print(f"✓ SUCCESS")
            print(f"  recall={result['recall_at_k']:.3f}  mrr={result['mrr']:.3f}  p50={result['latency_p50_ms']:.1f}ms")
        else:
            print(f"✗ FAILED")
            print(f"\nFull error:\n{result.get('error_message', '(no error message)')}")
    except Exception:
        print("✗ EXCEPTION during cell execution:\n")
        traceback.print_exc()

if __name__ == "__main__":
    main()
