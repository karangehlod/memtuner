#!/usr/bin/env python3
"""
Compare different memory retrieval strategies on the same gold dataset.

This script:
1. Generates a gold dataset
2. Runs benchmark with different retrieval strategies
3. Compares results side-by-side

Usage:
    python3 scripts/compare_retrieval_strategies.py
    
Or with custom gold data:
    python3 scripts/compare_retrieval_strategies.py --gold-dataset my_data.json
"""

import json
import sys
from pathlib import Path
from subprocess import run, PIPE
import argparse
from tabulate import tabulate


def generate_gold_dataset(output_path: str, seed: int = 42) -> str:
    """Generate a test gold dataset."""
    print(f"📊 Generating gold dataset: {output_path}")
    
    result = run([
        "benchmark",
        "generate-gold",
        "--seed", str(seed),
        "--users", "3",
        "--days", "7",
        "--output", output_path,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to generate gold dataset:\n{result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    return output_path


def run_benchmark_strategy(
    strategy: str,
    config_path: str,
    gold_dataset: str,
    output_dir: str,
) -> dict:
    """Run benchmark with a specific retrieval strategy."""
    
    print(f"\n🚀 Running benchmark: {strategy}")
    print(f"   Config: {config_path}")
    print(f"   Gold: {gold_dataset}")
    
    result = run([
        "benchmark",
        "run",
        "--config", config_path,
        "--gold-dataset", gold_dataset,
        "--output-dir", output_dir,
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"⚠️  Benchmark failed:\n{result.stderr}")
        return None
    
    # Find and parse the results JSON
    output_path = Path(output_dir)
    json_files = list(output_path.glob("run_*.json"))
    
    if not json_files:
        print(f"❌ No results file found in {output_dir}")
        return None
    
    with open(json_files[0]) as f:
        results = json.load(f)
    
    print(f"✅ Completed: {results['run_id']}")
    return results


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from results."""
    if not results or 'scenario_results' not in results:
        return {}
    
    scenario = results['scenario_results'][0]
    cost = results.get('cost_summary', {})
    
    return {
        'Recall@K': f"{scenario.get('recall_at_k', 0):.1%}",
        'Precision@K': f"{scenario.get('precision_at_k', 0):.1%}",
        'Noise Ratio': f"{scenario.get('contamination_rate', 0):.1%}",
        'Temporal Acc': f"{scenario.get('temporal_accuracy', 0):.1%}",
        'Module Acc': f"{scenario.get('module_accuracy', 0):.1%}",
        'Total Cost': f"${cost.get('total_cost', 0):.4f}",
        'Correct Recalls': f"{scenario.get('correct_recalls', 0)}/{scenario.get('total_queries', 0)}",
    }


def create_configs(config_dir: str = "configs"):
    """Create config files for different strategies if they don't exist."""
    
    strategies = {
        'default.yaml': """memory:
  enabled:
    short_term: [episodic_buffer]
    long_term: [episodic_store, preference_store]

benchmark:
  seed: 42
  evaluation_horizon: 14
  scenarios: [delayed_recall]
  recall_k: 5
""",
        
        'semantic.yaml': """memory:
  enabled:
    short_term: [episodic_buffer]
    long_term: [semantic_store]

benchmark:
  seed: 42
  evaluation_horizon: 14
  scenarios: [delayed_recall]
  recall_k: 5
""",
    }
    
    config_path = Path(config_dir)
    config_path.mkdir(exist_ok=True)
    
    created = []
    for filename, content in strategies.items():
        filepath = config_path / filename
        if not filepath.exists():
            filepath.write_text(content)
            created.append(filename)
    
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Compare different memory retrieval strategies"
    )
    parser.add_argument(
        "--gold-dataset",
        default="comparison_gold.json",
        help="Path to gold dataset (will be generated if not found)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for gold generation"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["default.yaml", "semantic.yaml"],
        help="Config files to compare"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 70)
    print("MEMORY RETRIEVAL STRATEGY COMPARISON")
    print("=" * 70)
    
    # Generate gold dataset if needed
    gold_dataset = args.gold_dataset
    if not Path(gold_dataset).exists():
        generate_gold_dataset(gold_dataset, seed=args.seed)
    else:
        print(f"✓ Using existing gold dataset: {gold_dataset}")
    
    # Create configs
    print("\n📋 Setting up configs...")
    created = create_configs()
    if created:
        print(f"   Created: {', '.join(created)}")
    
    # Run benchmarks
    results = {}
    output_base = Path("comparison_results")
    output_base.mkdir(exist_ok=True)
    
    for strategy in args.strategies:
        config_path = f"configs/{strategy}"
        if not Path(config_path).exists():
            print(f"⚠️  Config not found: {config_path}")
            continue
        
        strategy_name = Path(strategy).stem
        output_dir = str(output_base / strategy_name)
        Path(output_dir).mkdir(exist_ok=True)
        
        benchmark_results = run_benchmark_strategy(
            strategy=strategy_name,
            config_path=config_path,
            gold_dataset=gold_dataset,
            output_dir=output_dir,
        )
        
        if benchmark_results:
            results[strategy_name] = benchmark_results
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Build comparison table
    rows = []
    for strategy, benchmark_results in results.items():
        metrics = extract_metrics(benchmark_results)
        row = [strategy] + [metrics.get(k, 'N/A') for k in [
            'Recall@K', 'False Pos Rate', 'Temporal Acc', 'Module Acc', 'Total Cost', 'Correct Recalls'
        ]]
        rows.append(row)
    
    headers = ['Strategy', 'Recall@K', 'False Pos Rate', 'Temporal Acc', 'Module Acc', 'Total Cost', 'Correct']
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    recalls = {s: float(extract_metrics(r).get('Recall@K', '0%').rstrip('%')) 
               for s, r in results.items()}
    costs = {s: float(extract_metrics(r).get('Total Cost', '$0').lstrip('$'))
             for s, r in results.items()}
    
    best_recall = max(recalls, key=recalls.get) if recalls else None
    lowest_cost = min(costs, key=costs.get) if costs else None
    
    if best_recall:
        print(f"\n✅ Best Recall: {best_recall} ({recalls[best_recall]:.1%})")
    if lowest_cost:
        print(f"💰 Lowest Cost: {lowest_cost} (${costs[lowest_cost]:.4f})")
    
    print("\nℹ️  For detailed interpretation, see MEMORY_SYSTEM_GUIDE.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
