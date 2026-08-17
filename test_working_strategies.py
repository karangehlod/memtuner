#!/usr/bin/env python3
"""
Test only the working strategies (bm25 and llm_rerank) to verify benchmark works.
Skip HF and Ollama which have network/proxy issues.
"""

import subprocess
import json
import os

os.chdir('/Users/karangehlod/Codes/Agenticmemory_benchmark')

print("="*70)
print("Running benchmark with working strategies only")
print("="*70)

# Run benchmark with 10 queries
result = subprocess.run([
    'python', '-m', 'benchmark.cli.main', 'analyze',
    '--dataset', 'data/locomo10.json',
    '--output', 'data/test_working_only',
    '--max-queries', '10',
    '--seed', '42',
], capture_output=True, text=True)

print("\n📊 Benchmark Output:")
print(result.stdout[:2000])

if result.returncode != 0:
    print("\n❌ Benchmark stderr:")
    print(result.stderr[-1000:])
else:
    print("\n✅ Benchmark completed successfully!")

    # Try to load and display report
    report_path = 'data/test_working_only/benchmark_report.json'
    if os.path.exists(report_path):
        print(f"\n📋 Report generated at: {report_path}")
        with open(report_path) as f:
            report = json.load(f)

        print("\n📊 Strategy Comparison Results:")
        if 'strategy_comparison' in report:
            for result in report['strategy_comparison']:
                print(f"  {result['strategy']:.<30} Recall={result['recall']*100:.1f}%")

        print(f"\n📈 Report Status: {report.get('status', 'unknown')}")
        if 'runtime_error' in report and report['runtime_error']:
            print(f"⚠️  Runtime Error: {report['runtime_error']}")
    else:
        print(f"⚠️  Report not found at {report_path}")

print("\n" + "="*70)
