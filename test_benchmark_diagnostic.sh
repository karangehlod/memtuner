#!/bin/bash
# Comprehensive benchmark diagnostic script
# Run from venv: bash test_benchmark_diagnostic.sh

set -e

echo "=================================================="
echo "BENCHMARK COMPONENT TEST"
echo "=================================================="
echo ""

# Test 1: Component tests
echo "[1/2] Testing individual components..."
python test_components.py
if [ $? -ne 0 ]; then
    echo "Component tests failed. Fix issues above."
    exit 1
fi

echo ""
echo "[2/2] Running full benchmark..."
echo ""

# Run with detailed output
python -m benchmark.cli.main analyze \
    --dataset data/locomo10.json \
    --output data/diagnostic_benchmark \
    --max-queries 5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ BENCHMARK COMPLETED SUCCESSFULLY"
    echo "=================================================="
    echo ""
    echo "Results saved to: data/diagnostic_benchmark/"
    ls -lh data/diagnostic_benchmark/
else
    echo ""
    echo "=================================================="
    echo "✗ BENCHMARK FAILED"
    echo "=================================================="
    exit 1
fi
