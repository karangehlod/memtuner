#!/usr/bin/env bash
# ==============================================================================
# Paper Experiment Runner
# ==============================================================================
# Runs all benchmark configurations needed for the research paper.
# Produces results in analysis_output/paper/
#
# Requirements:
#   pip install -e ".[embeddings]"
#   ollama pull nemotron-3-nano:4b   (for LLM judge)
#
# Usage:
#   ./scripts/run_paper_experiments.sh
#
# Estimated time: ~30-60 minutes depending on hardware
# ==============================================================================

set -euo pipefail

OUTPUT_BASE="analysis_output/paper"
LOCOMO_DATA="data/locomo10.json"
LONGMEMEVAL_DIR="data/longmemeval"

mkdir -p "$OUTPUT_BASE"

echo "════════════════════════════════════════════════════════════════"
echo "  PAPER EXPERIMENT SUITE"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ==============================================================================
# Experiment 1: Multi-model embedding comparison on LoCoMo (3 seeds)
# ==============================================================================
echo "━━━ Experiment 1: Embedding Model Comparison (LoCoMo) ━━━"

for SEED in 42 123 456; do
  for MODEL in "all-MiniLM-L6-v2" "BAAI/bge-base-en-v1.5"; do
    LABEL=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]')
    OUT="$OUTPUT_BASE/locomo_${LABEL}_seed${SEED}"
    echo "  → $LABEL seed=$SEED → $OUT"
    BENCHMARK_EMBEDDING_MODEL="$MODEL" \
    .venv/bin/python -m benchmark.cli.main analyze \
      --dataset "$LOCOMO_DATA" \
      --output "$OUT" 2>/dev/null
  done
done

echo ""
echo "━━━ Experiment 2: Embedding Model Comparison (LongMemEval) ━━━"

for SEED in 42 123 456; do
  for MODEL in "all-MiniLM-L6-v2" "BAAI/bge-base-en-v1.5"; do
    LABEL=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]')
    OUT="$OUTPUT_BASE/longmemeval_${LABEL}_seed${SEED}"
    echo "  → $LABEL seed=$SEED → $OUT"
    BENCHMARK_EMBEDDING_MODEL="$MODEL" \
    .venv/bin/python -m benchmark.cli.main analyze \
      --pack longmemeval \
      --data-dir "$LONGMEMEVAL_DIR" \
      --output "$OUT" 2>/dev/null
  done
done

# ==============================================================================
# Experiment 3: Hybrid RRF with BGE-Base (strongest retrieval combination)
# ==============================================================================
echo ""
echo "━━━ Experiment 3: Hybrid RRF + BGE-Base (LoCoMo) ━━━"

BENCHMARK_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" \
.venv/bin/python -m benchmark.cli.main analyze \
  --dataset "$LOCOMO_DATA" \
  --output "$OUTPUT_BASE/locomo_hybrid_bge" 2>/dev/null

echo "  → Done: $OUTPUT_BASE/locomo_hybrid_bge"

echo ""
echo "━━━ Experiment 4: Hybrid RRF + BGE-Base (LongMemEval) ━━━"

BENCHMARK_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" \
.venv/bin/python -m benchmark.cli.main analyze \
  --pack longmemeval \
  --data-dir "$LONGMEMEVAL_DIR" \
  --output "$OUTPUT_BASE/longmemeval_hybrid_bge" 2>/dev/null

echo "  → Done: $OUTPUT_BASE/longmemeval_hybrid_bge"

# ==============================================================================
# Experiment 4: LLM-as-Judge with nemotron-3-nano (50 queries)
# ==============================================================================
echo ""
echo "━━━ Experiment 5: LLM Judge (nemotron-3-nano, 50 queries) ━━━"

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  BENCHMARK_LLM_BASE_URL=http://localhost:11434/v1 \
  BENCHMARK_LLM_MODEL=nemotron-3-nano:4b \
  BENCHMARK_JUDGE_MODEL=nemotron-3-nano:4b \
  .venv/bin/python -m benchmark.cli.main analyze \
    --pack longmemeval \
    --data-dir "$LONGMEMEVAL_DIR" \
    --max-queries 50 \
    --with-llm-judge \
    --output "$OUTPUT_BASE/longmemeval_judge_50q" 2>/dev/null
  echo "  → Done: $OUTPUT_BASE/longmemeval_judge_50q"
else
  echo "  ⚠ Ollama not running — skipping LLM judge experiment"
  echo "  Start with: ollama serve"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  EXPERIMENTS COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results in: $OUTPUT_BASE/"
echo ""
echo "Key files:"
find "$OUTPUT_BASE" -name "benchmark_report.json" | sort
echo ""
echo "To aggregate results, compare JSON reports across seeds."
