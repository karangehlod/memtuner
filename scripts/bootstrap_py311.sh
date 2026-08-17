#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv311"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VALIDATION_OUTPUT="$ROOT_DIR/analysis_output/environment-validation-311.json"
STAMP_FILE="$ROOT_DIR/analysis_output/bootstrap-py311.ok"

echo "[bootstrap_py311] root: $ROOT_DIR"
echo "[bootstrap_py311] python: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[bootstrap_py311] error: $PYTHON_BIN not found on PATH" >&2
  echo "[bootstrap_py311] install Python 3.11, or rerun with PYTHON_BIN=/path/to/python3.11" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/analysis_output"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -e '.[dev]'
"$VENV_DIR/bin/python" -m benchmark.cli.main validate \
  --config "$ROOT_DIR/configs/locomo.yaml" \
  --check-environment \
  --environment-output "$VALIDATION_OUTPUT"

printf 'python=%s\nvenv=%s\nvalidation=%s\n' \
  "$PYTHON_BIN" \
  "$VENV_DIR" \
  "$VALIDATION_OUTPUT" > "$STAMP_FILE"

echo "[bootstrap_py311] complete"
echo "[bootstrap_py311] venv: $VENV_DIR"
echo "[bootstrap_py311] validation: $VALIDATION_OUTPUT"
echo "[bootstrap_py311] stamp: $STAMP_FILE"