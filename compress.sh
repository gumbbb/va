#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUTPUT_NAME="${1:-orchestrator-on-edge}"
OUTPUT_FILE="../${OUTPUT_NAME}.tar.gz"

echo "[INFO] Including: cache (agents, orchestrator), voice_processing/cache"
echo "[INFO] Excluding: .git, jetson-containers/data/models, llama-cpp/models, __pycache__, venv, .venv, *.pyc"
echo "[INFO] Writing: $OUTPUT_FILE"

tar -czvf "$OUTPUT_FILE" \
  --exclude='.git' \
  --exclude='jetson-containers/data/models' \
  --exclude='llama-cpp/models' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='env' \
  --exclude='*.pyc' \
  .

echo "[DONE] Created: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
