#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-orch_v1}"
TMUX_SESSION="${TMUX_SESSION:-orch_edge}"

echo "[INFO] Waiting 120 seconds before starting orchestration..."
sleep 2

cd "$PROJECT_DIR"

echo "[INFO] Project: $PROJECT_DIR"
echo "[INFO] COMPOSE_PROJECT: $COMPOSE_PROJECT"
echo "[INFO] TMUX_SESSION: $TMUX_SESSION"

_read_env_var() {
  local env_file="$1"
  local key="$2"
  [[ -f "$env_file" ]] || return 1
  local line
  line="$(
    sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$/\\1/p" "$env_file" | tail -n 1
  )"
  [[ -n "${line:-}" ]] || return 1
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "$line"
}

echo "[INFO] Reading optional env vars from .env (if present)..."
SPEAKER_MAC_ADDRESS="${SPEAKER_MAC_ADDRESS:-$(_read_env_var "$PROJECT_DIR/.env" SPEAKER_MAC_ADDRESS || true)}"
SUDO_PWD="${SUDO_PWD:-$(_read_env_var "$PROJECT_DIR/.env" SUDO_PWD || true)}"

echo "[INFO] SPEAKER_MAC_ADDRESS: ${SPEAKER_MAC_ADDRESS:-<empty>}"
echo "[INFO] SUDO_PWD: ${SUDO_PWD:+<set>}"

echo "[INFO] Checking existing tmux session '$TMUX_SESSION'..."
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "[INFO] Existing tmux session found, killing it..."
  tmux kill-session -t "$TMUX_SESSION" || true
fi

echo "[INFO] Creating new tmux session '$TMUX_SESSION'..."

tmux new-session -d -s "$TMUX_SESSION" -n "llama"
MODEL_DIR="${MODEL_DIR:-$(_read_env_var "$PROJECT_DIR/.env" LLM_MODEL_DIR || true)}"
MODEL_FILE="${MODEL_FILE:-$(_read_env_var "$PROJECT_DIR/.env" LLM_MODEL_FILE || true)}"
LLAMA_PORT="${LLAMA_PORT:-$(_read_env_var "$PROJECT_DIR/.env" LLM_PORT || true)}"

: "${MODEL_DIR:=$PROJECT_DIR/models}"
: "${MODEL_FILE:=qwen3-1.7b.Q4_K_M_v3.gguf}"
: "${LLAMA_PORT:=8080}"

echo "[INFO] LLM MODEL_DIR: $MODEL_DIR"
echo "[INFO] LLM MODEL_FILE: $MODEL_FILE"
echo "[INFO] LLM PORT: $LLAMA_PORT"

JETSON_DIR="$PROJECT_DIR/jetson-containers"

echo "[INFO] Using JETSON_DIR: $JETSON_DIR"

LLAMA_IMAGE="$(cd "$JETSON_DIR" && ./autotag llama_cpp | tail -n 1 | tr -d '\r' | sed 's/[[:space:]]//g' | sed 's/\.$//')"
: "${LLAMA_IMAGE:?Failed to resolve llama_cpp image tag}"

echo "[INFO] Starting llama-server in tmux window 'llama'..."
tmux send-keys -t "$TMUX_SESSION:llama" "cd \"$JETSON_DIR\" && ./jetson-containers run -v \"$MODEL_DIR:/models\" \"$LLAMA_IMAGE\" llama-server -m \"/models/$MODEL_FILE\" --port \"$LLAMA_PORT\" --host 0.0.0.0 -ngl 99 -t 12 -fa on --parallel 1   --jinjava --n-predict 80 --ctx-size 512 --top-p 0.9 --batch-size 512 " C-m


echo "[INFO] Creating 'warmup' window..."
tmux new-window -t "$TMUX_SESSION" -n "warmup"
tmux send-keys -t "$TMUX_SESSION:warmup" "sleep 10; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"Say hello briefly.\"} ] }' >/dev/null || true; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"What is 2 plus 2?\"} ] }' >/dev/null || true; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"Give me a very short fun fact.\"} ] }' >/dev/null || true; echo \"[DONE] warmup\"; exec bash" C-m

echo "[INFO] Creating 'bluetooth' window (run Bluetooth setup first)..."
tmux new-window -t "$TMUX_SESSION" -n "bluetooth"
tmux send-keys -t "$TMUX_SESSION:bluetooth" "cd \"$PROJECT_DIR\" && chmod +x ./setup_blue.sh || true; ./setup_blue.sh; exec bash" C-m

echo "[INFO] Waiting 30 seconds after Bluetooth setup before starting docker compose..."
sleep 30

echo "[INFO] Creating 'stack' window and starting docker compose (after Bluetooth)..."
tmux new-window -t "$TMUX_SESSION" -n "stack" "cd \"$PROJECT_DIR\" && docker compose up"

echo "[DONE] tmux session created. Attach with:"
echo "  tmux attach -t \"$TMUX_SESSION\""

while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
  sleep 5
done
