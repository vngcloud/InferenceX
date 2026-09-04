#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for LiveCodeBench, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, SCENARIO, RELEASE_VERSION, N, TEMPERATURE,
#               MAX_TOKENS, MULTIPROCESS, TIMEOUT, NUM_PROCESS_EVALUATE, OPENAI_TIMEOUT

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"
export OPENAI_KEY="$QUALITY_API_KEY"
export OPENAI_BASE_URL="$QUALITY_ENDPOINT"

LCB_DIR="${QUALITY_LCB_DIR:-$WORKSPACE_DIR/LiveCodeBench}"
PYTHON="${QUALITY_LCB_VENV:-$LCB_DIR/.venv-lcb}/bin/python"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

SCENARIO="${SCENARIO:-codegeneration}"
RELEASE_VERSION="${RELEASE_VERSION:-release_latest}"
N="${N:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MULTIPROCESS="${MULTIPROCESS:-4}"
TIMEOUT="${TIMEOUT:-6}"
NUM_PROCESS_EVALUATE="${NUM_PROCESS_EVALUATE:-12}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-90}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/livecodebench"

mkdir -p "$OUT_DIR"

export LCB_OUTPUT_DIR="$OUT_DIR/"

# LCB CLI has no --limit flag; pass LIMIT via LCB_LIMIT env var (patched main.py reads it)
export LCB_LIMIT="${LIMIT:-0}"

echo "=== LiveCodeBench run ==="
echo "  RUN_ID          : $RUN_ID"
echo "  Model           : $RAW_MODEL"
echo "  Scenario        : $SCENARIO"
echo "  Release version : $RELEASE_VERSION"
echo "  N (samples)     : $N"
echo "  Temperature     : $TEMPERATURE"
echo "  Max tokens      : $MAX_TOKENS"
echo "  Multiprocess    : $MULTIPROCESS"
echo "  Eval timeout    : ${TIMEOUT}s"
echo "  Output dir      : $OUT_DIR"
echo

cd "$LCB_DIR"

exec "$PYTHON" -m lcb_runner.runner.main \
  --model "$RAW_MODEL" \
  --scenario "$SCENARIO" \
  --release_version "$RELEASE_VERSION" \
  --n "$N" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --multiprocess "$MULTIPROCESS" \
  --timeout "$TIMEOUT" \
  --num_process_evaluate "$NUM_PROCESS_EVALUATE" \
  --openai_timeout "$OPENAI_TIMEOUT" \
  --evaluate \
  --use_cache \
  --continue_existing
