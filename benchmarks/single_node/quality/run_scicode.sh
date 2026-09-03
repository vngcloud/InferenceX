#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for SciCode, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, LIMIT, SPLIT, WITH_BACKGROUND, MAX_CONNECTIONS, MAX_TOKENS,
#               SAMPLE_SHUFFLE, RETRY_ON_ERROR

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"

export PATH="${QUALITY_SCICODE_VENV:-$WORKSPACE_DIR/.venv-scicode}/bin:$HOME/.local/bin:$PATH"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"

export OPENAI_API_KEY="$QUALITY_API_KEY"
export OPENAI_BASE_URL="$QUALITY_ENDPOINT"

SCICODE_DIR="${QUALITY_SCICODE_DIR:-$WORKSPACE_DIR/SciCode}"
INSPECT="${QUALITY_SCICODE_VENV:-$WORKSPACE_DIR/.venv-scicode}/bin/inspect"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

SPLIT="${SPLIT:-test}"
WITH_BACKGROUND="${WITH_BACKGROUND:-False}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-4}"
MAX_TOKENS="${MAX_TOKENS:-32784}"
LIMIT="${LIMIT:-}"
SAMPLE_SHUFFLE="${SAMPLE_SHUFFLE:-}"

RETRY_ON_ERROR="${RETRY_ON_ERROR:-2}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/scicode"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

LIMIT_ARG=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

SHUFFLE_ARG=()
if [[ -n "$SAMPLE_SHUFFLE" ]]; then
  SHUFFLE_ARG=(--sample-shuffle "$SAMPLE_SHUFFLE")
fi

cd "$SCICODE_DIR/eval/inspect_ai"

echo "=== SciCode run ==="
echo "  RUN_ID        : $RUN_ID"
echo "  Model         : $RAW_MODEL"
echo "  Split         : $SPLIT"
echo "  Output dir    : $OUT_DIR"
echo "  Log dir       : $LOG_DIR"
echo "  Max tokens    : $MAX_TOKENS"
echo "  Retry on error: $RETRY_ON_ERROR"
if [[ -n "$LIMIT" ]]; then
  echo "  Limit         : $LIMIT"
fi
if [[ -n "$SAMPLE_SHUFFLE" ]]; then
  echo "  Sample shuffle: seed=$SAMPLE_SHUFFLE"
fi
echo

"$INSPECT" eval scicode.py \
  --model "openai/${RAW_MODEL}" \
  --temperature 0 \
  --max-connections "$MAX_CONNECTIONS" \
  --max-tokens "$MAX_TOKENS" \
  --log-dir "$LOG_DIR" \
  --max-retries "$RETRY_ON_ERROR" \
  --no-fail-on-error \
  --metadata "run_id=${RUN_ID}" \
  "${LIMIT_ARG[@]}" \
  "${SHUFFLE_ARG[@]}" \
  -T split="$SPLIT" \
  -T output_dir="$OUT_DIR" \
  -T with_background="$WITH_BACKGROUND" \
  -T h5py_file="../data/test_data.h5" \
  -T mode=normal
