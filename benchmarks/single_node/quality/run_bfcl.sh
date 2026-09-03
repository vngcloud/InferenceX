#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for BFCL v4, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, BFCL_MODE, TEST_CATEGORY, NUM_THREADS, TEMPERATURE,
#               OPENAI_TIMEOUT, FULL_EVAL, OVERWRITE, TEST_CASE_IDS

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"

BFCL_DIR="${QUALITY_BFCL_DIR:-$WORKSPACE_DIR/BFCL/berkeley-function-call-leaderboard}"
PYTHON="${QUALITY_BFCL_VENV:-$BFCL_DIR/.venv-bfcl}/bin/python"
BFCL="${QUALITY_BFCL_VENV:-$BFCL_DIR/.venv-bfcl}/bin/bfcl"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"
OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/bfcl"
mkdir -p "$OUT_DIR"

export BFCL_PROJECT_ROOT="$OUT_DIR"

cat > "$OUT_DIR/.env" <<EOF
OPENAI_API_KEY=${QUALITY_API_KEY}
OPENAI_BASE_URL=${QUALITY_ENDPOINT}
EOF

BFCL_MODE="${BFCL_MODE:-FC}"
case "$BFCL_MODE" in
  FC)     BFCL_MODEL_KEY="${RAW_MODEL}-FC" ;;
  PROMPT) BFCL_MODEL_KEY="${RAW_MODEL}-PROMPT" ;;
  *) echo "BFCL_MODE must be FC or PROMPT (got: $BFCL_MODE)" >&2; exit 1 ;;
esac

TEST_CATEGORY="${TEST_CATEGORY:-simple_python,multiple,parallel,parallel_multiple,irrelevance}"
NUM_THREADS="${NUM_THREADS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-90}"

PARTIAL_EVAL_FLAG="--partial-eval"
if [[ "${FULL_EVAL:-0}" == "1" ]]; then
  PARTIAL_EVAL_FLAG=""
fi

RUN_IDS_ARG=()
if [[ -n "${TEST_CASE_IDS:-}" ]]; then
  IDS_FILE="$OUT_DIR/test_case_ids_to_generate.json"
  if [[ -f "$TEST_CASE_IDS" ]]; then
    cp "$TEST_CASE_IDS" "$IDS_FILE"
  else
    printf '%s\n' "$TEST_CASE_IDS" > "$IDS_FILE"
  fi
  RUN_IDS_ARG=(--run-ids)
fi

OVERWRITE_ARG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARG=(--allow-overwrite)
fi

echo "=== BFCL v4 run ==="
echo "  RUN_ID            : $RUN_ID"
echo "  Mode              : $BFCL_MODE  (is_fc_model via OpenAICompletionsHandler)"
echo "  BFCL model key    : $BFCL_MODEL_KEY"
echo "  Wire model id     : $RAW_MODEL"
echo "  Endpoint          : $QUALITY_ENDPOINT"
echo "  Test categories   : $TEST_CATEGORY"
echo "  Threads           : $NUM_THREADS"
echo "  Temperature       : $TEMPERATURE"
echo "  BFCL_PROJECT_ROOT : $OUT_DIR"
echo "  Bin               : $BFCL"
echo

cd "$BFCL_DIR"

"$BFCL" generate \
  --model "$BFCL_MODEL_KEY" \
  --test-category "$TEST_CATEGORY" \
  --num-threads "$NUM_THREADS" \
  --temperature "$TEMPERATURE" \
  "${RUN_IDS_ARG[@]}" \
  "${OVERWRITE_ARG[@]}"

"$BFCL" evaluate \
  --model "$BFCL_MODEL_KEY" \
  --test-category "$TEST_CATEGORY" \
  $PARTIAL_EVAL_FLAG

SCORE_FILE="$OUT_DIR/score/data_overall.csv"
if [[ -f "$SCORE_FILE" ]]; then
  "$PYTHON" - "$SCORE_FILE" <<'PY'
import csv, sys, pathlib
path = pathlib.Path(sys.argv[1])
with path.open(newline="") as f:
    rows = list(csv.reader(f))
if not rows:
    print("(empty score file)")
    sys.exit(0)
header, data = rows[0], rows[1:]
wanted = ["Rank", "Model", "Overall Acc", "Non-Live AST Acc", "Live Acc",
          "Multi Turn Acc", "Relevance Detection", "Irrelevance Detection",
          "Organization", "License"]
idx = [header.index(w) for w in wanted if w in header]
print(" | ".join(f"{header[i]}: {data[0][i]}" for i in idx) if data else "(no rows)")
print(f"\nFull table: {path}")
PY
else
  echo "Score file not found: $SCORE_FILE"
fi

echo
echo "=== BFCL artifacts ==="
echo "  Results : $OUT_DIR/result/$(echo "$BFCL_MODEL_KEY" | tr '/' '_')/"
echo "  Scores  : $OUT_DIR/score/$(echo "$BFCL_MODEL_KEY" | tr '/' '_')/"
echo "  Overall : $OUT_DIR/score/data_overall.csv"
echo "  Non-live: $OUT_DIR/score/data_non_live.csv"
