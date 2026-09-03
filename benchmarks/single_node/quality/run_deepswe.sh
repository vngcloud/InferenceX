#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for DeepSWE, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
# Requires Docker on the runner.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, N_TASKS, CCU, JOBS_DIR, JOB_NAME

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"
export PATH="$HOME/.local/bin:$PATH"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"

N_TASKS="${N_TASKS:-${LIMIT:-6}}"
CCU="${CCU:-$N_TASKS}"
RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"
JOBS_DIR="${JOBS_DIR:-$WORKSPACE_DIR/jobs/$RUN_ID/deepswe}"
JOB_NAME="${JOB_NAME:-${N_TASKS}tasks-ccu${CCU}}"

DEEPSWE_DIR="${QUALITY_DEEPSWE_DIR:-$WORKSPACE_DIR/deep-swe}"

echo "=== DeepSWE run ==="
echo "  RUN_ID     : $RUN_ID"
echo "  Model      : $RAW_MODEL"
echo "  N tasks    : $N_TASKS"
echo "  Concurrent : $CCU"
echo "  Jobs dir   : $JOBS_DIR"
echo

cd "$DEEPSWE_DIR"

uv tool run --from datacurve-pier pier run \
  -p tasks \
  --agent mini-swe-agent \
  --model "openai/${RAW_MODEL}" \
  --agent-kwarg model_class=litellm \
  --agent-env "MSWEA_API_KEY=$QUALITY_API_KEY" \
  --agent-env "OPENAI_API_KEY=$QUALITY_API_KEY" \
  --agent-env "OPENAI_BASE_URL=$QUALITY_ENDPOINT" \
  --agent-env "OPENAI_API_BASE=$QUALITY_ENDPOINT" \
  --n-tasks "$N_TASKS" \
  --sample-seed 0 \
  --n-concurrent "$CCU" \
  --agent-setup-timeout-multiplier 3 \
  --jobs-dir "$JOBS_DIR" \
  --job-name "$JOB_NAME" \
  --yes
