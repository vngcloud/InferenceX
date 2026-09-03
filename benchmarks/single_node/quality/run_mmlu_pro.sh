#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for MMLU-Pro, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, LIMIT, NUM_CONCURRENT, MAX_LENGTH, MAX_GEN_TOKS,
#               TASK, NUM_FEWSHOT, BATCH_SIZE

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"
export PATH="$HOME/.local/bin:$PATH"

LM_EVAL="${QUALITY_VENV:-$WORKSPACE_DIR/.venv-lmeval}/bin/lm-eval"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"
ENDPOINT="${QUALITY_ENDPOINT%/}/chat/completions"
export OPENAI_API_KEY="$QUALITY_API_KEY"
export HF_TOKEN="${HF_TOKEN:-}"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
TASK="${TASK:-mmlu_pro}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"

LIMIT="${LIMIT:-}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/mmlu_pro"
CACHE_DB="$OUT_DIR/cache.db"

mkdir -p "$OUT_DIR"

SUBSET_ARGS=()
if [[ -n "$LIMIT" ]]; then
  SUBSET_ARGS=(--limit "$LIMIT")
fi

echo "=== MMLU-Pro run ==="
echo "  RUN_ID        : $RUN_ID"
echo "  Model         : $RAW_MODEL"
echo "  Task          : $TASK"
echo "  Output dir    : $OUT_DIR"
echo "  Cache (resume): $CACHE_DB"
if [[ -n "$LIMIT" ]]; then
  echo "  Subset        : first ${LIMIT} per subtask"
fi
echo

"$LM_EVAL" run \
  --model openai-chat-completions \
  --model_args "model=${RAW_MODEL},base_url=${ENDPOINT},tokenizer_backend=None,tokenized_requests=False,num_concurrent=${NUM_CONCURRENT},max_length=${MAX_LENGTH}" \
  --tasks "$TASK" \
  --num_fewshot "$NUM_FEWSHOT" \
  --apply_chat_template \
  --batch_size "$BATCH_SIZE" \
  --gen_kwargs "temperature=0,max_gen_toks=${MAX_GEN_TOKS}" \
  --log_samples \
  --use_cache "$CACHE_DB" \
  --output_path "$OUT_DIR" \
  "${SUBSET_ARGS[@]}"
