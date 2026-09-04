#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for GPQA-Diamond, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh (from benchmark-tmpl.yml
# inputs), not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, LIMIT, HF_TOKEN, NUM_CONCURRENT, MAX_LENGTH, MAX_GEN_TOKS,
#               TASK, NUM_FEWSHOT, BATCH_SIZE, SAMPLE_N, SAMPLE_SEED

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"
export PATH="$HOME/.local/bin:$PATH"

LM_EVAL="${QUALITY_VENV:-$WORKSPACE_DIR/.venv-lmeval}/bin/lm-eval"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"
ENDPOINT="${QUALITY_ENDPOINT%/}/chat/completions"
export OPENAI_API_KEY="$QUALITY_API_KEY"
export HF_TOKEN="${HF_TOKEN:-}"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

MAX_LENGTH="${MAX_LENGTH:-10240}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-10240}"
TASK="${TASK:-gpqa_diamond_cot_n_shot}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"

LIMIT="${LIMIT:-}"
SAMPLE_N="${SAMPLE_N:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/gpqa"
CACHE_DB="$OUT_DIR/cache.db"

mkdir -p "$OUT_DIR"

SUBSET_ARGS=()
if [[ -n "$SAMPLE_N" ]]; then
  SAMPLE_JSON="$OUT_DIR/sample_indices.json"
  "${QUALITY_VENV:-$WORKSPACE_DIR/.venv-lmeval}/bin/python" -c "
import json, random
random.seed(${SAMPLE_SEED})
indices = sorted(random.sample(range(198), int('${SAMPLE_N}')))
json.dump({'${TASK}': indices}, open('${SAMPLE_JSON}', 'w'))
"
  SUBSET_ARGS=(--samples "$(cat "$SAMPLE_JSON")")
  echo "  Subset        : random ${SAMPLE_N} (seed=${SAMPLE_SEED})"
elif [[ -n "$LIMIT" ]]; then
  SUBSET_ARGS=(--limit "$LIMIT")
  echo "  Subset        : first ${LIMIT}"
fi

echo "=== GPQA run ==="
echo "  RUN_ID        : $RUN_ID"
echo "  Model         : $RAW_MODEL"
echo "  Task          : $TASK"
echo "  Output dir    : $OUT_DIR"
echo "  Cache (resume): $CACHE_DB"
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
