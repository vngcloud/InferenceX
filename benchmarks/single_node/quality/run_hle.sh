#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for HLE (Humanity's Last Exam), adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, LIMIT, NUM_CONCURRENT, MAX_LENGTH, MAX_GEN_TOKS,
#               TASK, NUM_FEWSHOT, BATCH_SIZE, REQUEST_TIMEOUT

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"
export PATH="$HOME/.local/bin:$PATH"

LM_EVAL="${QUALITY_VENV:-$WORKSPACE_DIR/.venv-lmeval}/bin/lm-eval"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"
ENDPOINT="${QUALITY_ENDPOINT%/}/chat/completions"
export OPENAI_API_KEY="$QUALITY_API_KEY"
export HF_TOKEN="${HF_TOKEN:-}"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

MAX_LENGTH="${MAX_LENGTH:-32768}"
# GLM reasoning tokens count against this budget. The 8k smoke still exhausted
# the budget on exact-match questions before the model emitted its final line.
MAX_GEN_TOKS="${MAX_GEN_TOKS:-${MAX_GEN_TOKENS:-16384}}"
TASK="${TASK:-hle}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
# lm-eval's default is a 300-second *total* aiohttp timeout. Streaming does not
# reset it when chunks arrive, and long HLE reasoning can legitimately exceed it.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"

LIMIT="${LIMIT:-}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/hle"
CACHE_DB="$OUT_DIR/cache.db"
TASKS_DIR="${QUALITY_TASKS_DIR:-$WORKSPACE_DIR/benchmarks/single_node/quality/tasks/hle}"

mkdir -p "$OUT_DIR"

SUBSET_ARGS=()
if [[ -n "$LIMIT" ]]; then
  SUBSET_ARGS=(--limit "$LIMIT")
fi

echo "=== HLE (Humanity's Last Exam) run ==="
echo "  RUN_ID        : $RUN_ID"
echo "  Model         : $RAW_MODEL"
echo "  Task          : $TASK"
echo "  Output dir    : $OUT_DIR"
echo "  Cache (resume): $CACHE_DB"
echo "  Max gen tokens: $MAX_GEN_TOKS"
echo "  HTTP timeout  : ${REQUEST_TIMEOUT}s"
if [[ -n "$LIMIT" ]]; then
  echo "  Subset        : first ${LIMIT} per subtask"
fi
echo

"$LM_EVAL" run \
  --model openai-chat-completions \
  --model_args "model=${RAW_MODEL},base_url=${ENDPOINT},tokenizer_backend=None,tokenized_requests=False,num_concurrent=${NUM_CONCURRENT},max_length=${MAX_LENGTH},timeout=${REQUEST_TIMEOUT}" \
  --tasks "$TASK" \
  --num_fewshot "$NUM_FEWSHOT" \
  --apply_chat_template \
  --batch_size "$BATCH_SIZE" \
  --gen_kwargs "temperature=0,max_gen_toks=${MAX_GEN_TOKS}" \
  --include_path "$TASKS_DIR" \
  --log_samples \
  --use_cache "$CACHE_DB" \
  --output_path "$OUT_DIR" \
  "${SUBSET_ARGS[@]}"

# lm-eval accepts an empty API completion and scores it as incorrect. That is
# not a valid smoke test: it usually means the reasoning budget was exhausted
# or the endpoint response shape was incompatible. Fail loudly and preserve
# the sample logs so the reason is visible in the artifact.
"${QUALITY_VENV:-$WORKSPACE_DIR/.venv-lmeval}/bin/python" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

sample_files = sorted(Path(sys.argv[1]).rglob("sample*.jsonl"))
if not sample_files:
    raise SystemExit("HLE validation failed: lm-eval produced no sample log")

empty = []
total = 0
for path in sample_files:
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        sample = json.loads(line)
        responses = sample.get("resps", [])
        strings = []
        stack = [responses]
        while stack:
            value = stack.pop()
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        if not any(value.strip() for value in strings):
            empty.append(f"{path.name}:{line_number}")

if total == 0:
    raise SystemExit("HLE validation failed: sample logs contain no records")
if empty:
    preview = ", ".join(empty[:10])
    raise SystemExit(
        f"HLE validation failed: {len(empty)}/{total} completions are empty "
        f"({preview}). The model may have exhausted max_gen_toks before emitting content."
    )
print(f"HLE response sanity check passed: {total}/{total} completions are non-empty")
PY
