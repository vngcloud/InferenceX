#!/usr/bin/env bash
#
# Agentic-replay Mode 1 launcher: MiniMaxAI/MiniMax-M2.5 on SGLang,
# driven by a recorded mooncake_trace JSONL through official AIPerf.
# Uses the official MiniMax-M2.5 SGLang launch args, plus harness host/port
# and context length for long-context replay.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMaxAI/MiniMax-M2.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-147456}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# An optional "#N" suffix on the input-file path replays only the first N
# records (a low-resource subset of a large committed trace).
trace_limit=""
if [[ "$INPUT_FILE" == *"#"* ]]; then
    trace_limit="${INPUT_FILE##*#}"
    INPUT_FILE="${INPUT_FILE%#*}"
fi

# The trace JSONL path is repo-relative; the container runs with cwd=/workspace.
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: trace input file not found: $INPUT_FILE (cwd=$(pwd))" >&2
    exit 1
fi

if [[ -n "$trace_limit" ]]; then
    head -n "$trace_limit" "$INPUT_FILE" > /workspace/_trace_subset.jsonl
    INPUT_FILE=/workspace/_trace_subset.jsonl
    echo "Subset trace to first $trace_limit records -> $INPUT_FILE"
fi

# Mode 1 (capacity sweep): strip recorded per-turn delays so the run is driven
# purely by --concurrency back-pressure with zero think-time.
if [[ "${STRIP_TRACE_DELAYS:-}" == "true" || "${STRIP_TRACE_DELAYS:-}" == "1" ]]; then
    python3 -c '
import json, sys
with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("delay", None)
        fout.write(json.dumps(rec) + "\n")
' "$INPUT_FILE" /workspace/_trace_nodelay.jsonl
    INPUT_FILE=/workspace/_trace_nodelay.jsonl
    echo "Stripped per-turn delays for capacity sweep -> $INPUT_FILE"
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

set -x
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tp "$TP" \
  --ep "$EP_SIZE" \
  --context-length "$MAX_MODEL_LEN" \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --chunked-prefill-size 16384 \
  --hicache-size 1200 \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Request count: explicit env override takes precedence; otherwise replay the
# trace exactly once, one request per dataset record.
if [[ -z "${REQUEST_COUNT:-}" ]]; then
    REQUEST_COUNT=$(grep -c . "$INPUT_FILE")
fi
echo "Replaying trace $INPUT_FILE: request-count=$REQUEST_COUNT at concurrency $CONC"

MODE1_ARGS=()
if [[ "${NO_FIXED_SCHEDULE:-}" == "true" || "${NO_FIXED_SCHEDULE:-}" == "1" ]]; then
    MODE1_ARGS+=(--no-fixed-schedule)
fi
if [[ -n "${NUM_WARMUP_SESSIONS:-}" ]]; then
    MODE1_ARGS+=(--num-warmup-sessions "$NUM_WARMUP_SESSIONS")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    --request-count "$REQUEST_COUNT" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    --extra-inputs ignore_eos:true \
    "${MODE1_ARGS[@]}"

stop_gpu_monitor
set +x
