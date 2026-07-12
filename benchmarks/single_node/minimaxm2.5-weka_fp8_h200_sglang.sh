#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"
export AIPERF_VENV_DIR="${AIPERF_VENV_DIR:-/tmp/aiperf-mooncake-agentx-weka-venv}"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    MAX_MODEL_LEN \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PUBLIC_DATASET="${PUBLIC_DATASET:-}"
if [[ "$CUSTOM_DATASET_TYPE" == "weka_trace" && -z "${INPUT_FILE:-}" && -z "$PUBLIC_DATASET" ]]; then
    PUBLIC_DATASET="semianalysis_cc_traces_weka_with_subagents_060826"
fi

SOURCE_ARGS=()
if [[ -n "${INPUT_FILE:-}" ]]; then
    if [[ ! -e "$INPUT_FILE" ]]; then
        echo "Error: trace input path not found: $INPUT_FILE (cwd=$(pwd))" >&2
        exit 1
    fi
    SOURCE_ARGS+=(--input-file "$INPUT_FILE")
elif [[ -n "$PUBLIC_DATASET" ]]; then
    SOURCE_ARGS+=(--public-dataset "$PUBLIC_DATASET")
else
    echo "Error: one of INPUT_FILE or PUBLIC_DATASET is required" >&2
    exit 1
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --ep "$EP_SIZE" \
  --context-length "$MAX_MODEL_LEN" \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --chunked-prefill-size 16384 \
  --enable-metrics \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

STOP_ARGS=(--benchmark-duration "${BENCHMARK_DURATION:-${DURATION:-300}}")

REPLAY_ARGS=()
if [[ -n "${TOKENIZER:-}" ]]; then
    REPLAY_ARGS+=(--tokenizer "$TOKENIZER")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    "${SOURCE_ARGS[@]}" \
    "${STOP_ARGS[@]}" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-42}" \
    "${REPLAY_ARGS[@]}"

stop_gpu_monitor
set +x
