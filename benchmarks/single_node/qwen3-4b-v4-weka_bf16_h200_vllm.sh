#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# weka_trace is a directory of per-conversation JSON files.
if [[ ! -e "$INPUT_FILE" ]]; then
    echo "Error: trace input path not found: $INPUT_FILE (cwd=$(pwd))" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--served-model-name "$SERVED_MODEL_NAME" \
--tensor-parallel-size "$TP" \
--dtype bfloat16 \
--gpu-memory-utilization 0.85 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

STOP_ARGS=(--benchmark-duration "${BENCHMARK_DURATION:-${DURATION:-90}}")

REPLAY_ARGS=()
if [[ "${NO_FIXED_SCHEDULE:-true}" == "true" || "${NO_FIXED_SCHEDULE:-true}" == "1" ]]; then
    REPLAY_ARGS+=(--no-fixed-schedule)
fi
REPLAY_ARGS+=(--inter-turn-delay-cap-seconds "${INTER_TURN_DELAY_CAP_SECONDS:-60}")
REPLAY_ARGS+=(--use-think-time-only)
REPLAY_ARGS+=(--warmup-request-count "${WARMUP_REQUEST_COUNT:-2}")
REPLAY_ARGS+=(--workers-max "${WORKERS_MAX:-64}")
REPLAY_ARGS+=(--benchmark-grace-period "${BENCHMARK_GRACE_PERIOD:-120}")
REPLAY_ARGS+=(--extra-inputs "ignore_eos:${IGNORE_EOS:-true}")
REPLAY_ARGS+=(--extra-inputs "temperature:${TEMPERATURE:-0}")
if [[ -n "${TOKENIZER:-}" ]]; then
    REPLAY_ARGS+=(--tokenizer "$TOKENIZER")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    "${STOP_ARGS[@]}" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    "${REPLAY_ARGS[@]}"

stop_gpu_monitor
set +x
