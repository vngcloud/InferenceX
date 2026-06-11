#!/usr/bin/env bash
# Fixed-seq-len launcher: Qwen/Qwen3.5-27B (bf16, TP=1) on vLLM,
# benchmarked through AIPerf smoke settings on the GreenNode H200 runner.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-5-27b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-$CONC}"
if (( SERVER_MAX_NUM_SEQS > 256 )); then
    SERVER_MAX_NUM_SEQS=256
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$SERVER_MAX_NUM_SEQS" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

SEARCH_ARGS=()
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    SEARCH_ARGS+=(--search-recipe "$SEARCH_RECIPE" --concurrency-min "${CONCURRENCY_MIN}" --concurrency-max "${CONCURRENCY_MAX}")
    if [[ -n "${SLA_MS:-}" ]]; then SEARCH_ARGS+=(--sla-ms "$SLA_MS"); fi
    if [[ -n "${SEARCH_MAX_ITERATIONS:-}" ]]; then SEARCH_ARGS+=(--search-max-iterations "$SEARCH_MAX_ITERATIONS"); fi
fi
if [[ -n "${BENCHMARK_DURATION:-}" ]]; then
    SEARCH_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
    if [[ -n "${BENCHMARK_GRACE_PERIOD:-}" ]]; then
        SEARCH_ARGS+=(--benchmark-grace-period "$BENCHMARK_GRACE_PERIOD")
    fi
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --isl "$ISL" \
    --osl "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    --extra-inputs ignore_eos:true \
    "${SEARCH_ARGS[@]}"

stop_gpu_monitor
set +x
