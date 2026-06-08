#!/usr/bin/env bash
# Fixed-seq-len launcher: Qwen/Qwen3.5-27B (bf16, TP=1) on SGLang,
# benchmarked through AIPerf on the GreenNode H200 runner.

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

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-$CONC}"
if (( SERVER_MAX_RUNNING_REQUESTS > 256 )); then
    SERVER_MAX_RUNNING_REQUESTS=256
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp "$TP" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" \
    --chunked-prefill-size 8192 \
    --decode-log-interval 1 \
    --mem-fraction-static 0.90 \
    --cuda-graph-max-bs "$SERVER_MAX_RUNNING_REQUESTS" \
    --context-length "$MAX_MODEL_LEN" \
    --attention-backend flashinfer \
    --stream-interval 50 \
    --tokenizer-worker-num 6 \
    --disable-radix-cache \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

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
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --port "$PORT" \
    --backend sglang-oai \
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
    "${SEARCH_ARGS[@]}"

stop_gpu_monitor
set +x
