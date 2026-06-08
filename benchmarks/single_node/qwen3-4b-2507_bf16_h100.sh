#!/usr/bin/env bash
#
# Fixed-seq-len launcher: Qwen/Qwen3-4B-Instruct-2507 (bf16, TP=1) on vLLM,
# benchmarked through official AIPerf. Small model used to validate the AIPerf
# search-recipe plumbing (e.g. --search-recipe max-throughput-itl-sla) end to
# end on the GreenNode H200 runner.
#
# In search mode CONC carries the upper search bound (CONCURRENCY_MAX) so the
# vLLM server is sized for the largest concurrency AIPerf's native BO may probe;
# the adapter forwards the [CONCURRENCY_MIN, CONCURRENCY_MAX] range and records
# the single winning point AIPerf converges on.

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

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

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

# Optional AIPerf native BO search recipe (config-driven via env). When set, the
# adapter delegates to `aiperf --search-recipe` over [CONCURRENCY_MIN,
# CONCURRENCY_MAX] and records the winning point AIPerf's BO selects.
SEARCH_ARGS=()
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    SEARCH_ARGS+=(--search-recipe "$SEARCH_RECIPE" --concurrency-min "${CONCURRENCY_MIN}" --concurrency-max "${CONCURRENCY_MAX}")
    if [[ -n "${SLA_MS:-}" ]]; then SEARCH_ARGS+=(--sla-ms "$SLA_MS"); fi
    if [[ -n "${SEARCH_MAX_ITERATIONS:-}" ]]; then SEARCH_ARGS+=(--search-max-iterations "$SEARCH_MAX_ITERATIONS"); fi
fi
# Optional duration-based measurement (config-driven). When set, each BO-probed
# concurrency is measured for BENCHMARK_DURATION seconds instead of a fixed
# request count; BENCHMARK_GRACE_PERIOD must exceed one request's decode time.
if [[ -n "${BENCHMARK_DURATION:-}" ]]; then
    SEARCH_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
    if [[ -n "${BENCHMARK_GRACE_PERIOD:-}" ]]; then
        SEARCH_ARGS+=(--benchmark-grace-period "$BENCHMARK_GRACE_PERIOD")
    fi
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
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
    "${SEARCH_ARGS[@]}"

stop_gpu_monitor
set +x
