#!/usr/bin/env bash
# Gemma 4 31B FP8 on 2x H100 via vLLM + LMCache (local CPU-tier, no remote).
#
# LMCache sits alongside vLLM as a KV-cache layer, pooling and reusing KV
# blocks across requests within the run. Local-only mode uses CPU RAM as
# the cache backend; no Redis or RDMA required.
#
# Requires BENCHMARK_CLIENT=aiperf — only AIPerf supports --server-metrics-url
# which is how we scrape the /metrics endpoint for cache hit counters.
#
# Three LMCache env vars control the local tier (exported below):
#   LMCACHE_LOCAL_CPU=True                 enable CPU-RAM backend
#   LMCACHE_MAX_LOCAL_CPU_SIZE=<GB>        cap cache size; default 5 GB
#   LMCACHE_CHUNK_SIZE=256                 KV block granularity (tokens)
#
# Counterpart config key: gemma4-fp8-h100-2x-vllm-lmcache in nvidia-master.yaml.
# Selected by runners/launch_h100-greennode.sh for framework=vllm-lmcache.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ "${BENCHMARK_CLIENT:-inferencex_native}" != "aiperf" ]]; then
    echo "ERROR: gemma4_fp8_h100_vllm-lmcache.sh requires BENCHMARK_CLIENT=aiperf" >&2
    echo "       Set benchmark-client: aiperf in the config entry." >&2
    exit 1
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# LMCache local-only CPU tier — no config file needed.
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"

start_gpu_monitor

set -x
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

SEARCH_ARGS=()
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    SEARCH_ARGS+=(
        --search-recipe "$SEARCH_RECIPE"
        --concurrency-min "${CONCURRENCY_MIN}"
        --concurrency-max "${CONCURRENCY_MAX}"
    )
    if [[ -n "${SLA_MS:-}" ]]; then SEARCH_ARGS+=(--sla-ms "$SLA_MS"); fi
    if [[ -n "${SEARCH_MAX_ITERATIONS:-}" ]]; then
        SEARCH_ARGS+=(--search-max-iterations "$SEARCH_MAX_ITERATIONS")
    fi
fi

DURATION_ARGS=()
if [[ -n "${BENCHMARK_DURATION:-}" ]]; then
    DURATION_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
    if [[ -n "${BENCHMARK_GRACE_PERIOD:-}" ]]; then
        DURATION_ARGS+=(--benchmark-grace-period "$BENCHMARK_GRACE_PERIOD")
    fi
else
    DURATION_ARGS+=(--request-count "$((CONC * 10))" --warmup-request-count "$((CONC * 2))")
fi

BENCH_SERVING_DIR="${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}"

AIPERF_CMD=(
    python3 "${BENCH_SERVING_DIR}/utils/bench_serving/aiperf_adapter.py"
    --model "$MODEL"
    --url "http://0.0.0.0:${PORT}"
    --endpoint-type chat
    --isl "$ISL"
    --osl "$OSL"
    --result-filename "$RESULT_FILENAME"
    --result-dir /workspace/
    --server-metrics-url "http://0.0.0.0:${PORT}/metrics"
    "${DURATION_ARGS[@]}"
)
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    AIPERF_CMD+=("${SEARCH_ARGS[@]}")
else
    AIPERF_CMD+=(--concurrency "$CONC")
fi

ensure_aiperf

"${AIPERF_CMD[@]}"
BENCHMARK_EXIT_CODE=$?

stop_gpu_monitor
set +x
exit "$BENCHMARK_EXIT_CODE"
