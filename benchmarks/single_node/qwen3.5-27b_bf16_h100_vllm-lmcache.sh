#!/usr/bin/env bash
# Qwen3.5-27B BF16 on H100 (single node) via vLLM + LMCache (local CPU-tier).
#
# Uses LMCacheMPConnector which supports Qwen3.5-27B's GDN/Mamba hybrid layers.
# LMCacheConnectorV1 does not implement SupportsHMA and crashes at startup for
# this model (ValueError: Hybrid KV cache manager is disabled...).
#
# Required for hybrid model compatibility:
#   --mamba-cache-mode align   GDN does not support 'all' mode
#   --enable-prefix-caching    needed for KV hit-rate measurement via /metrics
#   --max-num-batched-tokens   must be in [784, 1568) to match the GDN block size
#
# Requires BENCHMARK_CLIENT=aiperf.
#
# Counterpart config keys: qwen3.5-27b-bf16-h100-1x-vllm-lmcache-8k1k
#                          qwen3.5-27b-bf16-h100-1x-vllm-lmcache-8k1k-gn00

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
    echo "ERROR: qwen3.5-27b_bf16_h100_vllm-lmcache.sh requires BENCHMARK_CLIENT=aiperf" >&2
    exit 1
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
LMCACHE_LOG=/workspace/lmcache_server.log
PORT=${PORT:-8888}
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"
# Block size for Qwen3.5-27B GDN layers is 784 (derived at runtime by vLLM).
# --max-num-batched-tokens must be in [784, 1568) for mamba-cache-mode=align.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1567}"
LMCACHE_CPU_SIZE_GB="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
# Chunk size must match the vLLM-derived GDN block size (784 for Qwen3.5-27B BF16 tp=1).
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-784}"

start_gpu_monitor

# LMCacheMPConnector requires a separate lmcache server process.
lmcache server \
    --chunk-size "$LMCACHE_CHUNK_SIZE" \
    --l1-size-gb "$LMCACHE_CPU_SIZE_GB" \
    --eviction-policy LRU \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!

# Give the lmcache server a moment to initialize before vLLM connects.
sleep 3

set -x
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prime LMCache lookup counters before aiperf starts. OTel counters for
# lmcache_mp_lookup_* are lazy-initialized: they don't appear at port 8080
# until the first lookup request flows through LMCacheMPConnector. aiperf
# discovers metrics once at startup, so they must exist before it begins.
python3 -c "
import urllib.request, json, os
req = urllib.request.Request(
    'http://0.0.0.0:${PORT}/v1/chat/completions',
    data=json.dumps({'model': os.environ['MODEL'],
                     'messages': [{'role': 'user', 'content': 'hello'}],
                     'max_tokens': 1}).encode(),
    headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req, timeout=60)
except Exception:
    pass
" || true
sleep 2

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
    --server-metrics-url "http://0.0.0.0:8080/metrics"
    "${DURATION_ARGS[@]}"
)
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    AIPERF_CMD+=("${SEARCH_ARGS[@]}")
else
    AIPERF_CMD+=(--concurrency "$CONC")
fi

# Snapshot the raw metrics endpoints so we can verify which LMCache metric
# names are actually exposed (port 8888 = vLLM, port 8080 = lmcache server).
curl -sf "http://0.0.0.0:${PORT}/metrics" | grep -E "^[^#]" | grep -i lmcache \
    > /workspace/vllm_lmcache_metrics_snapshot.txt 2>&1 || true
curl -sf "http://0.0.0.0:8080/metrics" \
    > /workspace/lmcache_server_metrics_snapshot.txt 2>&1 || true

ensure_aiperf

"${AIPERF_CMD[@]}"
BENCHMARK_EXIT_CODE=$?

# Capture the LMCache server metrics AFTER the benchmark so lazy-initialized
# lookup counters (lmcache_mp_lookup_*) are visible in the artifact.
curl -sf "http://0.0.0.0:8080/metrics" \
    > /workspace/lmcache_server_post_metrics_snapshot.txt 2>&1 || true

# Export aiperf's server_metrics_export.json (what aiperf actually scraped and
# used to compute cache stats) for debugging. Named as _snapshot.txt so the
# existing artifact upload glob (*_metrics_snapshot.txt) picks it up.
cp "/workspace/${RESULT_FILENAME}_aiperf/server_metrics_export.json" \
   /workspace/aiperf_server_metrics_snapshot.txt 2>/dev/null || true

stop_gpu_monitor
set +x
exit "$BENCHMARK_EXIT_CODE"
