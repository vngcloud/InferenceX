#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION EP_SIZE DP_ATTENTION SPEC_DECODING
require_agentic_kv_offload_backend hicache

export MODEL_PATH=/mnt/models/nvidia/GLM-5.2-NVFP4
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
SGLANG_BACKEND_PORT="$PORT"
MAX_RUNNING_REQUESTS=$((2 * CONC))
USE_SGLANG_ROUTER=false
ROUTER_POLICY=consistent_hashing
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    if [ "$SPEC_DECODING" = "mtp" ]; then
        ROUTER_POLICY=cache_aware
    else
        export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    fi
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
CHUNKED_PREFILL_SIZE=8192
GRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    CHUNKED_PREFILL_SIZE=32768
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --tokenizer-worker-num "$TP"
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    )
else
    PARALLEL_ARGS+=(
        --kv-cache-dtype fp8_e4m3
        --bf16-gemm-backend cutedsl
        --max-prefill-tokens 8192
    )
    CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
    [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
    GRAPH_ARGS=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS")
fi

SPEC_ARGS=()
CRASH_ARGS=()
MEM_FRACTION_STATIC=0.92
HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
if [ "$SPEC_DECODING" = "mtp" ]; then
    MEM_FRACTION_STATIC=0.85
    GRAPH_ARGS=(--cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS")
    SPEC_ARGS=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
    CRASH_DUMP_FOLDER="$RESULT_DIR/crash_dumps"
    mkdir -p "$CRASH_DUMP_FOLDER"
    CRASH_ARGS=(--crash-dump-folder "$CRASH_DUMP_FOLDER")
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --quantization modelopt_fp4
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --tool-call-parser glm47
    --reasoning-parser glm45
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --schedule-policy lpm
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${GRAPH_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${CRASH_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
    --enable-cache-report
    --enable-hierarchical-cache
    --hicache-ratio "$HICACHE_RATIO"
    --hicache-write-policy write_back
    --hicache-io-backend direct
    --hicache-mem-layout page_first_direct
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    python3 -m sglang_router.launch_router \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy "$ROUTER_POLICY" \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --connect-timeout-secs 900 \
        --request-timeout-secs 14400 \
        --disable-health-check \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
