#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION

CACHE_ARGS=()
if [ "$KV_OFFLOADING" = "dram" ]; then
  require_agentic_kv_offload_backend hicache
  CACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-size 128
  )
else
  require_agentic_kv_offload_none
fi

export MODEL_PATH=/models/PhalaCloud/GLM-5.2-W4AFP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
  USE_SGLANG_ROUTER=true
  export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
  SGLANG_BACKEND_PORT=$((PORT + 1))
  SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
CHUNKED_PREFILL_SIZE=8192
PARALLEL_ARGS=(--tp-size "$TP")
GRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
  CHUNKED_PREFILL_SIZE=32768
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp "$TP"
    --enable-dp-attention
    --moe-a2a-backend deepep
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
  )
else
  CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
  [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
  GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --disable-shared-experts-fusion
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.85
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  "${GRAPH_ARGS[@]}"
  --context-length 500000
  --kv-cache-dtype fp8_e4m3
  --allow-auto-truncate
  --enable-metrics
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  --schedule-policy lpm
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
  python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy consistent_hashing \
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
