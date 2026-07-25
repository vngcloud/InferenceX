#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION
require_agentic_kv_offload_backend hicache

SPEC_DECODING="${SPEC_DECODING:-none}"

export MODEL_PATH="$HF_HUB_CACHE/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1"
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
# ponytail: spec-decode is prefill-bursty + long-context; borrow B300 recipe's crash/timeout guards
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
  USE_SGLANG_ROUTER=true
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
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
PARALLEL_ARGS=(--tp-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp 2
    --enable-dp-attention
    --moe-a2a-backend deepep
  )
fi

# EAGLE spec-decode: draft head is baked into the GLM-5.2 checkpoint (no separate draft model).
# Guards mirror glm5.2_fp8_b300-netperf recipe's spec path: size the decode cuda-graph to the full
# running-request budget (verification runs draft-tokens per req) and enable crash dumps + watchdog.
SPEC_ARGS=()
CRASH_ARGS=()
WATCHDOG_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
  CRASH_DUMP_FOLDER="$RESULT_DIR/crash_dumps"
  mkdir -p "$CRASH_DUMP_FOLDER"
  CRASH_ARGS=(--crash-dump-folder "$CRASH_DUMP_FOLDER")
  WATCHDOG_ARGS=(--watchdog-timeout 1800)
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --chunked-prefill-size 16384
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.85
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  --context-length 500000
  --kv-cache-dtype fp8_e4m3
  --allow-auto-truncate
  --enable-metrics
  --enable-cache-report
  --enable-hierarchical-cache
  --hicache-size 128
  --schedule-policy lpm
  "${SPEC_ARGS[@]}"
  "${CRASH_ARGS[@]}"
  "${WATCHDOG_ARGS[@]}"
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
    --policy cache_aware \
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
