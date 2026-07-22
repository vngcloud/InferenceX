#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_backend hicache

export MODEL_PATH="$HF_HUB_CACHE/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1"
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host 0.0.0.0
  --port "$PORT"
  --tp-size "$TP"
  --chunked-prefill-size 8192
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
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
