#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

# 4xH200 GLM-5.2-W4AFP8: TP=4 DP=4 EP=4, local HiCache (no shm_peer_l2), EAGLE, round_robin router.
# Comparison recipe: round_robin (no dp-aware/cache_aware) + local L2 only, to establish
# baseline prefix_cache_hit without cross-rank sharing or router-level prefix steering.

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING

require_agentic_kv_offload_backend hicache

export MODEL_PATH=/models/PhalaCloud/GLM-5.2-W4AFP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL

USE_SGLANG_ROUTER=true
SGLANG_BACKEND_PORT=$((PORT + 1))
SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
ROUTER_LOG="$RESULT_DIR/router.log"
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -lt "$TP" ] && MAX_RUNNING_REQUESTS=$TP
[ "$MAX_RUNNING_REQUESTS" -gt 128 ] && MAX_RUNNING_REQUESTS=128

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --served-model-name "$MODEL"
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  --tp "$TP"
  --dp 4
  --ep "$EP_SIZE"
  --enable-dp-attention
  --enable-dp-attention-local-control-broadcast
  --enable-dp-lm-head
  --tokenizer-worker-num "$TP"
  --dist-init-addr "127.0.0.1:$((PORT + 2000))"
  --numa-node 0 0 0 0
  --chunked-prefill-size 16384
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.88
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length 131072
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics
  --enable-cache-report
  --enable-hierarchical-cache
  --hicache-size 64
  --hicache-io-backend direct
  --hicache-write-policy write_through
  --hicache-mem-layout page_first_direct
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --schedule-policy fcfs
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

python3 -m sglang_router.launch_router \
  --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
  --policy round_robin \
  --balance-abs-threshold 8 \
  --request-id-headers x-correlation-id \
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

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
