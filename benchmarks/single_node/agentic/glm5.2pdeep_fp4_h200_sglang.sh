#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.2 W4AFP8 prod-replica + DeepEP MoE benchmark: identical to
# glm5.2prod_fp4_h200_sglang.sh except MoE runs EP8 via DeepEP
# (--moe-a2a-backend deepep, deepep_mode auto: low_latency decode +
# normal prefill). Runs on lmsysorg/sglang:dev which carries the
# W4AFP8+DeepEP scaling fix (#33669) and DeepEP release wheels (#34041).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING

CACHE_ARGS=()
if [ "$KV_OFFLOADING" = "dram" ]; then
  require_agentic_kv_offload_backend hicache
  CACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-size 128
    --hicache-io-backend direct
    --hicache-write-policy write_back
  )
else
  require_agentic_kv_offload_none
fi

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

export MODEL_PATH=/models/PhalaCloud/GLM-5.2-W4AFP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

# Prod env parity
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1

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
MAX_RUNNING_REQUESTS=256
PARALLEL_ARGS=(--tp-size "$TP")
GRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
  [ "$MAX_RUNNING_REQUESTS" -lt "$TP" ] && MAX_RUNNING_REQUESTS=$TP
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp 4
    --ep "$EP_SIZE"
    --enable-dp-attention
    --enable-dp-attention-local-control-broadcast
    --enable-dp-lm-head
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    --numa-node 0 0 0 0 1 1 1 1
  )
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --moe-a2a-backend deepep
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.75
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  "${GRAPH_ARGS[@]}"
  --context-length 500000
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  --schedule-policy fcfs
  --enable-prefill-delayer
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
