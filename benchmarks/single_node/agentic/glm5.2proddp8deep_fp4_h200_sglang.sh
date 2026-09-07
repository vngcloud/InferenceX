#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.2 W4AFP8 prod-exact DP8 benchmark, DeepEP variant of glm5.2proddp8.
# Identical to glm5.2proddp8_fp4_h200_sglang.sh (--dp 8, HiCache 128GB/rank
# direct write_back, EAGLE mtp, flashmla_sparse_q8 DSA prefill, dfs-weight,
# prefill-delayer, context-length 300000, prod env parity) with two changes:
#   1. Server image is the community lmsysorg/sglang:v0.5.18 (not VCR v0.5.16),
#      set via the config `image:` key.
#   2. MoE runs EP8 via DeepEP (--moe-a2a-backend deepep + --ep 8). deepep_mode
#      defaults to auto: low_latency decode + normal prefill.
# --hicache-size is PER-RANK GB: 8 ranks * 128 = 1024 GB host RAM total, guard
# N <= free_RAM/8 (~182 GiB/rank headroom) holds.
#
# Prod-exact 2-container topology (same as proddp8). The server image has no
# Rust smg binary, so the recipe launches the router as a host-network sidecar
# via the docker socket (mounted by the launcher on the *proddp8* name match).
# This arm overrides ROUTER_IMAGE to the v0.5.18 router patch build
# (sglang-router-patched:v0.5.18-slim-noargcanon) to pair with the 0.5.18 server.
# Router args mirror the live glm-52-fp8-h200-8x-router deployment.
#
# Model auto-resolves from HF cache: snapshot_download(PhalaCloud/GLM-5.2-W4AFP8)
# hits the pre-warmed /mnt/hf_hub_cache on the runner, misses fetch on demand.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING PORT USE_PROD_ROUTER ROUTER_IMAGE
require_agentic_kv_offload_backend hicache
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker CLI not mounted (launcher USE_PROD_ROUTER detection failed)" >&2; exit 1; }

# Pair the v0.5.18 community server with the matching v0.5.18 router patch build.
# The launcher defaults ROUTER_IMAGE to the v0.5.16 tag on the *proddp8* match;
# override it here so this arm runs the 0.5.18 router sidecar.
ROUTER_IMAGE=vcr.vngcloud.vn/60108-backend-worker/portal-external/dev/sglang-router-patched:v0.5.18-slim-noargcanon

CACHE_ARGS=(
  --enable-hierarchical-cache
  --hicache-size 128
  --hicache-io-backend direct
  --hicache-write-policy write_back
)

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

# Auto-download / resolve W4AFP8 from HF cache (pre-warmed on runner /mnt/hf_hub_cache).
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('PhalaCloud/GLM-5.2-W4AFP8'))")
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
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -lt 256 ] && MAX_RUNNING_REQUESTS=256
PARALLEL_ARGS=(--tp-size "$TP")
GRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
  [ "$MAX_RUNNING_REQUESTS" -lt "$TP" ] && MAX_RUNNING_REQUESTS=$TP
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp 8
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
  --context-length 300000
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  --schedule-policy dfs-weight
  --enable-prefill-delayer
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prod-exact 2-container router: Rust smg sidecar via docker socket.
ROUTER_CID=""
if [ "$USE_PROD_ROUTER" = "true" ]; then
  ROUTER_NAME="router-${RUNNER_NAME:-bench}-$$"
  ROUTER_CID=$(docker run -d --rm --network host \
    --label inferencex-bench=1 \
    --name "$ROUTER_NAME" \
    "$ROUTER_IMAGE" \
    launch \
    --host=0.0.0.0 --port="$PORT" --prometheus-port="$SGLANG_ROUTER_METRICS_PORT" \
    --policy=cache_aware --dp-aware \
    --worker-urls="http://localhost:$SGLANG_BACKEND_PORT" \
    --cache-threshold=0.3 --balance-abs-threshold=100000 --balance-rel-threshold=2.0 \
    --max-tree-size=67108864 --eviction-interval=300 \
    --health-check-interval-secs=15 --health-check-timeout-secs=10 --health-failure-threshold=5 \
    --request-timeout-secs=900 --retry-max-retries=2)
  trap 'docker rm -f "$ROUTER_CID" 2>/dev/null || true' EXIT
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_CID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
