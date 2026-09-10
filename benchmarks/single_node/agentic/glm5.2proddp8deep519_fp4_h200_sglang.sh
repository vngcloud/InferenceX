#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.2 W4AFP8 prod-exact DP8 benchmark, v0.5.19 pin of glm5.2proddp8deep.
# Byte-identical to glm5.2proddp8deep_fp4_h200_sglang.sh (reference run
# 34151182945, v0.5.18, 2026-09-07) except the header you are reading: the
# single change is the server image, lmsysorg/sglang:v0.5.19 instead of
# v0.5.18, set via the config `image:` key. Pinned as its own model-prefix
# (glm5.2proddp8deep519) so later edits to the 0.5.18 arm cannot silently
# redefine this A/B and so the result series stay separable.
# --hicache-size is PER-RANK GB: 8 ranks * 128 = 1024 GB host RAM total, guard
# N <= free_RAM/8 (~182 GiB/rank headroom) holds.
#
# Prod-exact 2-container topology (same as proddp8deep). Router stays the
# v0.5.18 router patch build sidecar (sglang-router-patched:v0.5.18-slim-
# noargcanon): no v0.5.19 router patch build exists in VCR, and the router
# talks plain HTTP to the server (worker-urls + /metrics), so it is not
# version-coupled to the server image. The v0.5.19 server image does ship
# its own Rust smg binary, but the sidecar topology is kept for prod parity.
# Router args mirror the live glm-52-fp8-h200-8x-router deployment.
#
# Model auto-resolves from HF cache: snapshot_download(PhalaCloud/GLM-5.2-W4AFP8)
# hits the pre-warmed /mnt/hf_hub_cache on the runner, misses fetch on demand.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING PORT USE_PROD_ROUTER ROUTER_IMAGE
require_agentic_kv_offload_backend hicache
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker CLI not mounted (launcher USE_PROD_ROUTER detection failed)" >&2; exit 1; }

# Keep the v0.5.18 router patch build sidecar (see header): router<->server is
# plain HTTP, and no v0.5.19 router patch build exists to pair with.
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
  # Router runs detached, so it never writes $ROUTER_LOG itself and $ROUTER_CID
  # is a container id, not a pid (kill -0 on a 64-hex id always reads as dead).
  # Stream the container logs into $ROUTER_LOG and hand wait_for_server_ready the
  # numeric pid of the `docker logs -f` follower, which exits iff the container dies.
  docker logs -f "$ROUTER_CID" > "$ROUTER_LOG" 2>&1 &
  ROUTER_LOG_PID=$!
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_LOG_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
