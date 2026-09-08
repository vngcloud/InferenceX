#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.2 FP8 (8-bit block-wise, zai-org/GLM-5.2-FP8) DeepEP DP8 prod-exact
# benchmark. Same prod-exact DP8 topology as glm5.2proddp8deep_fp4_h200_sglang.sh
# (--dp 8, --ep 8 DeepEP, HiCache 128GB/rank direct write_back, EAGLE mtp 3/1/4,
# flashmla_sparse_q8 DSA prefill, dfs-weight, prefill-delayer, context-length
# 300000, prod env parity SGLANG_DP_USE_GATHERV/NCCL_P2P_LEVEL=NVL/
# SGLANG_ENABLE_METRICS_DP_ATTENTION) with three deviations from the fp4
# (W4AFP8) sibling:
#   1. Weights = zai-org/GLM-5.2-FP8 (official FP8 8-bit, e4m3, weight_block_size
#      [128,128], activation_scheme dynamic) instead of PhalaCloud/GLM-5.2-W4AFP8.
#      sglang auto-detects quant_method=fp8 from config.json (verified in
#      benchmarks/single_node/agentic/glm5.2_fp8_b300-netperf_sglang.sh), so NO
#      --quantization flag is passed.
#   2. --mem-fraction-static 0.85 (vs 0.75). WARNING: FP8 weights are ~1.8x the
#      W4AFP8 size (~880GB/141 shards vs 400GB/40), so per-rank static budget is
#      cjommer than the w4afp8 arms. With the DSA sparse-indexer fp8_mqa_logits
#      transient scaling with context length (the same code path that OOMed the
#      prodll arm at mem 0.8 with W4AFP8), long-ctx (>50k) requests may OOM.
#      Mitigation: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces
#      fragmentation (recommended by the allocator traceback on the prodll OOM,
#      see docs/handoffs/2026-09-08-glm52-prodll-oom-memfrac-fix.md). Watch the
#      first warmup job; if OOM at long ctx, drop the fraction to 0.75/0.72 and
#      mark the arm as no longer prod-exact mem.
#   3. CCU ladder [1, 4, 8] (vs [1, 8, 16, 24, 32]) set via the config key
#      conc-list in nvidia-master.yaml. Lower-concurrency latency sweep, not a
#      throughput-saturation sweep.
# --max-running-requests follows prod-exact max(2*CONC, 256), so all three CCUs
# floor at 256 (the user's requested value).
# --hicache-size is PER-RANK GB: 8 ranks * 128 = 1024 GB host RAM total, guard
# N <= free_RAM/8 (~182 GiB/rank headroom) holds.
#
# Prod-exact 2-container topology (same as the fp4 proddp8* arms). The server
# image has no Rust smg binary, so the recipe launches the router as a
# host-network sidecar via the docker socket (mounted by the launcher on the
# *proddp8* name match, which this recipe's name matches). ROUTER_IMAGE is
# overridden to sglang-router-patched:v0.5.18-slim-noargcanon to pair with the
# 0.5.18 server. Router args mirror the live glm-52-fp8-h200-8x-router
# deployment.
#
# Model auto-resolves from HF cache: snapshot_download(zai-org/GLM-5.2-FP8)
# hits the pre-warmed /mnt/hf_hub_cache on the runner, misses fetch on demand.
# 141 shards * ~5GB = ~880GB download; must be pre-warmed before dispatch.

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

# Auto-download / resolve FP8 from HF cache (pre-warmed on runner /mnt/hf_hub_cache).
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.2-FP8'))")
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

# Prod env parity
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1

# Fragmentation mitigation: FP8 weights nearly fill per-rank budget at mem 0.85,
# and the DSA sparse-indexer allocates a context-length-scaled transient that
# caused the prodll arm to OOM at mem 0.8 with W4AFP8. expandable_segments lets
# the caching allocator recycle reserved-but-unallocated regions (8.72 GiB on
# the prodll traceback) before asking CUDA for more.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --moe-a2a-backend deepep
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.85
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
