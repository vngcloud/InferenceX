#!/usr/bin/env bash
set -euo pipefail
set -x
#
# GLM-5.3-W4AFP8 adaptive-spec-under-DPA campaign: static EAGLE arm.
# Same engine/router recipe as glm5.3proddp8deep521 (baseline run 37189063255,
# InferenceX ref 9fbe6df5), ported onto stock vng-benchmark: weights come from
# the canonical HF hub cache (PhalaCloud/GLM-5.3-W4AFP8@03179e95, staged on
# hgx-h200-02 and hash-verified against the HF revision) instead of the
# /data/hf-cache plain dir the old launcher mounted.
# SPEC_NUM_STEPS (default 3, draft tokens = steps + 1) lets thin per-k wrapper
# recipes run the static k sweep without duplicating this file.
# Spec: InferenceOptimization/docs/specs/2026-10-12-adaptive-spec-dpa.md

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

export MODEL_PATH=PhalaCloud/GLM-5.3-W4AFP8
MODEL_REVISION=03179e95916f9f79a92536e76da63f9251d93d4f
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SNAPSHOT_DIR="$HF_HUB_CACHE/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/$MODEL_REVISION"
[ -f "$SNAPSHOT_DIR/config.json" ] || { echo "FATAL: $SNAPSHOT_DIR/config.json not found: stage PhalaCloud/GLM-5.3-W4AFP8@$MODEL_REVISION into the node's HF hub cache first" >&2; exit 1; }
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
# The slurm launcher exports its own DCGM sidecar port; the docker path uses 9400.
export AIPERF_GPU_TELEMETRY_URL="${AIPERF_GPU_TELEMETRY_URL:-http://localhost:9400/metrics}"

# Prod env parity
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# hardware-hcm fabric: the 8 p2p RoCE links are Intel E810 (irdma), which
# NVSHMEM's IBRC transport cannot probe — DeepEP low-latency bring-up segfaults
# at main_nvshmem/.../ibrc.cpp:314 "NULL value" (smoke run 37134667065). A
# single-node TP8 job has no remote PEs, so disable the remote transport
# entirely; same env nguyennvc's dispatch-l3 GLM-5.3-W4AFP8 tp8/ep8 DeepEP
# engines run with on these exact nodes.
export NVSHMEM_REMOTE_TRANSPORT=none
# The silent smoke deaths were per-NUMA-zone exhaustion (delta 4), not the
# registration call: the default single 128G cudaHostRegister is fine once
# the zones have headroom (manual boot 1584), so registration chunking stays
# at the sglang default. Print native fault stacks if anything still dies.
export PYTHONFAULTHANDLER=1
# VRAM headroom: at mem-fraction 0.75 every GPU serves with only 0.05-0.9
# GiB free (ladder 37145396223: c32 ran with 26 recoverable torch OOM
# warnings and 7 lazy Triton loads at ~0.96 GiB free; c16 died at 0.05 GiB
# when the top-p bitmask kernel lazy-loaded mid-serve and NCCL could not
# cudaMalloc its 512MB sampler all_reduce buffer; retry 37176412590 was green
# but still served at 0.52-0.89 GiB free with 17 OOM warnings). Expandable
# segments stop the caching allocator from stranding freed blocks — the same
# env nguyennvc's dispatch-l3 engines run on these nodes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --speculative-num-steps "$SPEC_NUM_STEPS"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$((SPEC_NUM_STEPS + 1))"
  )
  # Extra spec flags from wrapper recipes (e.g. --speculative-adaptive); empty here.
  read -r -a _SPEC_EXTRA <<< "${SPEC_EXTRA_ARGS:-}"
  SPEC_ARGS+=("${_SPEC_EXTRA[@]}")
fi

SGLANG_BACKEND_PORT="$PORT"
ROUTER_ENABLED=false
if [ -n "${ROUTER_METADATA:-}" ]; then
  ROUTER_ENABLED=true
  SGLANG_BACKEND_PORT=$((PORT + 1))
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

# Slurm can hand us GPUs while the previous job's processes still hold VRAM
# (runs 36322826747/36322829133 died at boot on "memory capacity is
# unbalanced"). Wait up to 10 min for every visible GPU to drop under 2 GiB.
for _ in $(seq 60); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {busy=1} END {exit busy}' && break
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -lt 256 ] && MAX_RUNNING_REQUESTS=256
PARALLEL_ARGS=(--tp-size "$TP")
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
    --numa-node 0 1 2 3 0 1 2 3
  )
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --revision "$MODEL_REVISION"
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

# 64MB stack, as the prod containers and nguyennvc's dispatch-l3 engines run
# (the sbatch/enroot default is 8MB).
ulimit -s 65536

# Offline for the server only: the snapshot is staged locally, while aiperf
# may still fetch its tokenizer/dataset from the hub.
HF_HUB_OFFLINE=1 "${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prod router args (mirror the live glm-52-fp8-h200-8x-router sidecar), run
# in-container: no docker socket under enroot/pyxis.
if [ "$ROUTER_ENABLED" = "true" ]; then
  ROUTER_LOG="$RESULT_DIR/router.log"
  python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy cache_aware \
    --dp-aware \
    --cache-threshold 0.3 \
    --balance-abs-threshold 100000 \
    --balance-rel-threshold 2.0 \
    --max-tree-size 67108864 \
    --eviction-interval-secs 300 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$((PORT + 1000))" \
    --health-check-interval-secs 15 \
    --health-check-timeout-secs 10 \
    --health-failure-threshold 5 \
    --request-timeout-secs 900 \
    --retry-max-retries 2 > "$ROUTER_LOG" 2>&1 &
  ROUTER_PID=$!
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
