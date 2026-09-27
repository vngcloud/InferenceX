#!/usr/bin/env bash
# Shared body for the GLM-5.3-Flash FP8 4xH200 phase-2 lever arms. Each arm is a
# 2-line wrapper glm53flash-<arm>_fp8_h200_sglang_mtp.sh that sets ARM and
# sources this file, so the per-arm delta lives in one case statement below.
# Plan + verified flags: InferenceOptimization docs/handoffs/
# 2026-09-26-glm53flash-next-phase-test-plan.md §b.
#
# Base = the agentic twin (60f53616 glm5.3flash_fp8_h200_sglang_mtp.sh, run
# 34683194686) with image nightly-dev-20260926-1f6ce4b0 and MoE EP1:
#   - EP1 is mandatory with MTP on this image: sgl-project/sglang#40156 (from
#     #39574, fix #40172 unmerged) silently corrupts MoE dispatch when
#     moe_ep_size>1 under EAGLE.
#   - EP1 + deep_gemm boots on 1f6ce4b0: deep_gemm.py falls back to the compact
#     layout when D//8 < E ("DeepGEMM masked standard layout disabled").
# Arms (cumulative):
#   a0ep1  base only (separates image+EP+harness from the levers)
#   a1b    + extra_buffer_lazy + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK + bf16 ssm
#          (3 slots/req, mamba cap ~75 vs 31)
#   a2     + --enable-linear-replayssm-spec (#40517)
#   a4     + --schedule-policy hrrn (#32911)
#   a3     a2 + KV fp8_e4m3. No image supports fp8 KV on SM90 DSA yet, so
#          patches/glm53flash-a3-pr36904.patch (port of #36904 onto
#          1f6ce4b0) is applied to the in-container sglang checkout before boot.
#   dpa    a2 + DP-attention dp4 + sglang_router cache_aware. DPA disables
#          adaptive spec (adaptive_spec_params.py), so MTP is static 5/1/6.
#          --dp-size is required (#36840 silently resets DPA without it),
#          --mm-enable-dp-encoder is required (#36802 warmup hang).
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars ARM MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

FLASH_SNAPSHOT=/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
if [ -s "$FLASH_SNAPSHOT/config.json" ]; then
  export MODEL_PATH="$FLASH_SNAPSHOT"
else
  export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.3-Flash'))")
fi
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
# Slurm launcher exports its own DCGM sidecar port; docker path uses 9400.
export AIPERF_GPU_TELEMETRY_URL="${AIPERF_GPU_TELEMETRY_URL:-http://localhost:9400/metrics}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A slurm 4-GPU slice may be GPU0-3 (NUMA0) or GPU4-7 (NUMA1); bind to the
# NUMA node of the first visible GPU so concurrent arms on the two halves
# stay comparable.
GPU0_BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 | tr 'A-F' 'a-f' | sed 's/^0000//')
NUMA=$(cat "/sys/bus/pci/devices/$GPU0_BUS/numa_node" 2>/dev/null || echo 0)
[ "$NUMA" -ge 0 ] 2>/dev/null || NUMA=0

SPEC_ARGS=(
  --speculative-algorithm EAGLE
  --speculative-num-steps 5
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6
)
ADAPTIVE_ARGS=(--speculative-adaptive)
LEVER_ARGS=()
PARALLEL_ARGS=(--tp-size "$TP" --ep-size 1)
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32
USE_ROUTER=false
KV_DTYPE=bfloat16

MAMBA_ARGS=(--mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16)
case "$ARM" in
  a0ep1) ;;
  a1b) LEVER_ARGS=("${MAMBA_ARGS[@]}") ;;
  a2)  LEVER_ARGS=("${MAMBA_ARGS[@]}" --enable-linear-replayssm-spec) ;;
  a4)  LEVER_ARGS=("${MAMBA_ARGS[@]}" --enable-linear-replayssm-spec --schedule-policy hrrn) ;;
  a3)
    LEVER_ARGS=("${MAMBA_ARGS[@]}" --enable-linear-replayssm-spec)
    KV_DTYPE=fp8_e4m3
    SGL_ROOT=$(python3 -c "import os, sglang; print(os.path.dirname(os.path.dirname(os.path.dirname(sglang.__file__))))")
    patch -p1 --forward -d "$SGL_ROOT" < "$(dirname "$0")/patches/glm53flash-a3-pr36904.patch"
    ;;
  dpa)
    LEVER_ARGS=("${MAMBA_ARGS[@]}" --enable-linear-replayssm-spec)
    ADAPTIVE_ARGS=()
    PARALLEL_ARGS=(
      --tp-size "$TP" --dp-size "$TP" --enable-dp-attention
      --enable-dp-attention-local-control-broadcast --enable-dp-lm-head
      --mm-enable-dp-encoder --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    )
    # Node total, split per rank; mamba clamp sets the real cap.
    MAX_RUNNING_REQUESTS=64
    USE_ROUTER=true
    export SGLANG_DP_USE_GATHERV=1 NCCL_P2P_LEVEL=NVL SGLANG_ENABLE_METRICS_DP_ATTENTION=1
    ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
[ "$ARM" != a0ep1 ] && export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1

SERVER_PORT=$PORT
$USE_ROUTER && SERVER_PORT=$((PORT + 1))
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SERVER_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name "$MODEL"
  --host 0.0.0.0
  --port "$SERVER_PORT"
  "${PARALLEL_ARGS[@]}"
  --moe-runner-backend deep_gemm
  --numa-node "$NUMA" "$NUMA" "$NUMA" "$NUMA"
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.75
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length 300000
  --allow-auto-truncate
  --kv-cache-dtype "$KV_DTYPE"
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-cache-report
  --enable-hierarchical-cache
  --hicache-size 32
  "${SPEC_ARGS[@]}"
  "${ADAPTIVE_ARGS[@]}"
  "${LEVER_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SERVER_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if $USE_ROUTER; then
  ROUTER_LOG="$RESULT_DIR/router.log"
  python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SERVER_PORT" \
    --policy cache_aware \
    --dp-aware \
    --balance-abs-threshold 8 \
    --request-id-headers x-correlation-id \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$((PORT + 10000))" \
    --connect-timeout-secs 900 \
    --request-timeout-secs 14400 \
    --disable-health-check \
    --disable-retries > "$ROUTER_LOG" 2>&1 &
  ROUTER_PID=$!
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
