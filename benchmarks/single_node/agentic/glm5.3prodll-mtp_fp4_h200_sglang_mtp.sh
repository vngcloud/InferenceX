#!/usr/bin/env bash
set -eo pipefail
set -x

# GLM-5.3-W4AFP8 TP-only low-latency (prodll) + built-in EAGLE/MTP 5-1-6 on
# hardware-hcm: the apples-to-apples MTP baseline for the DFlash2 arm
# (glm5.3prodll-dflash2, run 37294572283) — same target weights, image
# v0.5.21, hicache pool, NUMA spread and scheduler flags as that arm; the
# only delta is spec decoding: EAGLE 5/1/6 (built-in MTP head, the
# glm5.2prodll low-latency convention, run 34197652789) instead of DFLASH +
# external DFlash2 draft. Dispatched because every existing GLM-5.3 MTP run
# is DP-attention (glm5.3proddp8deep521hrrn ladder 37208036979) and thus not
# a valid low-latency comparison, and the GLM-5.2 baseline differs in both
# image and weights. NOTE: both v0.5.21 GLM-5.3 spec arms currently show the
# early-EOS/empty-response pathology (see handoff 2026-10-05-glm53-dflash2
# §2026-10-06) — treat throughput as provisional until that bug is fixed.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT AIPERF_GPU_TELEMETRY_URL
require_agentic_kv_offload_backend hicache

MODEL_DIR=/data/hf-cache/GLM-5.3-W4AFP8
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: $MODEL_DIR/config.json not found — stage PhalaCloud/GLM-5.3-W4AFP8 on the node first" >&2; exit 1; }
export MODEL_PATH="$MODEL_DIR"

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
# Set by the launcher (its own DCGM sidecar port); validated above.

# Prod env parity (no-ops without dp-attention; kept so the arm matches the
# DFlash2 arm env 1:1).
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# Single-node TP8 job has no remote PEs; the E810 irdma fabric breaks
# NVSHMEM's IBRC probe (same env as the green arms on this pool).
export NVSHMEM_REMOTE_TRANSPORT=none
export PYTHONFAULTHANDLER=1
# VRAM headroom: expandable segments stop the caching allocator from
# stranding freed blocks (c16 died at 0.05 GiB free without it).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

# Slurm can hand us GPUs while the previous job's processes still hold VRAM.
# Wait up to 10 min for every visible GPU to drop under 2 GiB.
for _ in $(seq 60); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {busy=1} END {exit busy}' && break
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Low-latency running-batch ceiling of 32 (glm5.2prodll / dflash2 convention):
# in-flight never exceeds CONC, so this is a cap, not a forced batch.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32

SPEC_ARGS=()
case "$SPEC_DECODING" in
  mtp)
    SPEC_ARGS=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 5
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 6
    )
    ;;
  *)
    echo "FATAL: this arm is MTP-only (SPEC_DECODING must be mtp, got '$SPEC_DECODING')" >&2
    exit 1
    ;;
esac

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --host 0.0.0.0
  --port "$PORT"
  --tp-size "$TP"
  --numa-node 0 1 2 3 0 1 2 3
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
  --enable-hierarchical-cache
  --hicache-size 128
  --hicache-io-backend direct
  --hicache-write-policy write_back
  "${SPEC_ARGS[@]}"
  --schedule-policy dfs-weight
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

# 64MB stack, as the prod containers run (sbatch/enroot default is 8MB).
ulimit -s 65536

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
