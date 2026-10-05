#!/usr/bin/env bash
set -eo pipefail
set -x

# GLM-5.3-W4AFP8 TP-only low-latency (prodll) + DFlash2 draft on hardware-hcm.
# Base: glm5.3proddp8deep521hrrn (e5513e75, smoke 37203342192 + ladder
# 37208036979 green on hardware-hcm-8x_02), with the DP8/DeepEP/router parts
# stripped (DFLASH rejects DP-Attention, so the recipe is pure TP8 like the
# han-box glm5.2prodll low-latency arm, run 34197652789) and the spec
# decoding switched from built-in EAGLE/MTP to DFlash2:
#   - draft = incoai/GLM-5.3-DFlash2 (BF16 ~4.9GB, block_size 8 + conv/
#     selector geometry from its own dflash_config — no
#     --speculative-num-draft-tokens), resolved via snapshot_download from
#     the mounted /data/thanglq5/hf-cache/hub (pre-staged by sbatch
#     dflash2-dl; downloads on demand if missing).
#   - --speculative-draft-attention-backend fa4: the official GLM-5.3
#     DFlash2 recipe; the draft is a small dense model and does not run the
#     target's DSA backends.
#   - --speculative-draft-model-quantization unquant: sglang auto-inherits
#     the target's --quantization (w4afp8) into the draft when unset, which
#     would load the BF16 draft as w4afp8 and fail.
# DFLASH needs >= sglang v0.5.19 (DFlash2DraftModel, PR #35371); image is
# v0.5.21. Everything else mirrors the green W4AFP8 pool config: weights
# /data/hf-cache/GLM-5.3-W4AFP8, hicache 128GB/rank direct write_back,
# flashmla_sparse_q8 DSA prefill, kv fp8_e4m3, ctx 300000, glm45/glm47
# parsers, mem-frac 0.75, NUMA spread 0 1 2 3 0 1 2 3 (2 ranks per 504G
# zone), expandable_segments (c16 died at 0.05 GiB free without it,
# 37145396223).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT AIPERF_GPU_TELEMETRY_URL
require_agentic_kv_offload_backend hicache

MODEL_DIR=/data/hf-cache/GLM-5.3-W4AFP8
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: $MODEL_DIR/config.json not found — stage PhalaCloud/GLM-5.3-W4AFP8 on the node first" >&2; exit 1; }
export MODEL_PATH="$MODEL_DIR"

# DFlash2 draft: public, ~4.9GB BF16 (config.json + model.safetensors; it has no
# tokenizer — the target's tokenizer serves it). Uses the staged plain dir on
# the node-local cache when present (curl-staged by the sbatch cleanup2-dl job);
# otherwise snapshot_download fetches it on demand (small enough in-run).
DRAFT_STAGED_DIR=/data/thanglq5/hf-cache/GLM-5.3-DFlash2
if [ -f "$DRAFT_STAGED_DIR/model.safetensors" ] && [ -f "$DRAFT_STAGED_DIR/config.json" ]; then
  export DRAFT_MODEL_PATH="$DRAFT_STAGED_DIR"
else
  export DRAFT_MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('incoai/GLM-5.3-DFlash2'))")
fi
[ -f "$DRAFT_MODEL_PATH/config.json" ] || { echo "FATAL: DFlash2 draft config.json not found at $DRAFT_MODEL_PATH" >&2; exit 1; }

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
# Set by the launcher (its own DCGM sidecar port); validated above.

# Prod env parity (no-ops without dp-attention; kept so the arm matches the
# green W4AFP8 arms env 1:1).
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

# Low-latency running-batch ceiling of 32 (glm5.2prodll convention): in-flight
# never exceeds CONC, so this is a cap, not a forced batch.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32

SPEC_ARGS=()
case "$SPEC_DECODING" in
  draft_model)
    SPEC_ARGS=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFT_MODEL_PATH"
      --speculative-draft-attention-backend fa4
      --speculative-draft-model-quantization unquant
    )
    ;;
  *)
    echo "FATAL: this arm is DFlash2-only (SPEC_DECODING must be draft_model, got '$SPEC_DECODING')" >&2
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
