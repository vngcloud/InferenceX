#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.2 W4AFP8 prod-config low-latency (TP-only) benchmark on 8xH200.
# HiCache "kernel" I/O variant of glm5.2prodll: identical in every way EXCEPT the
# HiCache CPU<->GPU transfer path. Per the SGLang HiCache cookbook, the kernel
# I/O backend uses GPU-assisted transfer kernels (~3x faster L2<->GPU) and requires
# the page_first host layout. prodll uses direct + page_first_direct; this arm flips
# only io-backend (direct->kernel) + mem-layout (page_first_direct->page_first) to
# isolate that one lever's effect on cache-hit TTFT. A/B pair vs glm5.2prodll.
#
# Takes the prod-exact proddp8 server config and strips dp-attention: pure TP8,
# single container, NO router. Deviations from proddp8:
#   - no --dp / --enable-dp-attention / --ep (TP-only), so no sglang router
#   - EAGLE spec 5/1/6 (vs 3/1/4)
#   - --max-running-requests capped at 32 (matches prod --mem-fraction-static 0.75)
#   - no --enable-prefill-delayer
#   - server image community lmsysorg/sglang:v0.5.18 (config image key)
# Everything else mirrors prod 1:1: w4afp8, HiCache 128GB/rank direct write_back,
# flashmla_sparse_q8 DSA prefill, dfs-weight schedule, chunked-prefill 32768,
# context-length 300000, kv fp8_e4m3, glm47/glm45 parsers.
#
# Model auto-resolves from HF cache: snapshot_download(PhalaCloud/GLM-5.2-W4AFP8)
# hits the pre-warmed /mnt/hf_hub_cache on the runner, misses fetch on demand.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

CACHE_ARGS=(
  --enable-hierarchical-cache
  --hicache-size 128
  --hicache-io-backend kernel
  --hicache-mem-layout page_first
  --hicache-write-policy write_back
)

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
  )
fi

# Auto-download / resolve W4AFP8 from HF cache (pre-warmed on runner /mnt/hf_hub_cache).
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('PhalaCloud/GLM-5.2-W4AFP8'))")
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

# Prod env parity (SGLANG_DP_USE_GATHERV and SGLANG_ENABLE_METRICS_DP_ATTENTION
# are no-ops without dp-attention; kept so the arm matches prod env 1:1).
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# DSA indexer's fp8_mqa_logits needs a large transient buffer that scales with
# context length; at long ctx it OOM'd on reserved-but-unallocated fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Low-latency running-batch ceiling of 32 (vs proddp8's max(2*CONC,256) floor).
# In-flight never exceeds CONC, so 32 is a cap, not a forced batch: for CCU
# 1/4/8 the 2*CONC term (2/8/16) already sits below 32 and only CCU 16 hits it.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --host 0.0.0.0
  --port "$PORT"
  --tp-size "$TP"
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
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
