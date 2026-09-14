#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3-Flash (zai-org/GLM-5.3-Flash) FP8 agentic-coding trace-replay recipe
# on 4xH200, Arm D (mamba-ratio, agentic sizing). Copy of
# benchmarks/single_node/agentic/glm53flash-ab_fp8_h200_sglang_mtp.sh
# (mem 0.75 variant) with exactly 1 diff:
#   + --mamba-full-memory-ratio 0.4
# r derivation (agentic L, NOT the fixed 4.3): baseline agentic run
# 34683194686 c8 profile_export_aiperf.json -> input avg 95064 + output avg
# 903 = L~96000 tokens/request. Per-req costs from the mem-0.75 boot log (run
# 34805726628 c32): 5 mamba slots x 93.7MB = 468MB state, 96000 x 11.84KB =
# 1137MB KV. r* = 468/1137 = 0.4 (default 0.9 over-provisions mamba ~2x at
# agentic ctx; 0.4 frees ~4GB -> +350k KV tokens of prefix capacity). Mamba
# clamp at 0.4 ~= 22 reqs: fine for the [8,16] knee zone (c32 queues, as it
# already does at 31 today). Rest = AB: auto DSA, hicache 64 write_back,
# deep_gemm EP4, MTP adaptive 5/1/6, bf16 KV, glm45/glm47, numa 0x4, ctx
# 300000 + auto-truncate, chunked-prefill 32768, metrics + cache-report,
# expandable_segments, MAX_RUNNING_REQUESTS min(2*CONC,32).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

CACHE_ARGS=(
  --enable-hierarchical-cache
  --hicache-size 64
  --hicache-write-policy write_back
)

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
    --speculative-adaptive
  )
fi

# Auto-download / resolve from HF cache (pre-warmed on runner /mnt/hf_hub_cache,
# mounted rw at /root/.cache/huggingface/hub by the launcher). Pin the
# pre-warmed snapshot when present; fall back to snapshot_download for a cold
# runner (misses fetch on demand).
FLASH_SNAPSHOT=/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
if [ -s "$FLASH_SNAPSHOT/config.json" ]; then
  export MODEL_PATH="$FLASH_SNAPSHOT"
else
  export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.3-Flash'))")
fi
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

# DSA indexer fp8_mqa_logits transient buffer scales with context length; at
# long ctx it OOMs on reserved-but-unallocated fragmentation (GLM-5.2 prodll
# root cause). Flash has a DSA indexer too -> same guard.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Agentic convention 2*CONC capped at 32 (glm5.2prodll pattern). The mamba/KDA
# state cache clamp moves with the ratio: ~22 at r=0.4 (vs 31 at default 0.9).
# CCU 8/16 sit below it; CCU 32 queues -- same posture as the baseline knee.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host 0.0.0.0
  --port "$PORT"
  --tp-size "$TP"
  --ep-size 4
  --moe-runner-backend deep_gemm
  --numa-node 0 0 0 0
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.75
  --mamba-full-memory-ratio 0.4
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length 300000
  --allow-auto-truncate
  --kv-cache-dtype bfloat16
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
