#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3-Flash (zai-org/GLM-5.3-Flash) FP8 agentic-coding trace-replay recipe
# on 4xH200, Arm AB (combined free wins). Copy of
# benchmarks/single_node/agentic/glm5.3flash_fp8_h200_sglang_mtp.sh with
# exactly 4 diffs (baseline arm untouched, result series separable):
#   1. DROP --dsa-prefill-backend/--dsa-decode-backend tilelang pins -> auto
#      (flashmla_sparse prefill + fa3 decode on Hopper; upstream PR #36895
#      measured +9.0% c32 / +15.6% c128 on 8xH200 TP8/EP8 Flash).
#   2. --mem-fraction-static stays 0.75 (0.85 OOM, run 34800928433; 0.80 read
#      -2%/-11% from memory pressure, run 34803244404).
#   3. --hicache-size 32 -> 64 (host RAM ~1.4TB, 4x64=256GB fits; baseline
#      host pool hit 100% full at agentic c16+).
#   4. + --hicache-write-policy write_back (baseline write_through).
# Everything else = baseline agentic twin (ctx 300000 + auto-truncate,
# chunked-prefill 32768, metrics + cache-report, expandable_segments,
# MAX_RUNNING_REQUESTS min(2*CONC,32), MTP adaptive 5/1/6, bf16 KV,
# glm45/glm47, numa 0x4, snapshot pin eb9eb208 + snapshot_download fallback).

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
# state cache clamps the effective cap to 31 regardless: 2/16 for CCU 1/8 sit
# below it, CCU 16/32 hit 32 then clamp to 31 -- matching the 8k1k knee.
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
