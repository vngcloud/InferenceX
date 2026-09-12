#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3-Flash (zai-org/GLM-5.3-Flash) FP8 agentic-coding trace-replay recipe
# on 4xH200, the STANDARD 4xH200 layout (TP4/EP4 + deep_gemm MoE runner) from
# the boot-verified 8k1k arm (same flags, see
# benchmarks/single_node/fixed_seq_len/glm5.3flash_fp8_h200_sglang_mtp.sh and
# docs/handoffs/2026-09-10-glm53flash-4xh200-boot-config.md):
#   - EP4 keeps experts full-size (D=2048) so the DSV4 kernel's D//8 >= E_rank
#     assert passes with 73 experts/rank; EP1 would TP-shard to D=512 and die.
#   - native MTP draft (in-checkpoint nextn layer), adaptive EAGLE 5/1/6.
#   - HiCache L2 32GB/rank in the build-resolved default form (only
#     --hicache-size; this image resolves io=kernel, write_through, page_first).
#   - tilelang DSA prefill+decode, bf16 KV, glm45/glm47 parsers, mem 0.75,
#     numa-node 0x4 (all four GPUs on node 0).
# Agentic-only deltas vs the 8k1k arm: trace replay (cap-256k SemiAnalysis CC
# traces) instead of fixed 8k1k, --context-length 300000 + --allow-auto-truncate
# for the 256k-capped traces, --chunked-prefill-size 32768 (GLM-5.2 agentic
# arm convention for long prefills), server metrics + cache report, and
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (DSA indexer long-ctx
# transient-buffer fragmentation guard, same OOM class as GLM-5.2 prodll).
#
# --max-running-requests follows the repo agentic convention 2*CONC (capped at
# 32 like glm5.2prodll): the mamba/KDA state cache still binds at 31 (156
# slots / 5 per request for MTP rollback) -- server_args clamps any value down
# to 31 (the 8k1k arm's unset default 48 was clamped the same way). CCU 32 is
# the knee point: in-flight above 31 queues at the scheduler (8k1k measured:
# c32 knee, c48 +2% tput with TTFT p50 0.98->12s). The ~1M-token/rank KV pool
# handles concurrent agentic ctx via HiCache offload; --allow-auto-truncate
# guards the tail.
#
# Model pre-warmed in /mnt/hf_hub_cache on h200-greennode_06 (snapshot
# eb9eb208..., 62 safetensors shards, 313GB); snapshot_download hits the
# cache, misses fetch on demand. Tokenizer ships chat_template.jinja
# standalone, so template resolution is fine.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

CACHE_ARGS=(
  --enable-hierarchical-cache
  --hicache-size 32
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
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
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
