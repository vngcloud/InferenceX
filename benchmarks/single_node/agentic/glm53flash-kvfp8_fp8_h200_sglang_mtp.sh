#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3-Flash FP8 agentic-coding trace-replay recipe on 4xH200, kv-fp8 arm.
#
# Identical to glm5.3flash_fp8_h200_sglang_mtp.sh (STANDARD 4xH200 TP4/EP4 +
# deep_gemm, tilelang DSA prefill+decode, adaptive EAGLE 5/1/6, HiCache 32) with
# exactly two server-flag diffs plus a runtime source patch:
#   1. --kv-cache-dtype fp8_e4m3  (was bfloat16): 528B vs 1024B per KV element
#      -> ~+75% KV pool (1,048,384 -> 1,836,288 tok/rank, boot-measured), to
#      relieve the agentic prefix-cache thrash the bf16 twin hit at c16+
#      (run 34683194686: cache hit 92->20%, TTFT p50 39s).
#   2. --schedule-policy dfs-weight (was default fcfs): cache-aware DFS weighting
#      on the prefix tree, prioritizes shared-prefix agentic multi-turn requests.
#
# fp8 KV on GLM-5.3-Flash DSA does NOT work on the stock image: config has
# index_kpool=4, and on SM90 the intersection of {backends that accept fp8 KV}
# and {backends that accept index_kpool>1} is empty (issue #36830). The open
# draft PR #39349 opens the tilelang fp8-KV gate (read the scaled 528B layout,
# dequant to BF16 for the TileLang consumer). It is pure Python (TileLang JITs at
# runtime -> no image build, no CUDA compile), so we overlay the three patched
# runtime files onto the image's sglang before launch. The patched copies under
# patches/pr39349/ are hand-ported to THIS image (lmsysorg/sglang:glm-5.3-flash,
# older than main: _check_tilelang_dsa_fp8_kv vs main's
# _check_dsa_backend_constraints) and boot-verified on h200-greennode_06
# (docs/handoffs/2026-09-14-glm53flash-kvfp8-dfsweight-mamba-bump.md). Both DSA
# phases MUST stay pinned to tilelang: the gate only opens for
# prefill==decode==tilelang + latent512/RoPE0 + dcp_size==1; auto-resolve picks
# flashmla_kv and dies at CUDA-graph capture as before.
#
# All other flags, env, and the max-running-requests clamp match the bf16 arm.
# The mamba/KDA state cache still binds max_running_requests at 31 -- fp8 KV
# buys capacity, not the concurrency ceiling.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

# Overlay the PR #39349 fp8-KV runtime patch (3 Python files) onto the image's
# installed sglang. Resolve the live package dir so this is correct whether the
# image ships an editable install or site-packages.
PATCH_DIR="$(cd "$(dirname "$0")/patches/pr39349" && pwd)"
SGLANG_DIR="$(python3 -c 'import os, sglang; print(os.path.dirname(sglang.__file__))')"
cp "$PATCH_DIR/dequant_k_cache.py" "$SGLANG_DIR/kernels/ops/attention/dsa/dequant_k_cache.py"
cp "$PATCH_DIR/overrides.py"       "$SGLANG_DIR/srt/arg_groups/overrides.py"
cp "$PATCH_DIR/dsa_backend.py"     "$SGLANG_DIR/srt/layers/attention/dsa_backend.py"
python3 -c 'import sglang.srt.arg_groups.overrides, sglang.srt.layers.attention.dsa_backend' \
  && echo "PR#39349 fp8-KV patch applied to $SGLANG_DIR"

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

FLASH_SNAPSHOT=/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
if [ -s "$FLASH_SNAPSHOT/config.json" ]; then
  export MODEL_PATH="$FLASH_SNAPSHOT"
else
  export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.3-Flash'))")
fi
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

# DSA indexer fp8_mqa_logits transient buffer scales with context length; guard
# against reserved-but-unallocated fragmentation OOM at long ctx (prodll class).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Agentic convention 2*CONC capped at 32; mamba/KDA state cache clamps to 31.
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
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
  --schedule-policy dfs-weight
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
