#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 W4AFP8 fixed-seq-len 8k1k baseline arm: the serving config of the
# glm5.2prodll low-latency agentic baseline (run 34197652789, branch
# bench/glm52-lowlatency -- TP8-only, no dp-attention, no router, no EP,
# EAGLE 5/1/6 on the baked-in MTP head, HiCache 128GB/rank direct write_back,
# flashmla_sparse_q8 DSA prefill, dfs-weight, chunked-prefill 32768,
# context 300000, kv fp8_e4m3, mem-fraction 0.75, min(2*CONC,32) running
# requests, glm47/glm45 parsers) driven by the fixed ISL/OSL 8k1k sweep
# instead of the cc-traces replay. Image is lmsysorg/sglang:v0.5.19, not the
# baseline run's v0.5.18: the DSpark twin arm (glm5.2dsparkll8k1k) needs the
# AlayaNeW draft, boot-verified on v0.5.19 only (2026-09-12 handoff §6), and
# both arms of the A/B pair share one image so the pair stays single-variable
# (v0.5.19-vs-v0.5.18 was measured at +1..4% tput on deep519, run 34508949422).
# --trust-remote-code is passed here too (inert for the PhalaCloud target,
# boot-verified 2026-09-12) so the DSpark twin differs by SPEC_ARGS only.
# Boot-verified 2026-09-12 on 8xh200-1 (h200-greennode_07), TP8, stock image:
# KV pool 800,896 tokens/rank, ~26 GB VRAM free at steady state.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

nvidia-smi

# Auto-download / resolve W4AFP8 target from HF cache (pre-warmed on the
# runner /mnt/hf_hub_cache, misses fetch on demand) -- same as the prodll arm.
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('PhalaCloud/GLM-5.2-W4AFP8'))")

SERVER_LOG=/workspace/server.log

start_gpu_monitor

SPEC_ARGS=()
if [ "${SPEC_DECODING:-mtp}" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
  )
fi

# Prod env parity (SGLANG_DP_USE_GATHERV and SGLANG_ENABLE_METRICS_DP_ATTENTION
# are no-ops without dp-attention; kept so the arm matches prod env 1:1).
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# DSA indexer's fp8_mqa_logits needs a large transient buffer that scales with
# context length; at long ctx it OOM'd on reserved-but-unallocated fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Low-latency running-batch ceiling of 32 (mirrors the prodll baseline: for
# CCU 1/4/8/16 the 2*CONC term already sits below 32 and only CCU 32 hits it).
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 32 ] && MAX_RUNNING_REQUESTS=32

set -x
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --quantization w4afp8 \
    --tp-size "$TP" \
    --chunked-prefill-size 32768 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.75 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --context-length 300000 \
    --kv-cache-dtype fp8_e4m3 \
    --dsa-prefill-backend flashmla_sparse_q8 \
    --allow-auto-truncate \
    --enable-metrics \
    --enable-metrics-for-all-schedulers \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-size 128 \
    --hicache-io-backend direct \
    --hicache-write-policy write_back \
    "${SPEC_ARGS[@]}" \
    --schedule-policy dfs-weight > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template \
    --server-pid "$SERVER_PID"

stop_gpu_monitor
set +x
