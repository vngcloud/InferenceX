#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 W4AFP8 fixed-seq-len 8k1k DSpark arm: byte-identical server config
# to the glm5.2ll8k1k EAGLE baseline twin (benchmarks/single_node/fixed_seq_len/
# glm5.2ll8k1k_fp4_h200_sglang.sh -- itself the serving config of the glm5.2prodll
# agentic baseline run 34197652789) except the SPEC_ARGS block: EAGLE 5/1/6 on
# the baked-in MTP head is swapped for DSPARK on the external
# AlayaNeW/GLM-5.2-DSpark bf16 draft. The A/B pair is single-variable
# (speculative algorithm only): same image lmsysorg/sglang:v0.5.19, same TP8-only
# topology, same quant, same cache, same ladder, same trace-free 8k1k driver.
# Draft flags, boot-verified 2026-09-12 on 8xh200-1 (docs/handoffs/
# 2026-09-12-glm52-dspark-alayanew-boot-test.md §6):
#   - --speculative-draft-model-quantization unquant keeps the draft in bf16;
#     if it inherits w4afp8 the MoE-quant applies to the dense draft and breaks.
#   - --trust-remote-code is REQUIRED for this draft: the HF repo ships custom
#     code (modeling_qwen3_dspark.py etc.) and transformers' resolve_trust_
#     remote_code aborts without it (inert for the PhalaCloud W4AFP8 target).
#   - --speculative-dspark-block-size 8 = the checkpoint's own config (gamma 8,
#     verify window 9); auto-infer would take the same value.
# DSpark KV-pool footprint per rank at TP8 (boot-measured): draft weights
# 1.37 GB + draft device KV 7.64 GB (9.5 KB/token, TP-sharded) + draft verify
# graphs 1.02 GB on top of the EAGLE baseline -- fits mem-fraction 0.75 with
# ~26 GB VRAM free at steady state, KV pool 800,896 tokens/rank.

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

# Auto-download / resolve from HF cache (pre-warmed on the runner
# /mnt/hf_hub_cache, misses fetch on demand) -- same as the baseline twin.
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('PhalaCloud/GLM-5.2-W4AFP8'))")
DRAFT_MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('AlayaNeW/GLM-5.2-DSpark'))")

SERVER_LOG=/workspace/server.log

start_gpu_monitor

SPEC_ARGS=(
  --speculative-algorithm DSPARK
  --speculative-draft-model-path "$DRAFT_MODEL_PATH"
  --speculative-draft-model-quantization unquant
  --speculative-dspark-block-size 8
)

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
