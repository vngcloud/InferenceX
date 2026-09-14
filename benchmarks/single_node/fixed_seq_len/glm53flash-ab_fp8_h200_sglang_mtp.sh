#!/usr/bin/env bash

# GLM-5.3-Flash (zai-org/GLM-5.3-Flash) FP8 fixed-seq-len 8k1k recipe on
# 4xH200 (TP4/EP4), Arm AB (combined free wins). Copy of
# benchmarks/single_node/fixed_seq_len/glm5.3flash_fp8_h200_sglang_mtp.sh
# with exactly 4 diffs (baseline arm untouched, result series separable):
#   1. DROP --dsa-prefill-backend/--dsa-decode-backend tilelang pins -> auto
#      (flashmla_sparse prefill + fa3 decode on Hopper; upstream PR #36895
#      measured +9.0% c32 / +15.6% c128 on 8xH200 TP8/EP8 Flash).
#   2. --mem-fraction-static 0.75 -> 0.85 (22.5GB/rank idle in baseline boot;
#      grows both KDA-state and KV pools before touching their split).
#   3. --hicache-size 32 -> 64 (host RAM ~1.4TB, 4x64=256GB fits).
#   4. + --hicache-write-policy write_back (baseline write_through suspected
#      in the c8 TTFT-p99 16.3s outlier; prod 5.2 runs write_back).
# Everything else = baseline (deep_gemm EP4, MTP adaptive 5/1/6, bf16 KV,
# glm45/glm47, numa 0x4, no --max-running-requests so c32/c48 still probe the
# mamba-clamped knee).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

# Model pre-warmed in /mnt/hf_hub_cache on h200-greennode_06 (snapshot
# eb9eb208..., 62 safetensors shards, 313GB); snapshot_download hits the
# cache, misses fetch on demand. Tokenizer ships its chat template as a
# standalone chat_template.jinja, so --use-chat-template resolves fine.
export MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-5.3-Flash'))")

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP" \
    --ep-size 4 \
    --moe-runner-backend deep_gemm \
    --numa-node 0 0 0 0 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --speculative-adaptive \
    --enable-hierarchical-cache \
    --hicache-size 64 \
    --hicache-write-policy write_back \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 > "$SERVER_LOG" 2>&1 &

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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
