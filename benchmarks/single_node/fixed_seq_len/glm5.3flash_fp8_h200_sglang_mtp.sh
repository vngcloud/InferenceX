#!/usr/bin/env bash

# GLM-5.3-Flash (zai-org/GLM-5.3-Flash) FP8 fixed-seq-len 8k1k recipe on
# 4xH200 (TP4/EP1), low-latency config. First InferenceX arm for the
# glm5_next arch (Glm5NextForConditionalGeneration): needs the dedicated
# image lmsysorg/sglang:glm-5.3-flash (post-PR-#36507 build, not in any
# sglang release).
#
# Server flags are the boot-verified handoff config
# (docs/handoffs/2026-09-10-glm53flash-4xh200-boot-config.md §2026-09-11):
# the STANDARD 4xH200 layout is TP4/EP4 + deep_gemm MoE runner (empirically
# boot-verified: EP4 keeps experts full-size so D=2048 and the DSV4 kernel's
# D//8 >= E_rank assert passes with 73 experts/rank). Do NOT drop to EP1 with
# deep_gemm on <=4 GPUs: 289 fused experts/rank at EP1 are TP-sharded to
# D=512 and exceed the D//8=64 ceiling (assert crash at warmup); EP1 would
# also fall back to the default triton runner with an untuned E=289 config.
#   - native MTP draft (num_nextn_predict_layers=1 in the checkpoint, no
#     external draft model), adaptive EAGLE 5/1/6.
#   - HiCache L2 host offload 32GB/rank (resolved defaults in this build:
#     kernel IO, write_through, page_first).
#   - tilelang DSA prefill+decode backends, bf16 KV, glm45/glm47 parsers,
#     mem-fraction-static 0.75, numa-node 0x4 (all four GPUs on node 0).
#
# --max-running-requests is intentionally NOT set: the mamba/KDA state cache
# caps running requests at 31 (156 slots / 5 per request for MTP rollback)
# regardless, so the conc 32/48 ladder points measure scheduler-queueing
# above that cap (the knee), not admitted concurrency. The ~1M-token/rank KV
# pool is not the binding constraint at 8k1k (31 reqs x ~9k tokens ~= 27%).

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
    --mem-fraction-static 0.75 \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --speculative-adaptive \
    --enable-hierarchical-cache \
    --hicache-size 32 \
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
