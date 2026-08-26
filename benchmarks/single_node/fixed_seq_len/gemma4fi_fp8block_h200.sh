#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k -- BF16-KV +
# FlashInfer variant for an exploration run on shared h200-greennode_03.
#
# Differences vs gemma4cp_fp8block_h200.sh (the fp8-KV chunked-prefill variant):
#   - VLLM_ATTENTION_BACKEND=FLASHINFER (explicit)
#   - NO --kv-cache-dtype : KV stays BF16 (auto). For Gemma's sliding-window
#     layers this re-enables vLLM's hybrid KV-cache manager, so the pool is
#     larger than the fp8-KV path -- the point of this A/B.
#   - MAX_MODEL_LEN forced to 65536 (64k) per the requested serve config;
#     immaterial to 8k1k throughput (each request is only ~9k), the KV pool is
#     leftover-memory sized either way, so the fp8-KV vs BF16-KV comparison holds.
#   - CUDA_VISIBLE_DEVICES=4 (GPU 3 is the fp8-KV variant's card; keep them on
#     separate cards so the two runs never contend even if the runner ever
#     grants two slots).
# Same checkpoint / weights (fp8block) / chunked-prefill / batched-tokens 16384 /
# tool+reasoning parsers otherwise.
#
# Exploration-only: dispatched via e2e-tests.yml against a branch, never merged.

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

BENCH_GPU=4

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    # Requested 64k serve ceiling; overrides the scenario-computed ~9472.
    MAX_MODEL_LEN=65536
fi

start_gpu_monitor

set -x
CUDA_VISIBLE_DEVICES="$BENCH_GPU" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

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
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
