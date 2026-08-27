#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU sglang, fixed-seq-len 8k1k -- sglang twin
# of gemma4fi_fp8block_h200.sh (vLLM). Same model / TP1 / 8k1k / conc [1,8,32,64]
# / fp8block weights / chunked-prefill / tool+reasoning parsers.
#
# Differences vs the vLLM variant:
#   - sglang serve instead of vllm serve
#   - --mem-fraction-static 0.85 (vs vLLM --gpu-memory-utilization 0.92)
#   - --chunked-prefill-size 16384 (vs --max-num-batched-tokens + --enable-chunked-prefill)
#   - --context-length 65536 (vs --max-model-len 65536)
#   - CUDA_VISIBLE_DEVICES=4 (same GPU as gemma4fi, free on h200-greennode_03)
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

export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    MAX_MODEL_LEN=65536
fi

start_gpu_monitor

set -x
CUDA_VISIBLE_DEVICES="$BENCH_GPU" sglang serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tp "$TP" \
    --mem-fraction-static 0.85 \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$CONC" \
    --chunked-prefill-size 16384 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 > "$SERVER_LOG" 2>&1 &

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
