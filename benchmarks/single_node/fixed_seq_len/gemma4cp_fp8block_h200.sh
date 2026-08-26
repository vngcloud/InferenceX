#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k -- chunked-prefill
# variant of gemma4_fp8block_h200.sh for an exploration run on the shared
# h200-greennode_03 box (GPUs 6,7 held by another tenant at dispatch time).
#
# Differences vs the gemma4_fp8block_h200.sh baseline:
#   - CUDA_VISIBLE_DEVICES=3 : pin to one free card so --gpu-memory-utilization
#     0.92 never touches the tenant's GPUs (the launcher runs --gpus all, and
#     the baseline recipe has no pin, so it would otherwise land on GPU 0).
#   - --enable-chunked-prefill / --long-prefill-token-threshold 8192
#   - --max-num-batched-tokens 16384 (baseline: 8192)
# Same checkpoint / precision / fp8_e4m3 KV / tool+reasoning parsers otherwise.
# MAX_MODEL_LEN stays scenario-computed (isl+osl+slack, ~9216) -- fixed-seq-len
# does not need the model's full context.
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

# The one free card picked from `nvidia-smi` on h200-greennode_03 (0-5 free,
# 6-7 tenant). nvidia-smi ignores CUDA_VISIBLE_DEVICES so the gpu monitor still
# logs all 8 cards -- that is cosmetic telemetry only; throughput is TP-derived.
BENCH_GPU=3

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

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# max_num_batched_tokens must not exceed max_num_seqs * max_model_len, else the
# warmup dummy batch overruns a single sequence's max length and CUDA graph
# capture dies with an illegal memory access (hit at conc 1: 16384 > 1*9472).
# Cap it to CONC*MAX_MODEL_LEN so low-conc points stay valid; higher conc keeps
# the full 16384. (conc 1 -> 9472, conc>=2 -> 16384 for max_model_len 9472.)
MAX_BATCHED_TOKENS=16384
CAP=$(( CONC * MAX_MODEL_LEN ))
if [ "$MAX_BATCHED_TOKENS" -gt "$CAP" ]; then MAX_BATCHED_TOKENS="$CAP"; fi

set -x
CUDA_VISIBLE_DEVICES="$BENCH_GPU" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
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
