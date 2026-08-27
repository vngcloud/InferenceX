#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k -- STAGE 1 of the
# optimization roadmap: vLLM v0.28.0, no speculative decoding.
#
# THIS IS A ONE-VARIABLE CLONE of gemma4fi_fp8block_h200.sh, the locked baseline
# that produced run 32946716838 (2026-08-26, h200-greennode_03, branch
# bench/gemma4cp-8k1k-1gpu @ b5a6d354). The engine config below is copied
# verbatim; the ONLY intended difference lives in configs/nvidia-master.yaml,
# where this recipe's key pins image vllm/vllm-openai:v0.28.0 instead of
# v0.25.0. Purpose: establish whether the image upgrade alone is neutral or
# beneficial, before any drafter is introduced in Stage 2/3.
#
# Baseline metrics to beat (total tok/s/GPU | median TTFT | mean TPOT):
#   conc  1:   728.6 | 0.551 s | 11.75 ms
#   conc  8: 3,729.4 | 0.572 s | 17.92 ms
#   conc 32: 6,301.8 | 0.657 s | 42.88 ms
#   conc 64: 6,907.3 | 1.304 s | 79.18 ms
#
# DELIBERATE DEVIATION, THE ONLY ONE: BENCH_GPU 4 -> 6. GPU index is not a
# performance variable on a homogeneous 8xH200 SXM box at TP1, but contention
# is: GPUs 6,7 are the pair granted to us on h200-greennode_03 as of
# 2026-08-27, and card 4's tenancy has since changed hands. Trading a nominal
# variable to remove a real confounder.
#
# KEPT ON PURPOSE, THOUGH IT IS DEAD CODE: VLLM_ATTENTION_BACKEND=FLASHINFER.
# It did NOT take effect on the baseline -- that run's startup log reports
# FlashAttention 4, and FlashInfer is excluded for gemma4 by
# FlashInferBackend.supports_mm_prefix() -> False (multimodal PrefixLM
# bidirectional attention); the candidate set logged is
# ['FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'], with no FlashInfer in it.
# So the baseline's true identity is BF16 KV + FA4. The env var is retained
# because a control arm must be byte-identical, but v0.28.0 could honour it
# differently -- hence the backend-verdict grep after server readiness below.
# If that grep does not say FLASH_ATTN, this run is NOT an image-only A/B and
# the numbers must not be compared to the baseline table above.
#
# NO --kv-cache-dtype, and it must stay that way for every stage on this
# hardware. fp8 KV pins gemma4 to the Triton kernel on SM90 and nothing lifts
# it: FA3 takes fp8 but not head_size 512, FA4 takes head_size 512 but wants
# SM100 for fp8, FlashInfer is excluded independently. Measured cost at 8k1k
# conc 8 (32946716838 bf16 vs 32946714039 fp8, same box/day/image): TTFT mean
# 798 -> 1376 ms, TPOT 17.9 -> 22.5 ms, 0.45 -> 0.36 req/s.
#
# CAVEATS WHEN READING THE RESULTS.
#   - Power/util telemetry is NOT comparable. start_gpu_monitor here takes no
#     --gpu-ids, and nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the CSV covers
#     all 8 cards including other tenants' load. Left unfixed on purpose: the
#     fix exists on vng-benchmark-gemma4-sgl but importing it would add a
#     second variable. Treat power as out of scope for this comparison.
#   - conc 1 runs only 10 prompts (--num-prompts is CONC*10), so its "p99" is
#     max-of-10, not a percentile. conc 64 gets 640 prompts and is sound.
#
# RUNNER: pin the host at dispatch with --runner-node-filter h200-greennode_03.
# The config key uses the three-box cluster:h200-greennode pool (as the
# baseline did), so without the filter this lands on _01 or _04 and host
# variance contaminates the delta.
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

BENCH_GPU=6

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

# Stage-1 integrity check. The baseline resolved to FLASH_ATTN (FA4) despite
# VLLM_ATTENTION_BACKEND=FLASHINFER. If v0.28.0 resolves differently, the
# image is no longer the only variable and the comparison is void -- so print
# the verdict into the job log rather than inferring it later.
echo "===== attention backend selection ====="
grep -E "attention backend|FlashAttention version|kv_cache_dtype|Using .* backend" "$SERVER_LOG" || true
echo "======================================="

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
