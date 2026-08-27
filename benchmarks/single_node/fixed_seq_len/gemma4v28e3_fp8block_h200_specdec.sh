#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k -- STAGE 2 of the
# optimization roadmap: vLLM v0.28.0 + EAGLE3 speculative decoding.
#
# ONE-FLAG CLONE of gemma4v28_fp8block_h200.sh (Stage 1). The engine config is
# copied verbatim; the only additions are the draft-checkpoint prefetch and a
# single --speculative-config. Stage 1 already isolated the v0.25.0 -> v0.28.0
# image move, so the delta measured here is attributable to EAGLE3 alone --
# provided Stage 1 came back clean. If Stage 1 showed a backend or throughput
# shift, re-anchor against Stage 1's numbers, NOT the v0.25.0 baseline table.
#
# THE DRAFTER. RedHatAI/gemma-4-31B-it-speculator.eagle3, ~2B params in bf16,
# num_speculative_tokens=3 -- the value the model card itself demonstrates. The
# exact --speculative-config JSON below is byte-for-byte the invocation that
# ran green in Actions run 33041047279 (gemma4mc30, mooncake trace, tp1 x dp2,
# vLLM v0.25.0), so the drafter is known-loadable against this checkpoint
# family rather than assumed to be.
#
# WHAT THIS COSTS BEFORE IT PAYS. The drafter is resident on the same card and
# is charged against --gpu-memory-utilization 0.92, so the KV pool shrinks by
# roughly the drafter's bf16 footprint versus Stage 1. That is a real cost of
# speculative decoding, not a confound -- but it means a throughput regression
# at conc 64 could be KV pressure rather than a bad acceptance rate. Read the
# reported KV-cache size and any preemption counter alongside the latency
# table before concluding anything about acceptance.
#
# ACCEPTANCE IS THE NUMBER THAT MATTERS. Speculative decoding trades extra
# compute per step for fewer steps; it wins only when the acceptance rate is
# high enough. The post-readiness grep below dumps vLLM's speculative-decoding
# lines into the job log so the accepted-token statistics can be read directly
# instead of inferred from the throughput delta.
#
# ROADMAP GATE. This variant is adopted only if it lifts production-weighted
# output throughput by >=15% without worsening p99 TTFT or p99 ITL by >5%.
#
# BASELINE (v0.25.0, run 32946716838) total tok/s/GPU | median TTFT | mean TPOT:
#   conc  1:   728.6 | 0.551 s | 11.75 ms
#   conc  8: 3,729.4 | 0.572 s | 17.92 ms
#   conc 32: 6,301.8 | 0.657 s | 42.88 ms
#   conc 64: 6,907.3 | 1.304 s | 79.18 ms
#
# NOT A CLONE OF gemma4_fp8block_h200_specdec.sh, despite sharing the _specdec
# suffix. That file is a 2-replica + vllm-router recipe with --kv-cache-dtype
# fp8_e4m3, and its eagle3 flags are commented out (SPEC_DECODE_FLAGS=()) from
# an old diagnostic. It is unusable as a TP1 single-GPU control arm. The suffix
# is forced by runners/launch_h200-greennode.sh, which maps spec-decoding
# "draft_model" -> _specdec; this branch's matrix schema does not accept
# "eagle3" as a spec-decoding value (utils/matrix_logic/validation.py allows
# only mtp/draft_model/none), and eagle3 is an external-draft method, so
# draft_model is the correct and only available category.
#
# BENCH_GPU 6, VLLM_ATTENTION_BACKEND=FLASHINFER retained as dead code, and NO
# --kv-cache-dtype: all three inherited from Stage 1 deliberately. fp8 KV pins
# gemma4 to the Triton kernel on SM90 (FA3 rejects head_size 512, FA4 needs
# SM100 for fp8, FlashInfer is excluded by supports_mm_prefix() -> False) and
# cost +72% TTFT / +25% TPOT / -20% req/s at 8k1k conc 8. Hardware constraint,
# valid for every stage on this box.
#
# CAVEATS. Power telemetry is not comparable (start_gpu_monitor takes no
# --gpu-ids on this branch and nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the
# CSV covers all 8 cards). conc 1 runs only 10 prompts, so its "p99" is
# max-of-10.
#
# RUNNER: pin with --runner-node-filter h200-greennode_03 at dispatch.
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

# EAGLE3 draft head lives in a separate repo; pull it into the HF cache so the
# --speculative-config "model" id resolves the same way the target does.
DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
NUM_SPEC_TOKENS=3
if [[ "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi

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
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Stage-2 integrity check.
# VLLM_ATTENTION_BACKEND=FLASHINFER. If v0.28.0 resolves differently, the
# image is no longer the only variable and the comparison is void -- so print
# the verdict into the job log rather than inferring it later.
echo "===== attention backend selection ====="
grep -E "attention backend|FlashAttention version|kv_cache_dtype|Using .* backend" "$SERVER_LOG" || true
echo "===== speculative decoding ====="
grep -E "[Ss]peculative|num_speculative_tokens|drafter|[Ee]agle|MTP" "$SERVER_LOG" | head -40 || true
echo "==============================="
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
