#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k -- STAGE 3 of the
# optimization roadmap: vLLM v0.28.0 + native Gemma-4 MTP. SHARED BODY.
#
# This file is NOT dispatchable on its own. runners/launch_h200-greennode.sh
# builds the script name from the matrix's model-prefix, so a sweep over
# speculative depth needs one file per depth. Rather than keep three near
# identical 160-line copies in sync by hand, the three dispatchable wrappers
#   gemma4v28m2_fp8block_h200_mtp.sh   NUM_SPEC_TOKENS=2
#   gemma4v28m4_fp8block_h200_mtp.sh   NUM_SPEC_TOKENS=4
#   gemma4v28m8_fp8block_h200_mtp.sh   NUM_SPEC_TOKENS=8
# each set that one integer and source this body. The depth is therefore
# provably the only thing that differs across the three arms.
#
# ONE-FLAG CLONE of gemma4v28_fp8block_h200.sh (Stage 1): engine config copied
# verbatim, plus the assistant-checkpoint prefetch and one
# --speculative-config. Compare against Stage 1, and against Stage 2's EAGLE3
# arm at its own depth of 3.
#
# THE DRAFTER, AND WHY "method":"mtp" AND NOT "draft_model".
# google/gemma-4-31B-it-assistant is architectures ["Gemma4AssistantForCausalLM"],
# model_type "gemma4_assistant": 4 layers, hidden_size 1024, and crucially
# backbone_hidden_size 5376 -- it consumes the TARGET's hidden states and is
# not a standalone LM. vLLM registers it as Gemma4MTPModel -> gemma4_mtp and
# its own docs (docs/features/speculative_decoding/mtp.md) are explicit:
#   "Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not
#    generic draft models, even though they are passed through the `model`
#    field in --speculative-config. Use "method": "mtp" ... If an older vLLM
#    release logs SpeculativeConfig(method='draft_model', ...) for a Gemma 4
#    assistant checkpoint, that release is treating the assistant as a generic
#    draft model and may fail during initialization for multimodal Gemma 4
#    targets."
# gemma4 IS a multimodal target here, so getting this wrong is an init crash,
# not a slow run. The integrity grep below prints the resolved
# SpeculativeConfig; if it says method='draft_model', abort the comparison.
#
# WHY v0.28.0 IS THE RIGHT FLOOR. vllm/model_executor/models/gemma4_mtp.py and
# the registry entry "Gemma4MTPModel": ("gemma4_mtp", "Gemma4MTP") are both
# present at tag v0.28.0 (cut 2026-08-24). The roadmap's Stage 1 image upgrade
# is what makes this stage possible at all.
#
# ONE KNOWN UPSTREAM BUG, CHECKED AND NOT APPLICABLE. vllm-project/vllm#53884
# ("Make Gemma4 MTP suppress_tokens masking CUDA-graph-safe", merged
# 2026-08-26, i.e. AFTER the v0.28.0 tag) makes engine init fail
# deterministically with "Cannot copy between CPU and CUDA tensors during CUDA
# graph capture" -- but only for an assistant whose generation_config carries a
# non-empty suppress_tokens. The PR names google/gemma-4-12B-it-assistant as
# the only such checkpoint and states the others are unaffected;
# google/gemma-4-31B-it-assistant's generation_config.json has no
# suppress_tokens key. If init nonetheless dies on that assertion, the fix is a
# newer image, not a recipe change.
#
# DEPTH SWEEP RATIONALE. Deeper speculation multiplies the reward when the
# drafter is right and the waste when it is wrong, and the crossover moves with
# batch size -- at conc 64 the verify step is already compute-bound, so depth 8
# may lose where depth 2 wins. That is exactly why every depth runs the full
# conc 1/8/32/64 ladder instead of a single point.
#
# BASELINE (v0.25.0, run 32946716838) total tok/s/GPU | median TTFT | mean TPOT:
#   conc  1:   728.6 | 0.551 s | 11.75 ms
#   conc  8: 3,729.4 | 0.572 s | 17.92 ms
#   conc 32: 6,301.8 | 0.657 s | 42.88 ms
#   conc 64: 6,907.3 | 1.304 s | 79.18 ms
#
# ROADMAP GATE. Adopted only if production-weighted output throughput improves
# by >=15% with p99 TTFT and p99 ITL no more than 5% worse.
#
# BENCH_GPU 6, the dead VLLM_ATTENTION_BACKEND=FLASHINFER, and the absence of
# --kv-cache-dtype are all inherited from Stage 1 on purpose; fp8 KV pins
# gemma4 to Triton on SM90 and costs +72% TTFT / +25% TPOT / -20% req/s.
# Power telemetry is not comparable on this branch; conc 1's "p99" is
# max-of-10. Pin the host with --runner-node-filter h200-greennode_03.
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

# The assistant/MTP head ships as a separate repo; prefetch it so the
# --speculative-config "model" id resolves offline like the target does.
DRAFT_MODEL="google/gemma-4-31B-it-assistant"
: "${NUM_SPEC_TOKENS:?set by the per-depth wrapper that sources this file}"
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
    --speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Stage-3 integrity check.
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
