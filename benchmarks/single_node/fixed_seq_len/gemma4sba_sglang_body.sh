#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU SGLang -- SPEED-Bench through aiperf.
# SHARED BODY, not dispatchable on its own.
#
# The SGLang counterpart of gemma4sba_body.sh (vLLM). The client, dataset and
# result shape are identical (run_speedbench_aiperf); only the engine launch
# differs, with every serving knob mapped to its closest SGLang flag:
#   vLLM                                  SGLang
#   --gpu-memory-utilization 0.92         --mem-fraction-static 0.88 (SGLang's
#                                          fraction excludes activations/graphs)
#   --max-model-len 65536                 --context-length 65536
#   --max-num-seqs $CONC                  --max-running-requests $CONC
#   --max-num-batched-tokens 16384        --chunked-prefill-size 16384
#   --no-enable-prefix-caching            --disable-radix-cache
#   mtp, num_speculative_tokens=N         NEXTN + Gemma4 assistant draft
#                                          (auto-promoted to FROZEN_KV_MTP),
#                                          num-steps N, topk 1, draft-tokens N+1
#   FLASHINFER, BF16 KV                   --attention-backend fa4, BF16 KV
#
# Requires the patched image ghcr.io/noridom1/sglang:v0.5.21-gemma4-<sha>
# (sglang#42019 FA4 on SM90 + sglang#39286 aux hidden-state captures); stock
# v0.5.21 falls back to Triton and mis-captures aux states for spec decoding.
#
# Wrapper variables: SB_ARM (base|mtp), SB_CATEGORY, SB_CONFIG, SB_IGNORE_EOS,
# NUM_SPEC_TOKENS (required for mtp). Same contract as gemma4sba_body.sh.
# Optional probe knobs: SB_MEM_FRACTION (0.88), SB_CHUNKED_PREFILL (16384),
# SB_SWA_EVICTION (SGLANG_SWA_EVICTION_INTERVAL, engine default 128).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME

: "${SB_ARM:?set by the wrapper that sources this file}"
: "${SB_CATEGORY:?set by the wrapper that sources this file}"
export SPEEDBENCH_CONFIG="${SB_CONFIG:-throughput_8k}"
export SPEEDBENCH_CATEGORY="$SB_CATEGORY"
export SPEEDBENCH_IGNORE_EOS="${SB_IGNORE_EOS:-1}"

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

SPEC_ARGS=()
DRAFT_MODEL=""
case "$SB_ARM" in
    base)
        NUM_SPEC_TOKENS=0
        ;;
    mtp)
        DRAFT_MODEL="google/gemma-4-31B-it-assistant"
        : "${NUM_SPEC_TOKENS:?the mtp wrapper must pin the draft depth}"
        # Chain draft (topk 1): N steps verify N+1 tokens, the same depth as
        # vLLM's num_speculative_tokens=N.
        SPEC_ARGS=(
            --speculative-algorithm NEXTN
            --speculative-draft-model-path "$DRAFT_MODEL"
            --speculative-num-steps "$NUM_SPEC_TOKENS"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "$((NUM_SPEC_TOKENS + 1))"
        )
        ;;
    *)
        echo "CRITICAL: unknown SB_ARM='$SB_ARM' (expected base|mtp)" >&2
        exit 1
        ;;
esac

if [[ -n "$DRAFT_MODEL" && "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi

resolve_speedbench_dataset || exit 1

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    MAX_MODEL_LEN=65536
fi

start_gpu_monitor

# FROZEN_KV_MTP sizes the SWA pool from the worst case at --max-running-requests
# (draft SWA layers included). At 64 that is ~107 GiB and startup aborts, so MTP
# above 32 runs with 32 slots and a shorter SWA eviction interval. Base is not
# affected and keeps $CONC.
MAX_RUNNING="$CONC"
if [[ "$SB_ARM" == "mtp" && "$CONC" -gt 32 ]]; then
    MAX_RUNNING=32
    export SGLANG_SWA_EVICTION_INTERVAL=32
fi

# KV-pool probe knobs, set by the probe wrappers; defaults are the sweep recipe.
# SB_SWA_EVICTION overrides the MTP cap above only when set.
MEM_FRACTION="${SB_MEM_FRACTION:-0.88}"
CHUNKED_PREFILL="${SB_CHUNKED_PREFILL:-16384}"
if [[ -n "${SB_SWA_EVICTION:-}" ]]; then
    export SGLANG_SWA_EVICTION_INTERVAL="$SB_SWA_EVICTION"
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --mem-fraction-static "$MEM_FRACTION"
    --context-length "$MAX_MODEL_LEN"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size 16384
    --disable-radix-cache
    --attention-backend fa4
    --enable-metrics
    "${SPEC_ARGS[@]}"
    --tool-call-parser gemma4
    --reasoning-parser gemma4
)

set -x
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo "===== SPEED-Bench arm (sglang) ====="
echo "arm=$SB_ARM draft=${DRAFT_MODEL:-none} num_speculative_tokens=$NUM_SPEC_TOKENS"
# Prove the two patches took effect: FA4 selected (not the Triton fallback),
# and the Gemma4 draft promoted to FROZEN_KV_MTP.
grep -iE "attention.backend|fa4|triton|FROZEN_KV_MTP|speculative" "$SERVER_LOG" | head -20 || true
echo "===================================="

SPEEDBENCH_SERVER_PID="$SERVER_PID" \
SPEEDBENCH_META="sb_arm=$SB_ARM num_speculative_tokens=$NUM_SPEC_TOKENS draft_model=${DRAFT_MODEL:-null} engine=sglang" \
    run_speedbench_aiperf
BENCH_RC=$?

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
exit "$BENCH_RC"
