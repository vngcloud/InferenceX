#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM -- SPEED-Bench through aiperf.
# SHARED BODY, not dispatchable on its own.
#
# The aiperf port of the Stage-5 gemma4sb cells (bench/gemma4cp-8k1k-1gpu,
# run 34502057305), which drove `vllm bench serve --dataset-name speed_bench`.
# The engine configuration is copied from that body so the two clients can be
# compared cell for cell; only the load generator changes:
#   - client: run_speedbench_aiperf (benchmark_lib.sh), the one SPEED-Bench
#     client shared by every engine and model, writing the same
#     benchmark_serving-shaped $RESULT_FILENAME.json plus acceptance length
#     from the vllm:spec_decode_* counters;
#   - dataset: the prepared SPEED-Bench snapshot pinned in benchmark_lib.sh
#     (all six splits, every category resolved, mixed included), instead of a
#     committed throughput_8k copy with the cais/hle rows skipped.
#
# launch_h200-greennode.sh builds the recipe path from the model-prefix, so
# each (arm x category) cell is a wrapper that sets:
#   SB_ARM          base | e3 | dflash | dflash2 | mtp
#   SB_CATEGORY     low_entropy | mixed | high_entropy (throughput_*), or a
#                   qualitative category (coding, math, ...)
#   SB_CONFIG       throughput_8k (default) | throughput_{1k,2k,16k,32k} | qualitative
#   SB_IGNORE_EOS   1 (default) | 0
#   NUM_SPEC_TOKENS required for mtp; e3/dflash/dflash2 have defaults
#
# Engine notes carried over from gemma4sb_body.sh: prefix caching is ON (changed
# from the Stage-5 OFF setting; cells may now reuse prefill); KV is left at auto (BF16) because fp8 KV pins
# gemma4 to Triton on SM90; the arm-specific --speculative-config is the only
# engine difference between cells. Unlike Stage 5 the GPU is not pinned to
# index 4; vLLM takes the first visible GPU (TP=1, one GPU per job).

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

# ---- Per-arm speculative configuration (verbatim from gemma4sb_body.sh) -----
SPEC_ARGS=()
DRAFT_MODEL=""
case "$SB_ARM" in
    base)
        NUM_SPEC_TOKENS=0
        ;;
    e3)
        DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
        SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}")
        ;;
    dflash)
        DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.dflash"
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
        SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"dflash\"}")
        ;;
    dflash2)
        # In-house checkpoint, only on h200-greennode_06 at /mnt/models (mounted
        # read-only at /models by the launcher); pin that node when dispatching.
        DRAFT_MODEL="/models/gemma4-31b-it-dflash2"
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
        SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"dflash\"}")
        ;;
    mtp)
        # method must be "mtp": the assistant checkpoint consumes the target's
        # hidden states and is not a standalone LM.
        DRAFT_MODEL="google/gemma-4-31B-it-assistant"
        : "${NUM_SPEC_TOKENS:?the mtp wrapper must pin the draft depth}"
        SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}")
        ;;
    *)
        echo "CRITICAL: unknown SB_ARM='$SB_ARM' (expected base|e3|dflash|dflash2|mtp)" >&2
        exit 1
        ;;
esac

if [[ -n "$DRAFT_MODEL" && "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi

# Installs the pinned aiperf venv and fetches the dataset before the server
# holds the GPU, so a dataset/token problem fails in seconds, not after boot.
resolve_speedbench_dataset || exit 1

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    # 256k serve ceiling, the same as the agentic tau2 recipes: vLLM's reported
    # "GPU KV cache size" is concurrency x max_model_len, so keeping it equal
    # makes the pool lines comparable across SPEED-Bench and tau2 runs.
    MAX_MODEL_LEN=262144
fi

start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --enable-prefix-caching \
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo "===== SPEED-Bench arm ====="
echo "arm=$SB_ARM draft=${DRAFT_MODEL:-none} num_speculative_tokens=$NUM_SPEC_TOKENS"
grep -E "[Ss]peculative|num_speculative_tokens|drafter|[Ee]agle|MTP|[Dd][Ff]lash" "$SERVER_LOG" | head -20 || true
echo "==========================="

SPEEDBENCH_SERVER_PID="$SERVER_PID" \
SPEEDBENCH_META="sb_arm=$SB_ARM num_speculative_tokens=$NUM_SPEC_TOKENS draft_model=${DRAFT_MODEL:-null}" \
    run_speedbench_aiperf
BENCH_RC=$?
set +x

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
exit "$BENCH_RC"
