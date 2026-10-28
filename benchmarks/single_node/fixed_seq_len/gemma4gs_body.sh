#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM -- GreenSpec through aiperf.
# SHARED BODY, not dispatchable on its own.
#
# Same engine configuration as gemma4sba_body.sh (prefix caching off, KV auto,
# the arm's --speculative-config the only engine difference); the workload is a
# GreenSpec subset instead of SPEED-Bench:
#   - client: run_greenspec_aiperf (benchmark_lib.sh), which picks the aiperf
#     greenspec_* preset for the selection, writes the benchmark_serving-shaped
#     $RESULT_FILENAME.json with acceptance length from vllm:spec_decode_*, and
#     runs the guard (allowlisted aggregates only, server log withheld) before
#     anything is uploaded;
#   - dataset: the private GreenSpec release pinned in benchmark_lib.sh, read
#     with GREENSPEC_HF_TOKEN.
# ISL/OSL in the matrix are labels only: every row brings its own prompt, and
# the output cap is GREENSPEC_MAX_TOKENS.
#
# Each (arm x selection) cell is a wrapper that sets:
#   GS_ARM          base | mtp
#   GS_SUBSET       diverse | agentic_coding | agentic_tools | llm_task
#   GS_LITE         1 | 0 (default 1)
#   GS_NUM_PROMPTS  requests per conc (default: the whole selection)
#   NUM_SPEC_TOKENS required for mtp

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    RESULT_FILENAME

: "${GS_ARM:?set by the wrapper that sources this file}"
: "${GS_SUBSET:?set by the wrapper that sources this file}"
export GREENSPEC_SUBSET="$GS_SUBSET"
export GREENSPEC_LITE="${GS_LITE:-1}"
export GREENSPEC_WARMUP_REQUESTS="${GREENSPEC_WARMUP_REQUESTS:-4}"
if [[ -n "${GS_NUM_PROMPTS:-}" ]]; then export GREENSPEC_NUM_PROMPTS="$GS_NUM_PROMPTS"; fi

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
case "$GS_ARM" in
    base)
        NUM_SPEC_TOKENS=0
        ;;
    mtp)
        DRAFT_MODEL="google/gemma-4-31B-it-assistant"
        : "${NUM_SPEC_TOKENS:?the mtp wrapper must pin the draft depth}"
        SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}")
        ;;
    *)
        echo "CRITICAL: unknown GS_ARM='$GS_ARM' (expected base|mtp)" >&2
        exit 1
        ;;
esac

if [[ -n "$DRAFT_MODEL" ]]; then hf download "$DRAFT_MODEL"; fi

# Installs the pinned aiperf venv and fetches the dataset before the server
# holds the GPU, so a dataset/token problem fails in seconds, not after boot.
resolve_greenspec_dataset || exit 1

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

# GreenSpec rows reach the 64k+ buckets; serve the full 256k context.
MAX_MODEL_LEN=262144

start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --no-enable-prefix-caching \
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo "===== GreenSpec arm ====="
echo "arm=$GS_ARM draft=${DRAFT_MODEL:-none} num_speculative_tokens=$NUM_SPEC_TOKENS"
echo "========================="

GREENSPEC_SERVER_PID="$SERVER_PID" \
GREENSPEC_SERVER_LOG="$SERVER_LOG" \
GREENSPEC_META="gs_arm=\"$GS_ARM\" num_speculative_tokens=$NUM_SPEC_TOKENS" \
    run_greenspec_aiperf
BENCH_RC=$?
set +x

stop_gpu_monitor
exit "$BENCH_RC"
