#!/usr/bin/env bash

# Gemma-4 31B FP8-block (RedHatAI/gemma-4-31B-it-FP8-block) single-GPU vLLM
# recipe, fixed-seq-len — router + eagle3 speculative decoding variant of
# gemma4_fp8block_h200.sh.
#
# Adds, on top of the baseline single-vllm-serve recipe:
#   - eagle3 speculative decoding (RedHatAI/gemma-4-31B-it-speculator.eagle3,
#     num_speculative_tokens=3)
#   - chunked-prefill anti-head-of-line-blocking tuning
#     (--long-prefill-token-threshold caps a single long prefill's tokens per
#     step so it can't starve other requests' decode steps;
#     --max-num-batched-tokens is raised to fit a full chunk plus decode
#     tokens from in-flight requests in the same step)
#   - a vllm-router (cache_aware policy) sitting in front of the single vLLM
#     backend, so the benchmark client always talks to the router's port
#
# CPU RAM as an L2 KV-cache tier via vLLM's native SimpleCPUOffloadConnector
# was tried and REMOVED: at conc>=128 (GPU KV cache saturating to ~99-100%)
# the engine hit a fatal `assert block.ref_cnt == 0` in
# vllm/v1/core/block_pool.py:get_new_blocks and died (EngineDeadError),
# poisoning every in-flight/queued request with 500s for the rest of that
# conc point. Confirmed reproducible across conc 128/160/192 on run
# https://github.com/vngcloud/InferenceX/actions/runs/32552828503 (server
# logs: server_logs_gemma4_8k1k_..._conc{128,160,192}_h200-greennode_03);
# conc 8/32 (GPU KV cache peaking at 10%/42%) never hit it. Root cause looks
# like SimpleCPUOffloadConnector's block reclaim/eviction bookkeeping racing
# the scheduler's free-list once the GPU tier fills — it's explicitly flagged
# experimental in vLLM (KVConnectorBase_V1 warning at server startup). Not
# reinstating until that's fixed upstream or root-caused further.
#
# TP is pinned to 1 (single backend); the router is still useful here for its
# cache_aware policy semantics even fronting one worker, and this recipe is
# the template for later multi-replica extension (more worker-urls) without
# changing the client-facing port.
#
# spec-decoding for this recipe is the external-draft-model category
# ("draft_model" in the matrix schema, not internal "mtp") since eagle3 uses
# a separate speculator checkpoint. See runners/launch_h200-greennode.sh's
# SPEC_SUFFIX for how this earns a script name distinct from the
# no-spec-decoding baseline sharing the same precision.

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

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi

DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
if [[ "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi
NUM_SPEC_TOKENS=3

BACKEND_LOG=/workspace/server.log
ROUTER_LOG=/workspace/router.log
BACKEND_PORT=$((PORT + 1))

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# --enable-chunked-prefill is on by default in this vLLM build; passed
# explicitly to document intent and survive a future default flip.
CHUNKED_PREFILL_FLAGS=(
    --enable-chunked-prefill
    --max-num-batched-tokens 16384
    --long-prefill-token-threshold 8192
)

set -x
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$BACKEND_PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    "${CHUNKED_PREFILL_FLAGS[@]}" \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$BACKEND_LOG" 2>&1 &

BACKEND_PID=$!

wait_for_server_ready --port "$BACKEND_PORT" --server-log "$BACKEND_LOG" --server-pid "$BACKEND_PID"

agentic_pip_install --quiet 'vllm-router==0.1.14'
if ! vllm-router --help 2>/dev/null | grep -q -- "cache_aware"; then
    echo "ERROR: installed vllm-router does not expose --policy cache_aware; aborting before benching" >&2
    exit 1
fi

vllm-router \
    --worker-urls "http://localhost:$BACKEND_PORT" \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port "$PORT" \
    --request-timeout-secs 14400 \
    --disable-retries > "$ROUTER_LOG" 2>&1 &

ROUTER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

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
