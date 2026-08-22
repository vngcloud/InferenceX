#!/usr/bin/env bash

# Gemma-4 31B FP8-block (RedHatAI/gemma-4-31B-it-FP8-block) vLLM recipe,
# fixed-seq-len — multi-replica router + eagle3 speculative decoding variant
# of gemma4_fp8block_h200.sh.
#
# Adds, on top of the baseline single-vllm-serve recipe:
#   - NUM_REPLICAS independent full-model TP1 backends, one per GPU
#     (CUDA_VISIBLE_DEVICES-pinned), each own port
#   - eagle3 speculative decoding (RedHatAI/gemma-4-31B-it-speculator.eagle3,
#     num_speculative_tokens=3) on every replica
#   - chunked-prefill anti-head-of-line-blocking tuning
#     (--long-prefill-token-threshold caps a single long prefill's tokens per
#     step so it can't starve other requests' decode steps;
#     --max-num-batched-tokens is raised to fit a full chunk plus decode
#     tokens from in-flight requests in the same step)
#   - a vllm-router (cache_aware policy) in front of all replicas, so the
#     benchmark client always talks to the router's port and load-balances
#     across backends by cache affinity
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
# TP has no real tensor-parallelism here: each replica is a full standalone
# TP1 copy of the model on its own GPU. TP is repurposed as NUM_REPLICAS
# (matrix's tp field = replica count) purely so utils/process_result.py's
# num_gpus = tp*pp*pcp bookkeeping divides tput_per_gpu correctly and the
# power-validation GPU-count expectation matches the real GPU footprint
# (NUM_REPLICAS distinct GPUs genuinely light up, one per replica) -- it is
# not a real vllm --tensor-parallel-size. Each individual backend is always
# launched with --tensor-parallel-size 1.
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

ROUTER_LOG=/workspace/router.log
NUM_REPLICAS="$TP"
# DIAGNOSTIC: pinned to GPUs 2,3 instead of 0,1 -- isolating GPU index as
# the one remaining untested variable against the original
# gemma4router_fp8block_h200.sh's 625 tok/s/gpu @ c32 (this script's own
# 0,1-pinned runs only got 320-350). NVLink topology confirmed symmetric
# (all pairs NV18, all links up) so this isn't expected to matter, but it
# hasn't actually been tested. Revert to the 0-indexed loop once compared.
REPLICA_GPUS=(2 3)
# DIAGNOSTIC: doubled (was ceil(CONC/NUM_REPLICAS), the exact expected
# per-replica load under even router splitting) to test whether the tight
# --max-num-seqs ceiling itself -- e.g. narrower CUDA-graph batch-size
# coverage, less scheduler admission headroom for bursts -- explains the
# 2-replica output_tput_per_gpu regression vs single-replica, independent
# of chunked-prefill/router-policy/eagle3 (all three already ruled out).
PER_REPLICA_MAX_SEQS=$(( 2 * (CONC + NUM_REPLICAS - 1) / NUM_REPLICAS ))

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# RULED OUT as the cause of the 2-replica c8/c32 output_tput_per_gpu
# regression (232/467 tok/s single-replica -> 140/350 tok/s 2-replica):
# disabling these produced 137 tok/s @ c8, statistically the same as with
# them on. Restored. Current suspect: router policy -- see ROUTER_POLICY
# below (testing round_robin against the cache_aware baseline).
CHUNKED_PREFILL_FLAGS=(
    --enable-chunked-prefill
    --max-num-batched-tokens 16384
    --long-prefill-token-threshold 8192
)

# DIAGNOSTIC: chunked-prefill and router policy (round_robin vs cache_aware)
# were both ruled out as causes of the 2-replica output_tput_per_gpu
# regression vs single-replica (232/467 -> 140/350, then 338 @ round_robin
# c32 -- statistically the same as cache_aware's 350). Disabling eagle3
# next to see if speculative decoding itself is the variable. Restore the
# --speculative-config line above once compared.
SPEC_DECODE_FLAGS=()

set -x

BACKEND_PORTS=()
BACKEND_PIDS=()
for i in "${!REPLICA_GPUS[@]}"; do
    gpu="${REPLICA_GPUS[$i]}"
    port=$((PORT + 1 + i))
    log="/workspace/server${i}.log"

    CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$port" \
        --served-model-name "$MODEL" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$PER_REPLICA_MAX_SEQS" \
        "${CHUNKED_PREFILL_FLAGS[@]}" \
        "${SPEC_DECODE_FLAGS[@]}" \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 > "$log" 2>&1 &

    BACKEND_PORTS+=("$port")
    BACKEND_PIDS+=("$!")
done

for i in "${!BACKEND_PORTS[@]}"; do
    wait_for_server_ready --port "${BACKEND_PORTS[$i]}" --server-log "/workspace/server${i}.log" --server-pid "${BACKEND_PIDS[$i]}"
done

agentic_pip_install --quiet 'vllm-router==0.1.14'

# round_robin (338 tok/s/gpu @ c32) was statistically the same as
# cache_aware (350) -- ruled out too. Reverted to the intended baseline.
ROUTER_POLICY=cache_aware
if ! vllm-router --help 2>/dev/null | grep -q -- "$ROUTER_POLICY"; then
    echo "ERROR: installed vllm-router does not expose --policy $ROUTER_POLICY; aborting before benching" >&2
    exit 1
fi

WORKER_URLS=()
for port in "${BACKEND_PORTS[@]}"; do WORKER_URLS+=("http://localhost:$port"); done

vllm-router \
    --worker-urls "${WORKER_URLS[@]}" \
    --policy "$ROUTER_POLICY" \
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
