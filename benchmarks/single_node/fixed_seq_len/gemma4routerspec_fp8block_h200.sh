#!/usr/bin/env bash

# Gemma-4 31B FP8-block vLLM recipe, fixed-seq-len, ROUTER over 2 independent
# replicas (not internal data-parallel), WITH EAGLE3 speculative decoding.
#
# This is the spec-on twin of gemma4router_fp8block_h200.sh: identical topology
# (two fully independent `vllm serve --tensor-parallel-size 1` processes pinned
# to GPUs 2,3, fronted by a `vllm-router` on $PORT with round_robin), same
# checkpoint / image / runner / 8k1k workload -- the only added axis is a
# --speculative-config on EACH replica pointing at RedHat's EAGLE3 speculator
# (trained for this exact target checkpoint). A 2B EAGLE3 draft proposes tokens
# that the 31B target verifies in parallel; output is unchanged, decode
# throughput/TPOT improve when proposals are accepted. gemma-4 EAGLE3 was
# upstreamed in vLLM ~v0.22.0, so the pinned v0.25.0 image supports it with no
# image change. This measures the router / isolated-KV-pool topology WITH spec
# decoding against its spec-off gemma4router sibling at equal CONC.
#
# GPU PINNING: the two replicas are pinned to GPUs 2 and 3 via
# CUDA_VISIBLE_DEVICES, keeping GPUs 0,1 free (they are held by another tenant
# on the _04 box this topology was first run on, and on _03 they are the pair
# the gemma4dp2 sweep's --data-parallel-size 2 grabs by default). nvidia-smi
# ignores CUDA_VISIBLE_DEVICES, so the GPU monitor is scoped with `--gpu-ids
# 2,3` to keep power/util telemetry from being polluted by 0,1. Each replica
# loads its own copy of the EAGLE3 draft and verifies independently.
#
# ROUTING POLICY: round_robin. The 8k1k fixed-seq-len workload is independent
# per-request (no multi-turn sessions), so there is no prefix cache to keep warm
# and no need for session-affine (consistent_hash) routing.
#
# NOTE ON ACCEPTANCE: the fixed-seq-len workload feeds *random* token content,
# which understates real EAGLE3 gains -- a draft head cannot predict random
# tokens. --use-chat-template wraps prompts in the tokenizer's chat structure
# (same mitigation as gemma4eagle) so at least the structural tokens are
# predictable, but a faithful acceptance-rate measurement needs the agentic
# (real-trace) scenario. Treat the fixed-seq-len number as a conservative floor.

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

# Two independent single-GPU replicas pinned to GPUs 2,3 (0,1 are occupied).
# Each replica is CUDA_VISIBLE_DEVICES-pinned to exactly one card, so it is TP1
# by construction regardless of the matrix's TP input (recorded via the
# gemma4routerspec model-prefix, like gemma4router records the router topology).
REPLICA_GPUS=(2 3)
GPU_IDS_CSV="$(IFS=,; echo "${REPLICA_GPUS[*]}")"

DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

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
# --speculative-config "model" id resolves offline for both replicas.
hf download "$DRAFT_MODEL"

BACKEND0_LOG=/workspace/server0.log
BACKEND1_LOG=/workspace/server1.log
ROUTER_LOG=/workspace/router.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# Backend ports sit above $PORT; the router owns $PORT so the client and the
# eval both hit the router endpoint unchanged.
BACKEND0_PORT=$((PORT + 1))
BACKEND1_PORT=$((PORT + 2))
ROUTER_PROM_PORT=$((PORT + 10000))

# Reap the two backends and the router on exit so the pinned GPUs are released
# even if a later step fails (the container is --rm, but explicit kills keep the
# card clean between sweep points and make logs unambiguous).
cleanup() {
    [[ -n "${ROUTER_PID:-}" ]]   && kill "$ROUTER_PID"   2>/dev/null || true
    [[ -n "${BACKEND0_PID:-}" ]] && kill "$BACKEND0_PID" 2>/dev/null || true
    [[ -n "${BACKEND1_PID:-}" ]] && kill "$BACKEND1_PID" 2>/dev/null || true
}
trap cleanup EXIT

start_gpu_monitor --gpu-ids "$GPU_IDS_CSV"

# --max-num-seqs = CONC per backend is non-binding: the router spreads requests
# across the two replicas so each sees ~CONC/2, and a replica that transiently
# receives more than that is not throttled below the client's --max-concurrency.
# --speculative-config carries the EAGLE3 draft; each replica loads its own copy
# and verifies independently.
launch_backend() {
    local gpu="$1" port="$2" log="$3"
    set -x
    CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$port" \
        --served-model-name "$MODEL" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.92 \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$CONC" \
        --max-num-batched-tokens 8192 \
        --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 > "$log" 2>&1 &
    set +x
}

launch_backend "${REPLICA_GPUS[0]}" "$BACKEND0_PORT" "$BACKEND0_LOG"
BACKEND0_PID=$!
launch_backend "${REPLICA_GPUS[1]}" "$BACKEND1_PORT" "$BACKEND1_LOG"
BACKEND1_PID=$!

wait_for_server_ready --port "$BACKEND0_PORT" --server-log "$BACKEND0_LOG" --server-pid "$BACKEND0_PID"
wait_for_server_ready --port "$BACKEND1_PORT" --server-log "$BACKEND1_LOG" --server-pid "$BACKEND1_PID"

# Standalone vllm-router (same package the minimaxm3 agentic recipe uses). Here
# it fronts TWO independent worker URLs with round_robin; --intra-node-data-
# parallel-size is left at its default 1 because each worker is a standalone TP1
# server, not an internal-DP engine.
pip install -q vllm-router==0.1.14

set -x
vllm-router \
    --worker-urls "http://localhost:$BACKEND0_PORT" "http://localhost:$BACKEND1_PORT" \
    --policy round_robin \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$ROUTER_PROM_PORT" \
    --request-timeout-secs 14400 \
    --disable-retries > "$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
set +x

wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

pip install -q datasets pandas

# --use-chat-template: see the acceptance note in the header. Spec-decoding
# throughput on raw random tokens is misleadingly low; the chat template is the
# fixed-seq-len mitigation.
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
    --trust-remote-code \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
