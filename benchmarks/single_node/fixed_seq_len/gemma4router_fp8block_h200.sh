#!/usr/bin/env bash

# Gemma-4 31B FP8-block vLLM recipe, fixed-seq-len, ROUTER over 2 independent
# replicas (not internal data-parallel).
#
# Topology contrast with the gemma4dp2 twin: instead of ONE `vllm serve
# --data-parallel-size 2` (a single API server + DP coordinator balancing across
# two engine cores), this launches TWO fully independent `vllm serve
# --tensor-parallel-size 1` processes -- each the full model on its own GPU, its
# own scheduler and its own KV pool -- fronted by a `vllm-router` on $PORT that
# load-balances HTTP requests across them. Two isolated schedulers mean a heavy
# request on one card cannot head-of-line-block the other replica's queue the way
# it can inside one shared DP front-end; the cost is no cross-replica KV sharing.
#
# GPU PINNING: the box's GPUs 0,1 are occupied by another tenant, so the two
# replicas are pinned to GPUs 2 and 3 via CUDA_VISIBLE_DEVICES. nvidia-smi
# ignores CUDA_VISIBLE_DEVICES, so the GPU monitor is scoped with
# `--gpu-ids 2,3` to keep power/util telemetry from being polluted by 0,1.
#
# ROUTING POLICY: round_robin. The 8k1k fixed-seq-len workload is independent
# per-request (no multi-turn sessions), so there is no prefix cache to keep warm
# and no need for session-affine (consistent_hash) routing.
#
# The 8k1k workload, checkpoint, image, runner and conc handling mirror the
# gemma4dp2 twin so the router-vs-internal-DP comparison is clean at equal CONC.

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
# gemma4router model-prefix, like gemma4dp2 records DP_SIZE=2 as a constant).
REPLICA_GPUS=(2 3)
GPU_IDS_CSV="$(IFS=,; echo "${REPLICA_GPUS[*]}")"

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
# Mirrors the gemma4dp2 per-rank intent.
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
    --worker-urls "http://localhost:$BACKEND0_PORT,http://localhost:$BACKEND1_PORT" \
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
