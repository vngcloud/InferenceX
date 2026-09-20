#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark — Vietinbank arm F: DP2 = 2 x TP2 replicas
# (two GPUs each) behind a vllm-router cache_aware. Same four-GPU budget as the
# 1 x TP4 baseline and the 4 x TP1 arm B, so the three are directly comparable
# at equal CCU: the only variable is how the four H200s are partitioned.
#
# Rationale: arm B (4 x TP1) lost badly — single-GPU prefill is ~2.5x slower and
# each replica's KV pool is a quarter of TP4's, which saturated at CCU 50 and
# collapsed the prefix hit rate (90% -> 51%). TP2 halves the replica count while
# keeping sharded prefill and a 2x larger per-replica KV pool, which is the
# plausible sweet spot flagged in the customer notes (context.md §13).
#
# Router topology mirrors the Gemma-4-31B-FP8 production deploy
# (gemma4-31b-fp8-h200-eagle-{a,b} + eagle-router): vllm-router 0.1.14,
# --policy cache_aware, --request-timeout-secs 14400, --disable-retries.
# Prefix-cache-aware routing is MANDATORY for a fair multi-turn agentic A/B:
# turn N+1 must reuse turn N's prefix on the same replica; round-robin would
# force an ~85K cold re-prefill every turn. vLLM native --data-parallel-size is
# deliberately NOT used: its DP load balancer is queue-aware only, not
# prefix-aware.
#
# Per-replica serving args = customer-exact (context.md §5) with
# tensor-parallel-size 2: BF16 weights, fp8 KV, FlashInfer, prefix caching ON,
# thinking ON, max-num-batched-tokens 32768.
# A/B baseline = qwen38-bf16-h200-vllm-agentic (run 35373606847, 1 x TP4).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_none

# Resolve model from HF cache (pre-downloaded on h200-greennode_06 = han-1 at
# /mnt/hf_hub_cache/models--Qwen--Qwen3.8-27B). Identical to the baseline arm.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

# benchmark_lib.sh reassigns AIPERF_UV_CACHE_DIR to an ephemeral /tmp dir at
# source time, so every dispatch cold-downloads aiperf's deps from PyPI (~1 GB).
# Restore the launcher's persistent mount (see gemma4tau2 recipe).
if [ -d /mnt/uv-cache ]; then
    export AIPERF_UV_CACHE_DIR=/mnt/uv-cache
fi

nvidia-smi

# SemiAnalysis CC traces (full dataset), identical to the baseline arm.
# build_replay_cmd caps context at $MAX_MODEL_LEN below to stay within the
# model's 262144 limit.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

# Two replicas; aiperf hits the router on $PORT. Per-replica /metrics feed the
# required vllm: server-metrics gate.
ROUTER_PORT="$PORT"
BACKEND0_PORT=$((PORT + 1))
BACKEND1_PORT=$((PORT + 2))
ROUTER_METRICS_PORT=$((PORT + 10000))
VLLM_ROUTER_VERSION=0.1.14

export AIPERF_SERVER_METRICS_URLS="http://localhost:${BACKEND0_PORT}/metrics,http://localhost:${BACKEND1_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
# DCGM exporter the greennode launcher starts alongside the job (host network).
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"
# Full DCGM fieldset the customer asked for (GPU/fabric metrics, not just
# serving-level TTFT/ITL): SM active/occupancy, NVLink TX/RX + error counters,
# throttle-violation reasons. launch_h200-greennode.sh reconfigures this same
# runner's dcgm-exporter from the sidecar CSV below (matching basename), so
# the fields named here always exist on the scrape it points at. Identical
# fieldset to the TP4 baseline arm, so the two are directly comparable.
export AIPERF_GPU_TELEMETRY_METRICS_CSV="benchmarks/single_node/agentic/qwen38dp2_bf16_h200.gpu_metrics.csv"

# Cap replay context length to model's max-model-len.
export MAX_MODEL_LEN=262144

# TP2 is a topology neither the TP4 baseline nor arm B ever ran, so its prefill
# shapes are new to this box's FlashInfer JIT cache. If boot warmup misses one
# and it compiles under load, the compile can outlast the default 300s
# execute-model RPC deadline and the engine core kills the server (fatal "RPC
# call to sample_tokens timed out", seen in run 35399747564 c33). 1800s lets a
# one-time compile finish instead.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

mkdir -p "$RESULT_DIR"
# worker-0's log MUST be $RESULT_DIR/server.log: the vLLM server-metrics adapter
# parses "GPU KV cache size: N tokens" from that exact path for
# kv_cache.gpu_total_tokens (both replicas are identical, so one suffices).
SERVER_LOG_0="$RESULT_DIR/server.log"
SERVER_LOG_1="$RESULT_DIR/server1.log"
ROUTER_LOG="$RESULT_DIR/router.log"

# Customer-exact vLLM args (context.md §5) with TP=2. $1 = comma-separated GPU
# pair, $2 = port, $3 = log path, $4 = label used in the command dump filename.
launch_replica() {
    local gpus="$1" port="$2" log="$3" label="$4"
    local cmd=(
        vllm serve "$MODEL_PATH"
        --served-model-name "$MODEL"
        --host 0.0.0.0
        --port "$port"
        --trust-remote-code
        --kv-cache-dtype fp8
        --tensor-parallel-size 2
        --max-model-len 262144
        --gpu-memory-utilization 0.90
        --enable-auto-tool-choice
        --enable-prefix-caching
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
        --max-num-batched-tokens 32768
        --attention-backend FLASHINFER
        --async-scheduling
        --default-chat-template-kwargs '{"enable_thinking": true}'
    )
    printf '%q ' "${cmd[@]}" | tee "$RESULT_DIR/vllm_command_${label}.txt"
    printf '\n' | tee -a "$RESULT_DIR/vllm_command_${label}.txt"
    CUDA_VISIBLE_DEVICES="$gpus" "${cmd[@]}" > "$log" 2>&1 &
}

launch_replica "0,1" "$BACKEND0_PORT" "$SERVER_LOG_0" gpu01
SERVER0_PID=$!
launch_replica "2,3" "$BACKEND1_PORT" "$SERVER_LOG_1" gpu23
SERVER1_PID=$!
wait_for_server_ready --port "$BACKEND0_PORT" --server-log "$SERVER_LOG_0" --server-pid "$SERVER0_PID"
wait_for_server_ready --port "$BACKEND1_PORT" --server-log "$SERVER_LOG_1" --server-pid "$SERVER1_PID"

# cache_aware router in front of the two replicas (prefix-match + load-balance
# fallback keeps session turns sticky to one replica).
agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
vllm-router \
    --worker-urls "http://localhost:$BACKEND0_PORT" "http://localhost:$BACKEND1_PORT" \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port "$ROUTER_PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$ROUTER_METRICS_PORT" \
    --request-timeout-secs 14400 \
    --disable-retries > "$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
wait_for_server_ready --port "$ROUTER_PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
