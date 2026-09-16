#!/usr/bin/env bash
set -eo pipefail
set -x

# Gemma-4-31B customer-service (tau2) agentic arm, 1:1 with prod serving.
#
# Replays the private tau2 corpus (thangquang09/customer-service-agent-traces,
# mooncake_trace raw-content: messages + tools verbatim) against a self-hosted,
# production-exact Gemma-4 vLLM stack: two TP=1 replicas (one GPU each) behind a
# vllm-router cache_aware. Distinguished from a cc-traces coding run purely by
# the model-prefix gemma4tau2; it reuses the agentic-coding scenario and the
# shared build_replay_cmd/replay path unchanged, so its artifact is
# schema-identical to a cc-traces arm and joins the same analysis.
#
# Differs from an existing cc-traces vLLM arm in only three places:
#   1. dataset source  -> local mooncake_trace file instead of a Weka public loader
#   2. tokenizer        -> google/gemma-4-31b-it (via $MODEL)
#   3. served engine    -> vLLM/Gemma instead of sglang/GLM
#
# Prod-exact serving, mirroring the Gemma-4-31B-FP8 production deploy
# (gemma4-31b-fp8-h200-eagle-{a,b} + eagle-router): vLLM v0.25.0 image, FP8-block
# weights, TP=1 x 2 replicas + vllm-router 0.1.14 cache_aware, BF16 KV (auto),
# max-model-len 262144, max-num-seqs 64, chunked-prefill on, spec-decode OFF,
# VLLM_ATTENTION_BACKEND left unset (= TRITON default, the prod A/B's
# non-override arm). gpu-memory-utilization 0.92, max-num-batched-tokens 16384
# and long-prefill-token-threshold 8192 are the prod deploy's values too.
#
# Required env (harness-provided): MODEL TP CONC KV_OFFLOADING RESULT_DIR
# DURATION PORT. The ambient HF_TOKEN must be able to read the private tau2
# dataset and the gated google/gemma-4-31b-it tokenizer.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT
require_agentic_kv_offload_none

# Prod serves the FP8-block quant of gemma-4-31b-it; $MODEL is the un-quantized
# HF id, used only as the wire/served name and the aiperf tokenizer.
WEIGHTS="RedHatAI/gemma-4-31B-it-FP8-block"
TAU2_REPO="thangquang09/customer-service-agent-traces"

# Two replicas front-ended by the router. aiperf hits the router on $PORT; the
# per-replica /metrics feed the required vllm: server-metrics gate below.
ROUTER_PORT="$PORT"
BACKEND0_PORT=$((PORT + 1))
BACKEND1_PORT=$((PORT + 2))
ROUTER_METRICS_PORT=$((PORT + 10000))
VLLM_ROUTER_VERSION=0.1.14

# The vLLM branch of the server-metrics schema; sglang: here would fail the
# required-metric gate / select the wrong backend adapter.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${BACKEND0_PORT}/metrics,http://localhost:${BACKEND1_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
# DCGM exporter the greennode launcher starts alongside the job (host network).
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

install_agentic_deps
nvidia-smi

# Model weights are public -> ambient HF_TOKEN. Idempotent: a cache hit is a
# no-op. Serve by HF id so vLLM resolves from the mounted HF_HUB_CACHE.
"$AIPERF_HF_CLI" download "$WEIGHTS"

# The tau2 corpus is a PRIVATE repo; the ambient HF_TOKEN (the official CI token)
# must be able to read it.
TAU2_DIR=$("$AIPERF_HF_CLI" download --repo-type dataset "$TAU2_REPO")
mapfile -t TAU2_FILES < <(find "$TAU2_DIR" -maxdepth 2 -name '*.jsonl' | sort)
if [ "${#TAU2_FILES[@]}" -ne 1 ]; then
    echo "Error: expected exactly one .jsonl in $TAU2_DIR, found ${#TAU2_FILES[@]}: ${TAU2_FILES[*]}" >&2
    exit 1
fi
TAU2_FILE="${TAU2_FILES[0]}"
echo "tau2 mooncake_trace input: $TAU2_FILE"

mkdir -p "$RESULT_DIR"
# worker-0's log MUST be $RESULT_DIR/server.log: the vLLM server-metrics adapter
# parses "GPU KV cache size: N tokens" from that exact path for
# kv_cache.gpu_total_tokens (both replicas are identical, so one suffices).
SERVER_LOG_0="$RESULT_DIR/server.log"
SERVER_LOG_1="$RESULT_DIR/server1.log"
ROUTER_LOG="$RESULT_DIR/router.log"

# Prod-exact vLLM args, spec-decode OFF and BF16 KV (no --kv-cache-dtype => auto).
# $1 = GPU index, $2 = port, $3 = log path.
launch_replica() {
    local gpu="$1" port="$2" log="$3"
    local cmd=(
        vllm serve "$WEIGHTS"
        --served-model-name "$MODEL" gemma-4-31b-it
        --host 0.0.0.0
        --port "$port"
        --tensor-parallel-size 1
        --gpu-memory-utilization 0.92
        --max-model-len 262144
        --max-num-seqs 64
        --max-num-batched-tokens 16384
        --enable-chunked-prefill
        --long-prefill-token-threshold 8192
        --enable-auto-tool-choice
        --tool-call-parser gemma4
        --reasoning-parser gemma4
    )
    printf '%q ' "${cmd[@]}" | tee "$RESULT_DIR/vllm_command_gpu${gpu}.txt"
    printf '\n' | tee -a "$RESULT_DIR/vllm_command_gpu${gpu}.txt"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1 &
}

launch_replica 0 "$BACKEND0_PORT" "$SERVER_LOG_0"
SERVER0_PID=$!
launch_replica 1 "$BACKEND1_PORT" "$SERVER_LOG_1"
SERVER1_PID=$!
wait_for_server_ready --port "$BACKEND0_PORT" --server-log "$SERVER_LOG_0" --server-pid "$SERVER0_PID"
wait_for_server_ready --port "$BACKEND1_PORT" --server-log "$SERVER_LOG_1" --server-pid "$SERVER1_PID"

# cache_aware router in front of the two replicas (prod-exact topology).
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

# Replay input override: swap the Weka public-dataset default for the local
# mooncake_trace file. build_replay_cmd appends $TRACE_SOURCE_FLAG verbatim, so
# we set it directly instead of calling the Weka-only resolve_trace_source.
export TRACE_SOURCE_FLAG="--custom-dataset-type mooncake_trace --input-file $TAU2_FILE"
# The local --input-file corpus is unpinned; runs are stamped submission_valid:
# false, which is expected and acceptable.
export AIPERF_UNSAFE_OVERRIDE=true
# Dual stop: 20-minute cap OR one full pass over the 16,798-turn pool, whichever
# fires first -- the cap is what keeps low-CCU smoke points from running forever.
# This is a deliberate recipe-specific override, not a discarded caller input:
# the matrix generator hardcodes duration=3600 for every agentic-coding row
# (infx/matrix/generate.py DEFAULT_AGENTIC_DURATION_SECONDS), with no per-arm
# field to set it, and that generator is synced with upstream and must not be
# forked for one arm. DURATION feeds --benchmark-duration inside build_replay_cmd.
export DURATION=1200

build_replay_cmd "$RESULT_DIR"
# Override the two Weka-shaped assumptions build_replay_cmd hardcodes:
#  - entry cap 393 (with-subagents corpus) -> the full tau2 corpus (1,576 sessions;
#    the loader treats it as min(cap, available)).
#  - add the turn-count half of the dual stop; --request-count and
#    --benchmark-duration coexist, stopping at whichever fires first.
REPLAY_CMD="${REPLAY_CMD/--num-dataset-entries 393/--num-dataset-entries 1576}"
if [[ "$REPLAY_CMD" != *"--num-dataset-entries 1576"* ]]; then
    echo "Error: build_replay_cmd no longer emits '--num-dataset-entries 393'; the tau2 entry-cap override missed. Refresh it before running." >&2
    exit 1
fi
REPLAY_CMD+=" --request-count 16798"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
