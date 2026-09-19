#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark — Vietinbank arm A: long-prefill chunk 16k.
# Identical to qwen38_bf16_h200.sh (prod-exact customer stack) except one flag:
#   --long-prefill-token-threshold 16384
# The vLLM v1 scheduler caps every prefill chunk at 16k tokens
# (v1/core/sched/scheduler.py: num_new_tokens = threshold when a request's
# remaining prefill exceeds it), so an ~85K prefill schedules as ~6 x 16k
# chunks instead of ~3 x 32k (max-num-batched-tokens 32768). Running decodes
# interleave between chunks: expected ITL-tail win under high concurrency at a
# slight cold-TTFT cost. A/B baseline = qwen38-bf16-h200-vllm-agentic
# (run 35373606847).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_none

# Resolve model from HF cache (pre-downloaded on h200-greennode_06 = han-1 at
# /mnt/hf_hub_cache/models--Qwen--Qwen3.8-27B).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi

# SemiAnalysis CC traces (full dataset), identical to the baseline arm.
# build_replay_cmd caps context at $MAX_MODEL_LEN below to stay within the
# model's 262144 limit.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

# Cap replay context length to model's max-model-len.
export MAX_MODEL_LEN=262144

# The 16k chunk cap creates prefill shapes that boot warmup never covers; the
# first on-load FlashInfer GDN prefill JIT compile for a new shape can exceed
# the default 300s execute-model RPC deadline, at which point the engine core
# kills the whole server (fatal "RPC call to sample_tokens timed out", seen in
# run 35399747564 c33). 1800s lets the one-time compile finish instead.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Customer-exact vLLM args (context.md §5) + long-prefill chunk cap 16k.
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --kv-cache-dtype fp8
    --tensor-parallel-size "$TP"
    --max-model-len 262144
    --gpu-memory-utilization 0.90
    --enable-auto-tool-choice
    --enable-prefix-caching
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --max-num-batched-tokens 32768
    --long-prefill-token-threshold 16384
    --attention-backend FLASHINFER
    --async-scheduling
    --default-chat-template-kwargs '{"enable_thinking": true}'
)

write_command "$RESULT_DIR/vllm_command.txt" "${VLLM_CMD[@]}"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
