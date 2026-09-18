#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwopus3.6-27B-v2 (Qwen3.6 fine-tune, BF16) AgentX benchmark — Vietinbank
# customer-support case. Replays SemiAnalysis CC traces against a prod-exact
# vLLM 0.25.1 stack with the customer's serving args (context.md §5):
#   BF16 weights, fp8 KV, TP4, FlashInfer, thinking ON.
# max-model-len raised to 300k (cho chắc; model native 262k + YaRN to 1M).
# Chat template: default (customer's qwen3.6-enhanced.jinja is private — not used).
# Tool-call parser: qwen3_xml (customer's config; known mismatch with Qwopus3.6
#   JSON format — ignored per request, does not affect trace replay latency).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_none

# Resolve model from HF cache (pre-downloaded on h200-greennode_06 = han-1 at
# /mnt/hf_hub_cache/models--Jackrong--Qwopus3.6-27B-v2).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi

# SemiAnalysis CC traces (full dataset). build_replay_cmd caps context at
# $MAX_MODEL_LEN below to stay within the model's 300k limit.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

# Cap replay context length to model's max-model-len.
export MAX_MODEL_LEN=300000

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Prod-exact vLLM args (context.md §5). --enable-prefix-caching is load-bearing
# for this workload (89% prefix hit rate in prod). No --max-num-seqs: customer
# config doesn't set it; vLLM default 256 is sufficient for CCU ≤ 50.
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --kv-cache-dtype fp8
    --tensor-parallel-size "$TP"
    --max-model-len 300000
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
