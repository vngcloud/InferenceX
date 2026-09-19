#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark — Vietinbank arm D: SGLang v0.5.19 TP4.
# Customer-mirror of the qwen38 TP4 vLLM baseline (run 35373606847) with the
# engine swapped to SGLang. Serving args translated from the customer's vLLM
# launch (customer-support context.md §5, serve/docker-compose.sglang.yml):
#   BF16 weights, fp8 KV, TP4, FlashInfer attention, chunked prefill 32768.
# SGLang defaults cover the remaining customer args: radix cache (prefix
# caching) and overlap scheduler (async scheduling) are always on; thinking
# is ON via the Qwen3 template default. Tool parser qwen3_coder is the
# closest SGLang equivalent of the customer's qwen3_xml (no qwen3_xml in
# SGLang). GDN prefill stays on the triton default: SGLang compiles it at
# boot, so there is no vLLM-style flashinfer GDN JIT stall (§14 A/B + run
# 35399747564 c33 crash postmortem).
# max-model-len 262144 matches the vLLM baseline arm (native max_position).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_none

# Resolve model from the runner's HF cache (pre-downloaded on
# h200-greennode_06 = han-1 at /mnt/hf_hub_cache/models--Qwen--Qwen3.8-27B —
# same snapshot the vLLM baseline serves; launcher mounts it at the container
# HF cache path). Prefer the mounted cache so the SGLang image needs no hf
# CLI; fall back to hf download only when the cache is missing.
CONTAINER_MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
elif [[ -d "$CONTAINER_MODEL" ]]; then
    export MODEL_PATH="$(ls -d "$CONTAINER_MODEL"/snapshots/*/ | head -1)"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi

# SemiAnalysis CC traces (full dataset), identical to the vLLM baseline arm.
# build_replay_cmd caps context at $MAX_MODEL_LEN below to stay within the
# model's 262144 limit.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

# Cap replay context length to model's max context length.
export MAX_MODEL_LEN=262144

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Customer-exact vLLM args translated to SGLang, plus the repo agentic
# convention MAX_RUNNING_REQUESTS=2*CCU (sglang's memory-derived auto cap
# could otherwise schedule more than the vLLM baseline's default).
MAX_RUNNING_REQUESTS=$((2 * CONC))
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --context-length 262144
    --mem-fraction-static 0.90
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --kv-cache-dtype fp8_e4m3
    --chunked-prefill-size 32768
    --attention-backend flashinfer
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-metrics
    --enable-cache-report
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
