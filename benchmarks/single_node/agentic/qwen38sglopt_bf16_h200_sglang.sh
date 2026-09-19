#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark — "best effort" SGLang v0.5.19 TP4.
# Base = arm D (qwen38sgl_bf16_h200_sglang.sh, run 35440329321): BF16 weights,
# fp8 KV, TP4, FlashInfer attention, chunked prefill 32768. Arm D had no
# speculative decoding at all and lost TTFT 1.5-2.7x vs vLLM; root cause
# (customer-support/2026-16-09-Vietinbank/context.md §23) = max_prefill_tokens
# default 16384 too small next to chunked_prefill 32768.
#
# This arm folds in every optimization already validated on our own live prod
# Qwen3.8-27B-FP8 deployment (1xH200, GLM-5.3-Flash k8s cluster,
# deployment/qwen38-27b-fp8) on top of arm D's TP4 BF16 base:
#   - --max-prefill-tokens 32768   (the untested TTFT-gap fix from §23)
#   - EAGLE self-speculative decoding via the model's built-in MTP head
#     (qwen3_5_mtp.py / Qwen3_5ForCausalLMMTP) — arm D ran with none at all
#   - --enable-linear-replayssm-spec + --mamba-full-memory-ratio 1.5
#   - --schedule-policy dfs-weight
#   - --enable-hierarchical-cache --hicache-size 24 (prod's live value)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend hicache

# Resolve model from the runner's HF cache, same snapshot as arm D
# (h200-greennode_06 = han-1, /mnt/hf_hub_cache/models--Qwen--Qwen3.8-27B).
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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

# Cap replay context length to model's max context length.
export MAX_MODEL_LEN=262144

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --context-length 262144
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --kv-cache-dtype fp8_e4m3
    --chunked-prefill-size 32768
    --max-prefill-tokens 32768
    --attention-backend flashinfer
    --schedule-policy dfs-weight
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-linear-replayssm-spec
    --enable-hierarchical-cache
    --hicache-size 24
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
