#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark — Vietinbank arm G: the closest SGLang
# equivalent of the customer's vLLM 0.25.1 launch (context.md §5), plus
# hierarchical cache with a 128 GB host pool.
#
# Difference from arm D (qwen38sgl_bf16_h200_sglang.sh, run 35440329321):
#   - --max-prefill-tokens 32768. The customer's single vLLM knob
#     --max_num_batched_tokens 32768 splits into two in SGLang; arm D set only
#     --chunked-prefill-size, so prefill still ran under the 16384 default and
#     lost TTFT 1.5-2.7x vs vLLM (context.md §23 root cause).
#   - --default-chat-template-kwargs '{"enable_thinking": true}' stated
#     explicitly instead of relying on the Qwen3 template default.
#   - --enable-hierarchical-cache --hicache-size 128 (arm E's lever, prod runs
#     24 GB on 1xH200; 128 GB fits the 1.4 TB free on h200-greennode_06).
#
# Deliberately NOT carried over from arm E / our prod deployment: EAGLE
# speculative decode (wins at CCU <=33 but regressed throughput -9% and ITL p99
# +106% at CCU 50, context.md §24), --enable-linear-replayssm-spec,
# --mamba-full-memory-ratio 1.5 (OOMs BF16 TP4, crash 35457315965),
# --schedule-policy dfs-weight (diverges from vLLM's fcfs), and
# --mem-fraction-static 0.85 (0.90 is what the customer sets).
#
# Radix cache = customer's --enable-prefix-caching, overlap scheduler =
# --async-scheduling, both SGLang defaults. Tool parser qwen3_coder is the
# closest match for qwen3_xml, which SGLang v0.5.19 does not ship. GDN prefill
# stays on the triton default: FlashInfer GDN prefill is SM100+, so H200 never
# hits the vLLM-style JIT stall (§14 A/B + run 35399747564 c33 postmortem).
# context-length 262144 matches every other arm so the ladder stays comparable.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend hicache

# Resolve model from the runner's HF cache (pre-downloaded on
# h200-greennode_06 = han-1 at /mnt/hf_hub_cache/models--Qwen--Qwen3.8-27B —
# same snapshot every other arm serves; the launcher mounts it at the container
# HF cache path). Fall back to hf download only when the cache is missing.
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

# SemiAnalysis CC traces (full dataset), identical to every other arm.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
# DCGM exporter the greennode launcher starts alongside the job (host network).
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"
# Full DCGM fieldset the customer asked for (GPU/fabric metrics, not just
# serving-level TTFT/ITL). launch_h200-greennode.sh reconfigures this runner's
# dcgm-exporter from the sidecar CSV below (matched by recipe basename), so the
# fields named here always exist on the scrape it points at. Same fieldset as
# the vLLM TP4/DP4/DP2 arms, so all four plot on one axis.
export AIPERF_GPU_TELEMETRY_METRICS_CSV="benchmarks/single_node/agentic/qwen38sglhc_bf16_h200_sglang.gpu_metrics.csv"

# Cap replay context length to model's max context length.
export MAX_MODEL_LEN=262144

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Customer-exact vLLM args translated to SGLang, plus the repo agentic
# convention MAX_RUNNING_REQUESTS=2*CCU (sglang's memory-derived auto cap could
# otherwise schedule more than the vLLM baseline's default), same as arm D.
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
    --max-prefill-tokens 32768
    --attention-backend flashinfer
    --enable-hierarchical-cache
    --hicache-size 128
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking": true}'
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
