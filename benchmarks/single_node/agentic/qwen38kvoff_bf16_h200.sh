#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark, vLLM engine — re-run of the c50-90
# breakpoint probe (qwen38-bf16-h200-vllm-agentic, KV_OFFLOADING=none) but
# with vLLM's native DRAM KV-offload path enabled: SimpleCPUOffloadConnector
# (KV_OFFLOAD_BACKEND=vllm-simple), fixed at a 256GB host pool. Same
# prod-exact server args otherwise (context.md §5, boot-bf16.sh): BF16
# weights, fp8 KV, TP4, FlashInfer, thinking ON.
#
# Every prior attempt at this c50-90 probe (runs under commits 444b0eb49/
# f232b177b/e910ba62e, 2026-09-21) was dispatched to the
# cluster:h200-greennode-slurm runner pool before this host was actually
# joined to the Slurm cluster -- every one of those jobs cancelled or failed
# at scheduling, not at the benchmark itself. The Slurm join was fixed
# 2026-09-23 (slurm.conf socket/GRES topology + controller sync).
#
# 2026-09-24: on vLLM v0.25.0/v0.25.1, SimpleCPUOffloadConnector crashed at
# KV cache init for this model regardless of --enforce-eager -- root cause
# was Qwen3.8-27B's hybrid Mamba/GDN block-size padding (1568 tokens vs the
# connector's assumed 32) confusing the connector's block-count math. Manual
# repro on the real GPU node confirmed vLLM v0.29.0 and v0.30.0 both FIX
# this (server boots, serves a real request); v0.25.1 does not (see
# .scratch/vietinbank_qwen38_vllm_kvoffload_v25_crash_log.md and
# _v251_crash_log.md). Image bumped to v0.29.0 accordingly.
#
# TOTAL_CPU_DRAM_GB is fixed at 256 regardless of the dram-utilization set in
# nvidia-master.yaml (see override below) -- matches the SGLang hicache arm's
# 256GB DRAM KV pool for a like-for-like comparison.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend vllm-simple

# Fixed 256GB DRAM KV pool, overriding whatever dram-utilization computed.
export TOTAL_CPU_DRAM_GB=256

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
# Same rationale as the sglhc high-CCU recipes: aiohttp's default 30s
# TCP_USER_TIMEOUT can trip under load at CCU>=70.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"
export AIPERF_GPU_TELEMETRY_METRICS_CSV="benchmarks/single_node/agentic/qwen38kvoff_bf16_h200.gpu_metrics.csv"

export MAX_MODEL_LEN=262144

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# vLLM's native SimpleCPUOffloadConnector: offload KV blocks to a fixed host
# DRAM pool instead of evicting on GPU pressure. Identical prefixes must hash
# to identical block keys, so pin PYTHONHASHSEED.
export PYTHONHASHSEED=42
CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": ${CPU_BYTES_PER_RANK},
    "enable_cross_layers_blocks": "true",
    "lazy_offload": false
  }
}
EOF
)

# Prod-exact vLLM args (context.md §5) plus the native DRAM KV-offload config.
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
    --attention-backend FLASHINFER
    --async-scheduling
    --default-chat-template-kwargs '{"enable_thinking": true}'
    --kv-transfer-config "$OFFLOAD_CONFIG"
)
# 2026-09-24: dropped --enforce-eager after the first real dispatch (c70/c80,
# runs 35953095125/35953102597) came in ~35% below SGLang hicache's
# throughput at the same CCU (73.6k/75.9k vs 112.8k/116.6k tok/s) -- eager
# mode disables CUDA graph capture, which was only kept to match the
# debugging config that confirmed the v0.25.x crash was fixed in v0.29.0/
# v0.30.0. That root cause (Mamba block-size mismatch) is unrelated to
# cudagraph capture, so there's no reason to keep paying the eager-mode
# throughput cost now. c70/c80 numbers above are kept as the eager-mode
# data point; c50/c60/c90 re-run with cudagraph enabled.

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
