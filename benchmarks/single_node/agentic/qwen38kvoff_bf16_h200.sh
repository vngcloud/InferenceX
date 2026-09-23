#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B BF16 AgentX benchmark, vLLM engine — re-run of the c60-90
# breakpoint probe (qwen38-bf16-h200-vllm-agentic, KV_OFFLOADING=none) but
# with vLLM's native DRAM KV-offload path enabled: SimpleCPUOffloadConnector
# (KV_OFFLOAD_BACKEND=vllm-simple), fixed at a 128GB host pool. Same
# prod-exact server args otherwise (context.md §5, boot-bf16.sh): BF16
# weights, fp8 KV, TP4, FlashInfer, thinking ON.
#
# Every prior attempt at this c60-90 probe (runs under commits 444b0eb49/
# f232b177b/e910ba62e, 2026-09-21) was dispatched to the
# cluster:h200-greennode-slurm runner pool before this host was actually
# joined to the Slurm cluster -- every one of those jobs cancelled or failed
# at scheduling, not at the benchmark itself. The Slurm join was fixed
# 2026-09-23 (slurm.conf socket/GRES topology + controller sync); this is the
# first real attempt on working Slurm runners.
#
# TOTAL_CPU_DRAM_GB is fixed at 128 regardless of the dram-utilization set in
# nvidia-master.yaml (see override below) -- the customer's ask here is a
# specific 128GB DRAM KV pool size, not a fraction of whatever's free on the
# node.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend vllm-simple

# Fixed 128GB DRAM KV pool, overriding whatever dram-utilization computed.
export TOTAL_CPU_DRAM_GB=128

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
    # NOTE 2026-09-24: --enforce-eager alone does NOT fix this arm. Kept
    # (harmless, simpler code path) but the real crash is upstream of
    # cudagraph profiling: with --enforce-eager still on, the SAME reshape
    # (gpu_model_runner.py _reshape_kv_cache_tensors ->
    # attn_utils._reshape_attention_kv_cache) fails on the REAL (non-profiling)
    # KV cache init too -- "shape '[440902, 2, 32, 1, 256]' is invalid for
    # input of size 147423232". Root cause: Qwen3.8-27B is a hybrid
    # Mamba/GDN+attention model, so vLLM pads the attention block size to
    # 1568 tokens to match the Mamba page size ("Setting attention block
    # size to 1568 tokens to ensure that attention page size is >= mamba
    # page size", interface.py:890) instead of the usual 32.
    # SimpleCPUOffloadConnector's own block-count/shape computation appears
    # to assume the default 32-token block size regardless (1568/32 = 49,
    # exactly the ratio of the two crash's mismatched block counts:
    # 25088/512 and 440902/8996). This is an upstream vLLM v0.25.0
    # SimpleCPUOffloadConnector incompatibility with hybrid Mamba/attention
    # models, not fixable via recipe/CLI args -- confirmed on 2 independent
    # attempts (with and without --enforce-eager), same ~49x mismatch both
    # times. This arm is BLOCKED until either vLLM patches the connector for
    # non-default block sizes, or a different offload backend (e.g.
    # mooncake) is tried instead.
    --enforce-eager
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
