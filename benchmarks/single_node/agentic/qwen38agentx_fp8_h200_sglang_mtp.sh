#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B AgentX (cc-traces, agentic-coding) arm: prod-exact serving plus
# the GLM-5.3-Flash dpafp8 levers that transfer to this mamba-hybrid family.
# The cc-traces twin of qwen38tau2_fp8_h200_sglang_mtp.sh (same serving config,
# same levers, same artifact schema): only the trace source differs -- the
# standard Weka public cc-traces loader instead of the private tau2
# mooncake_trace corpus -- so DURATION stays the generator's agentic default
# (3600) and the entry cap stays the loader's own.
#
# Serving shape = the prod Qwen3.8-27B-FP8 deployment (deployment/qwen38-27b-fp8
# on the GLM-5.3-Flash k8s cluster: sglang v0.5.19, 1x H200, TP1, FP8 weights,
# EAGLE off the built-in qwen3_5 MTP head 3/1/4, --enable-linear-replayssm-spec,
# --schedule-policy dfs-weight, HiCache 24), plus the levers validated on the
# GLM-5.3-Flash dpafp8 Current-Best ladder (docs/handoffs/2026-09-28-glm53flash
# -dpa-fp8-agentx-ladder.md, slurm job 939) and green on the qwen38tau2 ladder
# (run 37340703230, CCU 8/16/32/64 all success):
#   - --max-mamba-cache-size 384 pinned: the default --mamba-full-memory-ratio
#     under-sizes the hybrid state pool ~6x and crashed GLM at CCU 32; 384
#     slots at dp_size 1 = state cap 128 running requests (S=3 with
#     extra_buffer_lazy + skip-decode-lock) vs max-running 64 -> 2x headroom
#     for radix retention. Measured on this model (tau2 c8 boot log): 55.2 GB
#     pool = 143.7 MB/slot, KV fp8 32.2 KB/token -> token_equiv ~4,463.
#   - --mem-fraction-static 0.88 (survived the full ladder incl. c64 graphs)
#   - --mamba-radix-cache-strategy extra_buffer_lazy +
#     SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 (state slots per running request
#     S=5 -> 3)
#   - --kv-cache-dtype fp8_e4m3 (arm E qwen38sglopt already ran it green on
#     this exact model + image)
#   - --max-prefill-tokens 32768 next to --chunked-prefill-size 32768 (arm E's
#     TTFT fix: the 16384 default throttles prefill)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Not ported from GLM dpafp8: --schedule-policy hrrn (v0.5.19's choices are
# lpm/random/fcfs/dfs-weight/lof/priority/routing-key -- prod's dfs-weight
# stays) and --mamba-ssm-dtype bfloat16 (flashinfer's GDN MTP verify path
# asserts fp32 initial_state on this arch family; arm E ran green on the fp32
# default -- re-evaluate only on a newer image).
#
# Required env (harness-provided): MODEL TP CONC KV_OFFLOADING RESULT_DIR
# DURATION PORT EVAL_ONLY.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend hicache

# Prod serves the FP8 checkpoint; $MODEL is the un-quantized HF id, used only
# as the wire/served name and the aiperf tokenizer.
WEIGHTS="Qwen/Qwen3.8-27B-FP8"

# Single server, no router (TP1 mirrors the prod deployment; the GLM router
# was a dp-attention multi-worker lever that does not apply at dp1). aiperf
# hits the server on $PORT directly; the sglang: prefix selects the sglang
# branch of the server-metrics schema.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
# DCGM exporter the hardware-hcm launcher starts alongside the job.
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"
# Public cc-traces corpus via aiperf's Weka loader (same loader override as
# the GLM-5.3-Flash agentx arms; entry cap and pinning stay loader-owned).
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k

# benchmark_lib.sh reassigns AIPERF_UV_CACHE_DIR to an ephemeral /tmp dir at
# source time, so every dispatch cold-downloads aiperf's deps from PyPI.
# Restore the launcher's persistent mount so reruns hit the uv cache -- but
# only when the job user can actually write it: on hgx-h200-01 /mnt/uv-cache
# is root-owned 755 (created 2026-09-29), and uv aborts at
# "Failed to initialize cache at /mnt/uv-cache: CACHEDIR.TAG: Permission
# denied" (run 37331049314). The -w probe runs as the container's remapped
# user, so an unwritable mount falls back to benchmark_lib's ephemeral cache.
if [ -d /mnt/uv-cache ] && [ -w /mnt/uv-cache ]; then
    export AIPERF_UV_CACHE_DIR=/mnt/uv-cache
fi
install_agentic_deps
nvidia-smi

# Public checkpoint; idempotent cache hit against the mounted HF_HUB_CACHE
# (pre-staged on the pinned node by the dispatch checklist).
"$AIPERF_HF_CLI" download "$WEIGHTS"

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# AgentX concurrency counts live session trees; leave room for fan-out
# without spending state-pool slots above the pinned mamba budget. The 64 cap
# keeps max-running inside the 384-slot state pool's 128-request capacity
# with radix-retention headroom.
MAX_RUNNING_REQUESTS=$((2 * CONC))
if [ "$MAX_RUNNING_REQUESTS" -gt 64 ]; then
    MAX_RUNNING_REQUESTS=64
fi
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS

# GLM dpafp8 env levers (both honored by v0.5.19).
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$WEIGHTS"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --context-length 262144
    --mem-fraction-static 0.88
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
    --mamba-radix-cache-strategy extra_buffer_lazy
    --max-mamba-cache-size 384
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

build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
