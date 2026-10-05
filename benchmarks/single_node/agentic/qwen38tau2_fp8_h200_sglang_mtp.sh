#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-27B customer-service (tau2) agentic arm: prod-exact serving plus the
# GLM-5.3-Flash dpafp8 levers that transfer to this mamba-hybrid family.
#
# Replays the private tau2 corpus (thangquang09/customer-service-agent-traces,
# mooncake_trace raw content: messages + tools verbatim) against the prod
# Qwen3.8-27B-FP8 deployment's serving shape (deployment/qwen38-27b-fp8 on the
# GLM-5.3-Flash k8s cluster: sglang v0.5.19, 1x H200, TP1, FP8 weights, EAGLE
# off the built-in qwen3_5 MTP head 3/1/4, --enable-linear-replayssm-spec,
# --schedule-policy dfs-weight, HiCache 24), plus the levers validated on the
# GLM-5.3-Flash dpafp8 Current-Best ladder (docs/handoffs/2026-09-28-glm53flash
# -dpa-fp8-agentx-ladder.md, slurm job 939):
#   - --max-mamba-cache-size 384 pinned: the default --mamba-full-memory-ratio
#     under-sizes the hybrid state pool ~6x and crashed GLM at CCU 32; 384
#     slots at dp_size 1 = state cap 128 running requests (S=3 with
#     extra_buffer_lazy + skip-decode-lock) vs max-running 64 -> 2x headroom
#     for radix retention. Re-measure state_bytes_per_slot / kv_bytes_per_token
#     from this boot log before tuning further (sglang compute-mamba-ratio
#     skill: r* = (S + D) * token_equiv * dcp_size / L, D=0 with ReplaySSM).
#   - --mem-fraction-static 0.88 (GLM freed 35 GB of idle pool budget at 0.75)
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
# Differs from a cc-traces sglang arm in only two places: the dataset source
# (local mooncake_trace file instead of a Weka public loader) and the wire
# name $MODEL staying the un-quantized Qwen/Qwen3.8-27B (aiperf tokenizer)
# while the FP8 checkpoint is served. Reuses the agentic-coding scenario and
# build_replay_cmd unchanged, so the artifact is schema-identical to a
# cc-traces arm and joins the same analysis.
#
# Required env (harness-provided): MODEL TP CONC KV_OFFLOADING RESULT_DIR
# DURATION PORT EVAL_ONLY. The ambient HF_TOKEN must be able to read the
# private tau2 dataset.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT EVAL_ONLY
require_agentic_kv_offload_backend hicache

# Prod serves the FP8 checkpoint; $MODEL is the un-quantized HF id, used only
# as the wire/served name and the aiperf tokenizer.
WEIGHTS="Qwen/Qwen3.8-27B-FP8"
TAU2_REPO="thangquang09/customer-service-agent-traces"

# Single server, no router (TP1 mirrors the prod deployment; the GLM router
# was a dp-attention multi-worker lever that does not apply at dp1). aiperf
# hits the server on $PORT directly; the sglang: prefix selects the sglang
# branch of the server-metrics schema.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
# DCGM exporter the hardware-hcm launcher starts alongside the job.
export AIPERF_GPU_TELEMETRY_URL="http://localhost:9400/metrics"

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

# The tau2 corpus is a PRIVATE repo; the ambient HF_TOKEN must be able to read
# it. Resolve the snapshot dir via snapshot_download (returns just the path)
# instead of parsing `hf download` stdout, which newer CLIs decorate with a
# "Downloaded / path: ..." banner.
TAU2_DIR=$("$AIPERF_PYTHON" -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], repo_type='dataset'))" \
    "$TAU2_REPO")
mapfile -t TAU2_FILES < <(find "$TAU2_DIR" -maxdepth 2 -name '*.jsonl' | sort)
if [ "${#TAU2_FILES[@]}" -ne 1 ]; then
    echo "Error: expected exactly one .jsonl in $TAU2_DIR, found ${#TAU2_FILES[@]}: ${TAU2_FILES[*]}" >&2
    exit 1
fi
TAU2_FILE="${TAU2_FILES[0]}"
echo "tau2 mooncake_trace input: $TAU2_FILE"

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

# Replay input override: swap the Weka public-dataset default for the local
# mooncake_trace file. build_replay_cmd appends $TRACE_SOURCE_FLAG verbatim,
# so set it directly instead of calling the Weka-only resolve_trace_source.
export TRACE_SOURCE_FLAG="--custom-dataset-type mooncake_trace --input-file $TAU2_FILE"
# The local --input-file corpus is unpinned; runs are stamped
# submission_valid: false, which is expected and acceptable.
export AIPERF_UNSAFE_OVERRIDE=true
# Dual stop: 20-minute cap OR one full pass over the 16,798-turn pool,
# whichever fires first. Deliberate recipe-specific override, not a discarded
# caller input: the matrix generator hardcodes duration=3600 for every
# agentic-coding row (infx/matrix/generate.py DEFAULT_AGENTIC_DURATION_
# SECONDS) with no per-arm field, and tau2's turn pool is sized for ~1200s at
# CCU 8+. DURATION feeds --benchmark-duration inside build_replay_cmd.
export DURATION=1200

build_replay_cmd "$RESULT_DIR"
# Override the Weka-shaped entry cap (393, with-subagents corpus) for the full
# tau2 corpus (1,576 sessions; the loader treats it as min(cap, available)),
# and add the turn-count half of the dual stop; --request-count and
# --benchmark-duration coexist, stopping at whichever fires first.
REPLAY_CMD="${REPLAY_CMD/--num-dataset-entries 393/--num-dataset-entries 1576}"
if [[ "$REPLAY_CMD" != *"--num-dataset-entries 1576"* ]]; then
    echo "Error: build_replay_cmd no longer emits '--num-dataset-entries 393'; the tau2 entry-cap override missed. Refresh it before running." >&2
    exit 1
fi
REPLAY_CMD+=" --request-count 16798"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
