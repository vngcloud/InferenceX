#!/usr/bin/env bash
set -eo pipefail
set -x

# Gemma-4-31B customer-service (tau2) agentic arm, MTP speculative-decoding
# variant of benchmarks/single_node/agentic/gemma4tau2_fp8_h200.sh (the spec-none
# arm, feat/gemma4-tau2-agentic-arm). Same tau2 corpus replay (private
# thangquang09/customer-service-agent-traces, mooncake_trace raw-content) and
# the same agentic-coding scenario; the only deltas vs the spec-none arm:
#   1. spec decode ON  -> --speculative-config method mtp, draft
#      google/gemma-4-31B-it-assistant, num_speculative_tokens 4 (MTP4)
#   2. image           -> vllm/vllm-openai:v0.30.0 (the image the gemma4sbamtplo
#      vLLM cells validated the mtp speculative-config syntax on; v0.25.0 of the
#      spec-none arm predates it)
#   3. env             -> VLLM_DISABLE_COMPILE_CACHE=1, NCCL_P2P_LEVEL=NVL,
#      VLLM_ATTENTION_BACKEND=FLASHINFER (identical to the gemma4sba MTP cells)
#   4. topology        -> single replica, no vllm-router: the 1xH200
#      greennode-slurm nodes have one GPU per node, so the two-replica +
#      cache_aware-router prod topology of the spec-none arm does not fit.
#      aiperf talks to vLLM directly on $PORT.
# max-model-len stays 262144 and prefix caching stays ON (vLLM default, no
# flag), matching the spec-none arm; gpu-memory-utilization 0.92,
# max-num-seqs 64, max-num-batched-tokens 16384 and
# long-prefill-token-threshold 8192 are unchanged too.
#
# Required env (harness-provided): MODEL TP CONC KV_OFFLOADING RESULT_DIR
# DURATION PORT. The ambient HF_TOKEN must be able to read the private tau2
# dataset and the gated google/gemma-4-31b-it tokenizer.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT
require_agentic_kv_offload_none

# $MODEL is the un-quantized HF id (aiperf tokenizer + wire name); the engine
# serves the RedHatAI FP8-block quant behind it, draft MTP assistant included.
WEIGHTS="RedHatAI/gemma-4-31B-it-FP8-block"
DRAFT="google/gemma-4-31B-it-assistant"
TAU2_REPO="thangquang09/customer-service-agent-traces"

# aiperf hits the single vLLM replica on $PORT; its /metrics feeds the required
# vllm: server-metrics gate.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
# DCGM exporter the 1x greennode-slurm launcher starts alongside the job; the
# launcher exports AIPERF_GPU_TELEMETRY_URL itself, so it is not set here.

# benchmark_lib.sh reassigns AIPERF_UV_CACHE_DIR to an ephemeral /tmp dir at
# source time, so every dispatch cold-downloads aiperf's deps from PyPI (~1 GB,
# painfully slow on this box). Restore the launcher's persistent mount so c2 and
# reruns hit the uv cache instead of re-downloading.
if [ -d /mnt/uv-cache ]; then
    export AIPERF_UV_CACHE_DIR=/mnt/uv-cache
fi
install_agentic_deps
nvidia-smi

# Model weights + MTP draft are public -> ambient HF_TOKEN. Idempotent: a cache
# hit is a no-op. Serve by HF id so vLLM resolves from the mounted HF_HUB_CACHE.
"$AIPERF_HF_CLI" download "$WEIGHTS"
"$AIPERF_HF_CLI" download "$DRAFT"

# The tau2 corpus is a PRIVATE repo; the ambient HF_TOKEN (the official CI token)
# must be able to read it. Resolve the snapshot dir via snapshot_download (returns
# just the path) instead of parsing `hf download` stdout, which newer CLIs decorate
# with a "✓ Downloaded / path: ..." banner.
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
# The server log MUST be $RESULT_DIR/server.log: the vLLM server-metrics adapter
# parses "GPU KV cache size: N tokens" from that exact path for
# kv_cache.gpu_total_tokens.
SERVER_LOG="$RESULT_DIR/server.log"

# MTP4 engine args: identical to the spec-none arm except the speculative
# config; env identical to the gemma4sba MTP cells. Prefix caching stays ON
# (vLLM default; the spec-none arm and prod both run it on).
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER
cmd=(
    vllm serve "$WEIGHTS"
    --served-model-name "$MODEL" gemma-4-31b-it
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size 1
    --gpu-memory-utilization 0.92
    --max-model-len 262144
    --max-num-seqs 64
    --max-num-batched-tokens 16384
    --enable-chunked-prefill
    --long-prefill-token-threshold 8192
    --speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": 4}"
    --enable-auto-tool-choice
    --tool-call-parser gemma4
    --reasoning-parser gemma4
)
printf '%q ' "${cmd[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${cmd[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
