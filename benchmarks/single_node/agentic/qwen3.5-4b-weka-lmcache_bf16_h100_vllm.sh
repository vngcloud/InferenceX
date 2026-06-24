#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) smoke for Qwen/Qwen3.5-4B (bf16) on vLLM
# with LMCache CPU KV-offload enabled.
#
# LMCache 0.4.5 is pre-bundled in vllm/vllm-openai:v0.21.0 — no extra install.
# Two-tier cache: vLLM GPU HBM prefix cache (--enable-prefix-caching) feeds into
# LMCache CPU DRAM (LMCacheConnectorV1) when GPU blocks are evicted.
# Hit metrics: vllm:external_prefix_cache_hits_total / *_queries_total (GPU-side view)
# plus lmcache:num_hit_tokens_total / retrieve_hit_rate on the internal API server (:7001).
#
# ONE deliberate difference from the MiniMax SGLang launcher: aiperf runs in an
# ISOLATED venv. vLLM v0.21.0's API server imports anyio/starlette lazily while
# serving; install_agentic_deps upgrades that web stack and triggers
# `_IncludedRouter has no attribute 'path'` on /health then
# `cannot import name 'TaskHandle' from anyio._core._tasks` on the first request.
# Fix: clean venv (no --system-site-packages) so vLLM keeps the image's untouched
# system python; they share only the localhost socket. Venv lives in /tmp.
#
# Required env vars (provided by the agentic-coding workflow path):
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"

# benchmark-tmpl passes MAX_MODEL_LEN='0' for agentic-coding; fall back to 131072.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

# LMCache handles CPU KV-offload transparently via the KV-transfer connector;
# the standard vLLM offloading knob is not used here.
if [ "$OFFLOADING" != "none" ]; then
    echo "Error: OFFLOADING='$OFFLOADING' not supported for this launcher (expected: none)" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Server config (start BEFORE installing aiperf; see header) -------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1

# LMCache env — must be set before vllm serve forks the worker process.
export LMCACHE_CONFIG_FILE="/workspace/benchmarks/lmcache_cpu.yaml"
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0

echo "Starting vLLM server with LMCache CPU KV-offload..."
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --no-disable-hybrid-kv-cache-manager \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Isolate aiperf in its own venv, then resolve traces + install deps -----
# Clean venv (NO --system-site-packages): see header. Lives in /tmp.
AIPERF_VENV="${TMPDIR:-/tmp}/aiperf-venv"
python3 -m venv "$AIPERF_VENV"
# shellcheck disable=SC1091
source "$AIPERF_VENV/bin/activate"
python3 -m pip install -q --upgrade pip

resolve_trace_source      # installs the hf CLI into the venv
install_agentic_deps      # installs aiperf + deps into the venv

# ---- Run benchmark ----------------------------------------------------------
# Smoke subset: 64 of the 949 weka traces.
export WEKA_NUM_DATASET_ENTRIES="${WEKA_NUM_DATASET_ENTRIES:-64}"

build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
