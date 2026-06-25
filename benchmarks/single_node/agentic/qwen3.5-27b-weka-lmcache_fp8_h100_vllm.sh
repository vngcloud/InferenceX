#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) smoke for Qwen/Qwen3.5-27B-FP8 on vLLM
# with LMCache CPU KV-offload via the MP connector (standalone ZMQ server).
#
# Stack: vllm/vllm-openai:v0.23.0 + lmcache 0.5.0 (installed at runtime;
# v0.23.0 bundles lmcache 0.4.6 which is NOT a SupportsHMA subclass and
# triggers the hybrid-manager crash).
#
# Hybrid-attention requirement: Qwen3.5-27B interleaves linear_attention and
# full_attention layers. The in-process LMCacheConnectorV1 assumes one unified
# KV shape and crashes with "ValueError: failed to convert the KV cache specs
# to one unified type". Fix: --mamba-cache-mode align (vLLM 0.23.0+) equalizes
# block sizes; lmcache 0.5.0's LMCacheMPConnector is SupportsHMA.
#
# FP8 model: Qwen/Qwen3.5-27B-FP8 ships native fp8 weights with dynamic
# activation scheme. vLLM auto-detects quantization from quantization_config
# — do NOT pass --quantization. --dtype bfloat16 sets the compute/activation
# dtype around the fp8 weights.
#
# Deployment shape: lmcache server runs as a separate process in the SAME
# container (ZMQ tcp://localhost:5555). --ipc=host is not needed.
#
# Unified block size for Qwen3.5-27B: N=784.
# Confirmed from first-run crash log:
#   "AssertionError: In Mamba cache align mode, block_size (784) must be <=
#    max_num_batched_tokens (256)"
# Both --chunk-size (lmcache server) and --max-num-batched-tokens (vLLM) must
# equal N=784.
#
# Reference implementation for this connector pattern: qwen3.5-4b-weka-lmcache_bf16_h100_vllm.sh
# (Qwen3.5-4B, N=528). Copy this script for other hybrid 27B-class models and
# update LMCACHE_CHUNK_SIZE after running the discovery command.
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

# LMCache CPU DRAM budget for the MP server.
# DO NOT use $TOTAL_CPU_DRAM_GB here — that variable is for vLLM's native CPU
# KV-offloading and defaults to 600 GB (benchmark-tmpl.yml), which exceeds
# GreenNode H100 available RAM and causes the lmcache server to be OOM-killed.
LMCACHE_CPU_DRAM_GB="${LMCACHE_CPU_DRAM_GB:-10}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Server config (start BEFORE installing aiperf; see header) -------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1

# On any non-zero exit, tail both server logs so CI captures the crash cause.
_dump_logs_on_failure() {
  local rc=$?
  [ $rc -eq 0 ] && return
  echo "=== EXIT $rc — last 80 lines of server.log ==="
  tail -80 "$SERVER_LOG" 2>/dev/null || echo "(no server.log)"
  echo "=== last 80 lines of lmcache_server.log ==="
  tail -80 "${RESULT_DIR}/lmcache_server.log" 2>/dev/null || echo "(no lmcache_server.log)"
}
trap _dump_logs_on_failure EXIT

# Upgrade lmcache from bundled 0.4.6 to 0.5.0 — required so LMCacheMPConnector
# is a SupportsHMA subclass and vLLM keeps the hybrid KV manager on.
# Must happen before lmcache server or vllm starts.
pip install --no-cache-dir "lmcache==0.5.0"

LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-784}"

# ---- Start standalone LMCache MP server (ZMQ :5555) -------------------------
LMC_LOG="$RESULT_DIR/lmcache_server.log"
lmcache server \
  --chunk-size "$LMCACHE_CHUNK_SIZE" \
  --l1-size-gb "$LMCACHE_CPU_DRAM_GB" \
  --eviction-policy LRU \
  --port 5555 \
  --http-host 0.0.0.0 --http-port 8080 > "$LMC_LOG" 2>&1 &
LMC_PID=$!
echo "LMCache server PID: $LMC_PID"

LMC_READY=0
for i in $(seq 1 40); do
  if grep -qiE "ZMQ cache server is running|listening|started|serving|bound|fired|ready|MessageQueueServer|MPCacheServer" \
      "$LMC_LOG" 2>/dev/null; then
    LMC_READY=1
    break
  fi
  kill -0 "$LMC_PID" 2>/dev/null || { echo "LMCache server died:"; cat "$LMC_LOG"; exit 1; }
  sleep 1
done

if [ "$LMC_READY" -eq 0 ]; then
  echo "ERROR: LMCache MP server did not print a ready message within 40s." >&2
  echo "lmcache_server.log:" >&2; cat "$LMC_LOG" >&2
  kill "$LMC_PID" 2>/dev/null || true
  exit 1
fi

echo "LMCache server ready."

export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0

echo "Starting vLLM server with LMCache MP connector..."
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_MODEL_LEN" \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --max-num-batched-tokens "$LMCACHE_CHUNK_SIZE" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Isolate aiperf in its own venv, then resolve traces + install deps -----
# Clean venv (NO --system-site-packages): vLLM 0.23.0 lazily imports
# anyio/starlette; install_agentic_deps would upgrade them and break the live
# server. Clean venv keeps aiperf on its own python.
AIPERF_VENV="${TMPDIR:-/tmp}/aiperf-venv"
python3 -m venv "$AIPERF_VENV"
# shellcheck disable=SC1091
source "$AIPERF_VENV/bin/activate"
python3 -m pip install -q --upgrade pip

resolve_trace_source      # installs the hf CLI into the venv
install_agentic_deps      # installs aiperf + deps into the venv

# ---- Run benchmark ----------------------------------------------------------
# For smoke runs, cap at 64 traces via: export WEKA_NUM_DATASET_ENTRIES=64
# Standard capacity (900s) uses the full 949-trace corpus (no cap set here).
if [ -n "${WEKA_NUM_DATASET_ENTRIES:-}" ]; then
  export WEKA_NUM_DATASET_ENTRIES
fi

build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

scrape_lmcache_server_metrics "$RESULT_DIR"
write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
