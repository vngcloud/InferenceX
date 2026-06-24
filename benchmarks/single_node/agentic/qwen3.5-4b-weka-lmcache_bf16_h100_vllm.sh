#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) smoke for Qwen/Qwen3.5-4B (bf16) on vLLM
# with LMCache CPU KV-offload via the MP connector (standalone ZMQ server).
#
# Stack: vllm/vllm-openai:v0.23.0 + lmcache 0.5.0 (installed at runtime;
# v0.23.0 bundles lmcache 0.4.6 which is NOT a SupportsHMA subclass and
# triggers the same hybrid-manager crash as 0.4.5).
#
# Hybrid-attention requirement: Qwen3.5-4B interleaves linear_attention and
# full_attention layers. The old in-process LMCacheConnectorV1 assumes one
# unified KV shape, so vLLM disables its hybrid KV manager and aborts with
# "ValueError: failed to convert the KV cache specs to one unified type".
# Fix: --mamba-cache-mode align (vLLM 0.23.0+) equalizes block sizes so all
# layers share one block footprint; lmcache 0.5.0's LMCacheMPConnector is
# SupportsHMA, so vLLM keeps the hybrid KV manager on.
#
# Deployment shape: lmcache server runs as a separate process in the SAME
# container (ZMQ tcp://localhost:5555 + CUDA IPC). Same-container = shared
# IPC namespace by default; --ipc=host is not needed here.
#
# Unified block size for Qwen3.5-4B: N=528.
# Discovered via: vllm serve <model> --mamba-cache-mode align \
#   --enable-prefix-caching 2>&1 | grep "Setting attention block size"
# Both --chunk-size (lmcache server) and --max-num-batched-tokens (vLLM)
# must equal N.
#
# Hit metrics: vllm:external_prefix_cache_{hits,queries}_total on the vLLM
# engine port. The :7001 internal-API-server (in-process V1 path only) is
# absent here. process_agentic_result.py reads external_prefix_cache_* which
# works on both connector stacks — no result-collection changes needed.
#
# aiperf isolation: same isolated venv as the Qwen3-8B vLLM launcher — vLLM
# 0.23.0 lazily imports anyio/starlette; install_agentic_deps would upgrade
# them and break the live server. Clean venv keeps aiperf on its own python.
#
# Template for other hybrid models: copy this script, update LMCACHE_CHUNK_SIZE
# to the target model's unified block size (grep "Setting attention block size"
# from a one-time --mamba-cache-mode align boot), and adjust MAX_MODEL_LEN.
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
# GreenNode H100 available RAM (~112-200 GB) and causes the lmcache server to
# be OOM-killed silently before vLLM's EngineCore connects.
LMCACHE_CPU_DRAM_GB="${LMCACHE_CPU_DRAM_GB:-20}"

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
# set +x to suppress pip's verbose command echo from flooding the CI step log.
{ set +x; } 2>/dev/null
pip install --no-cache-dir "lmcache==0.5.0"
set -x

# Unified block size for Qwen3.5-4B (from --mamba-cache-mode align startup log:
# "Setting attention block size to 528 tokens"). Both the lmcache server
# --chunk-size and vLLM --max-num-batched-tokens must equal this value.
# Override via env var when adapting this script for another hybrid model.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-528}"

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

# Wait for the ZMQ listener. lmcache 0.5.0 prints "ZMQ cache server is running"
# on the happy path. The broader pattern list covers other version variants.
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
  --gpu-memory-utilization 0.85 \
  --max-model-len "$MAX_MODEL_LEN" \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --max-num-batched-tokens "$LMCACHE_CHUNK_SIZE" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Use a quiet poll instead of wait_for_server_ready (which tails the full server log
# to stdout). vLLM 0.23.0 + mamba-cache-mode align produces thousands of startup log
# lines; streaming them through tail -f floods the GitHub Actions step log past the
# ~50 MB per-step limit, causing the runner to disconnect mid-run.
# _dump_logs_on_failure trap above still captures the last 80 lines on any failure.
echo "Waiting for vLLM to become healthy (port $PORT)..."
for _i in $(seq 1 120); do
  if curl --output /dev/null --silent --fail "http://0.0.0.0:$PORT/health"; then
    echo "vLLM ready after $((_i * 5))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM died before becoming healthy. Last 80 lines of server.log:" >&2
    tail -80 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$_i" -eq 120 ]; then
    echo "Timeout (600s) waiting for vLLM. Last 80 lines of server.log:" >&2
    tail -80 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done

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
