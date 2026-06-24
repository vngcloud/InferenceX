#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) smoke for Qwen/Qwen3.5-4B (bf16) on SGLang
# with LMCache CPU KV-offload enabled.
#
# LMCache 0.4.5 is installed at runtime (not bundled in the SGLang image).
# Pin is mandatory: 0.4.6+ adds a positional `config_file` arg to
# LMCacheLayerwiseConnector.__init__() that SGLang 0.5.12 never passes
# → immediate TypeError crash on server startup.
#
# deps (resolve_trace_source + install_agentic_deps) are installed globally
# BEFORE the server starts (SGLang tolerates the anyio/starlette upgrade;
# vLLM does not — this is the key structural difference from the vLLM path).
#
# Hit metrics: server_lmcache_hit_rate derived from SGLang's native
# sglang:cached_tokens_total / sglang:prompt_tokens_total counters (not from
# LMCache-native Prometheus — the layerwise connector doesn't update those).
# --enable-metrics is required; without it /metrics is absent and aiperf
# drops all server-metrics scrapes silently.
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

# LMCache handles CPU KV-offload via SGLang's layerwise connector;
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

# ---- Install LMCache + resolve traces + install aiperf deps -----------------
# Global install (no isolated venv): SGLang's API server does NOT lazy-import
# the web stack after startup, so upgrading anyio/starlette here is safe.
pip install --break-system-packages "lmcache==0.4.5"

resolve_trace_source
install_agentic_deps

# ---- Patch aiperf server-metrics NaN filter (runtime) -----------------------
# SGLang emits sglang:fwd_occupancy=NaN (uninitialized gauge); orjson encodes
# NaN as null, failing ServerMetricsRecordMessage and silently dropping
# cache_hit_rate / cached_tokens_total. Apply the idempotent one-line patch.
NONFINITE_PATCH="$(dirname "$0")/patches/aiperf-skip-nonfinite-server-metrics.patch"
NONFINITE_TARGET="$AIPERF_DIR/src/aiperf/server_metrics/data_collector.py"
if grep -q "not math.isfinite" "$NONFINITE_TARGET" 2>/dev/null; then
    echo "aiperf nonfinite-metrics fix already present; skipping patch"
elif [ -f "$NONFINITE_PATCH" ]; then
    if git -C "$AIPERF_DIR" apply "$NONFINITE_PATCH" 2>/dev/null \
        || patch -p1 -d "$AIPERF_DIR" < "$NONFINITE_PATCH"; then
        echo "Applied aiperf nonfinite-metrics patch"
    else
        echo "WARNING: failed to apply aiperf nonfinite-metrics patch; server cache-hit metrics will be empty" >&2
    fi
else
    echo "WARNING: aiperf nonfinite-metrics patch not found at $NONFINITE_PATCH" >&2
fi

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

# LMCache env — must be set before sglang.launch_server starts.
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE="/workspace/benchmarks/lmcache_cpu.yaml"
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0

echo "Starting SGLang server with LMCache CPU KV-offload..."
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --context-length "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --enable-lmcache \
  --enable-metrics \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
