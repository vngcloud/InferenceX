#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) replay smoke for Qwen/Qwen3-4B-Instruct-2507 (bf16) on vLLM (H200).
#
# Same SemiAnalysis `inferencex-agentx-mvp` scenario and weka corpus
# (semianalysisai/cc-traces-weka-no-subagents-051226, 949 traces) as the MiniMax
# SGLang launcher — resolved + invoked entirely through aiperf
# (resolve_trace_source + build_replay_cmd in benchmark_lib.sh). The only load
# knob is --concurrency; the scenario locks cache-bust, inter-turn-delay-cap,
# ignore_eos, etc. build_replay_cmd passes no --warmup-* so there is no warmup leg.
#
# ONE deliberate difference from the MiniMax SGLang launcher: aiperf runs in an
# ISOLATED venv. SGLang installs aiperf's deps into the system python and is
# fine (it tolerates the upgraded anyio/starlette/fastapi). vLLM v0.21.0 is not:
# install_agentic_deps upgrades that web stack, and vLLM's API server imports
# the new anyio/starlette lazily while serving (-> `_IncludedRouter has no
# attribute 'path'` on /health, then `cannot import name 'TaskHandle' from
# anyio._core._tasks` on the first request, aborting the run). Reordering does
# NOT help — the bad import happens at request time, after the server is up. So
# we install aiperf into a clean venv (no system-site-packages); vLLM keeps the
# image's untouched system python, aiperf gets its own, and they share only the
# localhost socket. The venv lives in /tmp (no new dirs in /workspace).
#
# Required env vars (provided by the agentic-coding workflow path):
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"

# benchmark-tmpl passes MAX_MODEL_LEN='0' for agentic-coding; fall back to 131072
# — the value the proven qwen3-4b-2507 mooncake configs use on this exact
# model/image/runner. weka traces longer than this just become context-overflow
# turns, which the agentx-mvp scenario drops from the failure tally.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

# AgentX-MVP serves a fixed weka corpus; CPU/SSD KV offload is not wired for the
# vLLM launch path here. Only "none" is supported.
if [ "$OFFLOADING" != "none" ]; then
    echo "Error: OFFLOADING='$OFFLOADING' not supported for the vLLM AgentX-MVP launcher (expected: none)" >&2
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

echo "Starting vLLM server..."
# vLLM exposes the Prometheus /metrics endpoint on $PORT by default; aiperf's
# server-metrics scrape (on by default) auto-discovers localhost:$PORT/metrics
# and captures prefix cache-hit stats — the whole point of the weka corpus.
# --enable-prefix-caching is the V1 default but set explicitly to make intent clear.
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Isolate aiperf in its own venv, then resolve traces + install deps -----
# Clean venv (NO --system-site-packages): aiperf's anyio/starlette/fastapi
# upgrades land here only and never reach the running vLLM server's system
# python. See header. Lives in /tmp so we add no dirs under /workspace.
AIPERF_VENV="${TMPDIR:-/tmp}/aiperf-venv"
python3 -m venv "$AIPERF_VENV"
# shellcheck disable=SC1091
source "$AIPERF_VENV/bin/activate"
python3 -m pip install -q --upgrade pip

resolve_trace_source      # installs the hf CLI into the venv
install_agentic_deps      # installs aiperf + deps into the venv

# ---- Run benchmark ----------------------------------------------------------
# Smoke subset: 64 of the 949 weka traces is plenty to exercise the path +
# measure prefix cache-hit at conc 2/4 over a 90s window, and loads much faster.
# (Host RAM was never the limit — earlier failures were the dep clash above.)
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
