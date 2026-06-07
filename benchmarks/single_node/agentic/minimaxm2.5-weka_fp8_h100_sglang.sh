#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX-MVP (cc-traces-weka) replay benchmark for MiniMax-M2.5 FP8 on SGLang.
#
# Unlike the Mode 1 mooncake_trace launchers, this drives the official
# SemiAnalysis `inferencex-agentx-mvp` scenario over the upstream
# semianalysisai/cc-traces-weka-no-subagents-051226 corpus (949 traces),
# resolved + invoked entirely through aiperf (resolve_trace_source +
# build_replay_cmd in benchmark_lib.sh). The only load knob is --concurrency;
# the scenario locks cache-bust, inter-turn-delay-cap, ignore_eos, etc.
#
# Serving uses the official MiniMax-M2.5 SGLang launch args (TP/EP, minimax
# tool-call + reasoning parsers, page-size/chunked-prefill/hicache tuning).
#
# Required env vars (provided by the agentic-coding workflow path):
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
EP_SIZE=${EP_SIZE:-1}
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"

# benchmark-tmpl passes MAX_MODEL_LEN='0' for agentic-coding; fall back to the
# official MiniMax-M2.5 SGLang context length. Tune per dataset if needed.
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=147456
fi

# AgentX-MVP serves a fixed weka corpus; CPU/SSD KV offload is not wired for
# the SGLang launch path. Only "none" is supported here.
if [ "$OFFLOADING" != "none" ]; then
    echo "Error: OFFLOADING='$OFFLOADING' not supported for the SGLang AgentX-MVP launcher (expected: none)" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

echo "Starting SGLang server..."
# --enable-metrics exposes the Prometheus /metrics endpoint on $PORT so aiperf
# (server-metrics scrape on by default, auto-discovers localhost:$PORT/metrics)
# can capture sglang:cache_hit_rate / sglang:cached_tokens_total — the weka
# corpus's whole point is prefix-cache reuse, which is otherwise invisible.
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --ep "$EP_SIZE" \
  --context-length "$MAX_MODEL_LEN" \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --chunked-prefill-size 16384 \
  --hicache-size 1200 \
  --enable-metrics \
  --trust-remote-code > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
