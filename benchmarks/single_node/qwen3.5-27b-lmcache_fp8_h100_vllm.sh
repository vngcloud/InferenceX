#!/usr/bin/env bash
set -euo pipefail
#
# Agentic-replay launcher: Qwen/Qwen3.5-27B-FP8 (fp8, TP=1) on vLLM with LMCache CPU
# KV-offload via the MP connector (standalone ZMQ server).
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
# Unified block size for Qwen3.5-27B: N=784.
# Confirmed from first-run crash: "block_size (784) must be <= max_num_batched_tokens".
# LMCache MP connector enforces N <= max_num_batched_tokens < 2*N; use 2*N-1 for max throughput.
#
# Reference implementation: benchmarks/single_node/agentic/qwen3.5-27b-weka-lmcache_fp8_h100_vllm.sh
# (same model + connector, weka/agentx path). This script adapts the same LMCache setup
# for the agentic-replay (mooncake_trace / AIPerf) path.

source "$(dirname "$0")/../benchmark_lib.sh"

export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"

check_env_vars \
    MODEL \
    TP \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-27B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# An optional "#N" suffix on the input-file path replays only the first N records.
trace_limit=""
if [[ "$INPUT_FILE" == *"#"* ]]; then
    trace_limit="${INPUT_FILE##*#}"
    INPUT_FILE="${INPUT_FILE%#*}"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: trace input file not found: $INPUT_FILE (cwd=$(pwd))" >&2
    exit 1
fi

if [[ -n "$trace_limit" ]]; then
    head -n "$trace_limit" "$INPUT_FILE" > /workspace/_trace_subset.jsonl
    INPUT_FILE=/workspace/_trace_subset.jsonl
    echo "Subset trace to first $trace_limit records -> $INPUT_FILE"
fi

if [[ "${STRIP_TRACE_DELAYS:-}" == "true" || "${STRIP_TRACE_DELAYS:-}" == "1" ]]; then
    python3 -c '
import json, sys
with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("delay", None)
        fout.write(json.dumps(rec) + "\n")
' "$INPUT_FILE" /workspace/_trace_nodelay.jsonl
    INPUT_FILE=/workspace/_trace_nodelay.jsonl
    echo "Stripped per-turn delays for capacity sweep -> $INPUT_FILE"
fi

SERVER_LOG=/workspace/server.log
LMC_LOG=/workspace/lmcache_server.log
PORT=${PORT:-8888}

start_gpu_monitor

# LMCache CPU DRAM budget — DO NOT use $TOTAL_CPU_DRAM_GB (defaults to 600 GB in
# benchmark-tmpl.yml, which would OOM-kill the lmcache server on GreenNode H100).
LMCACHE_CPU_DRAM_GB="${LMCACHE_CPU_DRAM_GB:-5}"

# On any non-zero exit, tail both server logs so CI captures the crash cause.
_dump_logs_on_failure() {
  local rc=$?
  [ $rc -eq 0 ] && return
  echo "=== EXIT $rc — last 80 lines of server.log ==="
  tail -80 "$SERVER_LOG" 2>/dev/null || echo "(no server.log)"
  echo "=== last 80 lines of lmcache_server.log ==="
  tail -80 "$LMC_LOG" 2>/dev/null || echo "(no lmcache_server.log)"
}
trap _dump_logs_on_failure EXIT

# Upgrade bundled lmcache 0.4.6 → 0.5.0: LMCacheMPConnector must be SupportsHMA
# so vLLM keeps the hybrid KV manager on for Qwen3.5-27B.
pip install --no-cache-dir "lmcache==0.5.0"

# Unified block size for Qwen3.5-27B (N=784, confirmed from crash log).
# LMCache MP connector enforces: N <= max_num_batched_tokens < 2*N.
# Use 2*N-1 to maximise tokens-per-prefill-step within the constraint.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-784}"

# ---- Start standalone LMCache MP server (ZMQ :5555) -------------------------
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

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--served-model-name "$SERVED_MODEL_NAME" \
--tensor-parallel-size "$TP" \
--dtype bfloat16 \
--gpu-memory-utilization 0.90 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--mamba-cache-mode align \
--enable-prefix-caching \
--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-$((2 * LMCACHE_CHUNK_SIZE - 1))}" \
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

STOP_ARGS=()
DURATION_OVERRIDE="${BENCHMARK_DURATION:-${DURATION:-}}"
if [[ -n "$DURATION_OVERRIDE" ]]; then
    STOP_ARGS+=(--benchmark-duration "$DURATION_OVERRIDE")
    echo "Duration-based replay of $INPUT_FILE: benchmark-duration=${DURATION_OVERRIDE}s at concurrency $CONC"
else
    if [[ -z "${REQUEST_COUNT:-}" ]]; then
        REQUEST_COUNT=$(grep -c . "$INPUT_FILE")
    fi
    STOP_ARGS+=(--request-count "$REQUEST_COUNT")
    echo "Replaying trace $INPUT_FILE: request-count=$REQUEST_COUNT at concurrency $CONC"
fi

REPLAY_ARGS=()
if [[ "${NO_FIXED_SCHEDULE:-true}" == "true" || "${NO_FIXED_SCHEDULE:-true}" == "1" ]]; then
    REPLAY_ARGS+=(--no-fixed-schedule)
fi
REPLAY_ARGS+=(--inter-turn-delay-cap-seconds "${INTER_TURN_DELAY_CAP_SECONDS:-60}")
REPLAY_ARGS+=(--dataset-sampling-strategy "${DATASET_SAMPLING_STRATEGY:-shuffle}")
REPLAY_ARGS+=(--warmup-request-count "${WARMUP_REQUEST_COUNT:-20}")
REPLAY_ARGS+=(--workers-max "${WORKERS_MAX:-200}")
REPLAY_ARGS+=(--benchmark-grace-period "${BENCHMARK_GRACE_PERIOD:-120}")
REPLAY_ARGS+=(--extra-inputs "ignore_eos:${IGNORE_EOS:-true}")
REPLAY_ARGS+=(--extra-inputs "temperature:${TEMPERATURE:-0}")
if [[ -n "${NUM_WARMUP_SESSIONS:-}" ]]; then
    REPLAY_ARGS+=(--num-warmup-sessions "$NUM_WARMUP_SESSIONS")
fi
if [[ -n "${GOODPUT:-}" ]]; then
    REPLAY_ARGS+=(--goodput "$GOODPUT")
fi
if [[ -n "${TOKENIZER:-}" ]]; then
    REPLAY_ARGS+=(--tokenizer "$TOKENIZER")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    "${STOP_ARGS[@]}" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    "${REPLAY_ARGS[@]}"

stop_gpu_monitor
set +x
