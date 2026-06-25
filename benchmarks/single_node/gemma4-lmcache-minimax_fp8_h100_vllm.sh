#!/usr/bin/env bash
set -euo pipefail
#
# Capacity agentic-replay launcher: RedHatAI/gemma-4-31B-it-FP8-dynamic (fp8, TP=1) on vLLM
# with LMCache CPU KV-offload via the MP connector (standalone ZMQ server).
#
# Differs from gemma4-lmcache_fp8_h100_vllm.sh (smoke):
#   GPU_MEMORY_UTILIZATION=0.65  — lower to leave DRAM headroom for the larger CPU KV budget.
#   LMCACHE_CPU_DRAM_GB=20       — larger CPU KV budget for capacity sweep at conc [2,4,8].
#
# Stack: vllm/vllm-openai:v0.23.0 + lmcache 0.5.0 (installed at runtime;
# v0.23.0 bundles lmcache 0.4.6 which is NOT a SupportsHMA subclass and
# triggers the hybrid-manager crash for models with heterogeneous KV specs).
#
# Hybrid-attention note: Gemma 4 31B interleaves local (sliding window) and global
# attention layers. Unlike Qwen3.5 which mixes Mamba linear_attention + full_attention
# (true SupportsHMA requirement), Gemma 4 uses only attention layers. If vLLM reports
# "failed to convert the KV cache specs to one unified type", the MP connector path
# here is correct. If both layer types share the same KV shape (likely), the in-process
# V1 connector would also work — but the MP connector is safe for both cases.
#
# LMCACHE_CHUNK_SIZE discovery (run once before production use):
#   vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic \
#     --mamba-cache-mode align --enable-prefix-caching --max-model-len 4096 \
#     2>&1 | grep "Setting attention block size"
#   If output is absent (pure-attention model), set LMCACHE_CHUNK_SIZE=256 (default here).
#   If output shows "Setting attention block size to N tokens", set it to N and match
#   the lmcache server --chunk-size to N.
#
# FP8 model: RedHatAI/gemma-4-31B-it-FP8-dynamic uses compressed-tensors quantization.
# vLLM auto-detects it from quantization_config.json — do NOT pass --quantization.
# --dtype bfloat16 sets the compute/activation dtype around the fp8 weights.
#
# Deployment shape: lmcache server runs as a separate process in the SAME container
# (ZMQ tcp://localhost:5555). --ipc=host is not needed.
#
# Reference: benchmarks/single_node/gemma4-lmcache_fp8_h100_vllm.sh (smoke variant).

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

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-RedHatAI/gemma-4-31B-it-FP8-dynamic}"
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

# Capacity-run defaults: lower GPU util leaves DRAM headroom for the larger CPU KV budget.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
# LMCache CPU DRAM budget — DO NOT use $TOTAL_CPU_DRAM_GB (defaults to 600 GB in
# benchmark-tmpl.yml, which would OOM-kill the lmcache server on GreenNode H100).
LMCACHE_CPU_DRAM_GB="${LMCACHE_CPU_DRAM_GB:-20}"

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

# Upgrade bundled lmcache 0.4.6 → 0.5.0: LMCacheMPConnector must be SupportsHMA.
# Must happen before lmcache server or vllm starts.
pip install --no-cache-dir "lmcache==0.5.0"

# Chunk size for the lmcache server. For Gemma 4 (pure-attention hybrid),
# 256 is the lmcache_cpu.yaml default. Run the discovery command in the header
# if you observe cache misalignment or a "block_size must be <= max_num_batched_tokens"
# crash, and set LMCACHE_CHUNK_SIZE to the discovered N.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"

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
--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--enable-prefix-caching \
--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
set +x

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
