#!/usr/bin/env bash
# Qwen3.5-27B BF16 on H100 via vLLM + LMCache — agentic trace replay.
#
# Uses LMCacheMPConnector (required for Qwen3.5-27B's GDN/Mamba hybrid layers).
# A separate lmcache server process runs on port 8080; its Prometheus metrics
# expose lmcache_mp_lookup_* which are scraped via --server-metrics-url and
# extracted by process_agentic_result.py as lmcache_hit_rate.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR
#
# Counterpart config keys:
#   qwen3.5-27b-bf16-h100-1x-vllm-lmcache-8k1k
#   qwen3.5-27b-bf16-h100-1x-vllm-lmcache-8k1k-gn00

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
LMCACHE_LOG=/workspace/lmcache_server.log
PORT=${PORT:-8888}
if [[ -z "${MAX_MODEL_LEN:-}" ]] || [[ "$MAX_MODEL_LEN" == "0" ]]; then
    MAX_MODEL_LEN=131072
fi
# Block size for Qwen3.5-27B GDN layers is 784 (derived at runtime by vLLM).
# --max-num-batched-tokens must be in [784, 1568) for mamba-cache-mode=align.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1567}"
LMCACHE_CPU_SIZE_GB="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
# Chunk size must match the vLLM-derived GDN block size (784 for Qwen3.5-27B BF16 tp=1).
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-784}"

mkdir -p "$RESULT_DIR"

resolve_trace_source
install_agentic_deps
# install_agentic_deps pip-installs aiperf but doesn't guarantee PATH visibility
# on all images. ensure_aiperf (no-op if already available) installs into a venv
# and exports the venv bin dir onto PATH.
AIPERF_SOURCE_DIR="$AIPERF_DIR" ensure_aiperf

start_gpu_monitor

# LMCacheMPConnector requires a separate lmcache server process.
lmcache server \
    --chunk-size "$LMCACHE_CHUNK_SIZE" \
    --l1-size-gb "$LMCACHE_CPU_SIZE_GB" \
    --eviction-policy LRU \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!

sleep 3

set -x
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prime LMCache lookup counters before aiperf starts. OTel counters for
# lmcache_mp_lookup_* are lazy-initialized and must exist before aiperf
# discovers metrics at startup.
python3 -c "
import urllib.request, json, os
req = urllib.request.Request(
    'http://0.0.0.0:${PORT}/v1/chat/completions',
    data=json.dumps({'model': os.environ['MODEL'],
                     'messages': [{'role': 'user', 'content': 'hello'}],
                     'max_tokens': 1}).encode(),
    headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req, timeout=60)
except Exception:
    pass
" || true
sleep 2

build_replay_cmd "$RESULT_DIR"
# Scrape the LMCache server's Prometheus endpoint (lmcache_mp_lookup_*)
# so process_agentic_result.py can compute lmcache_hit_rate.
REPLAY_CMD+=" --server-metrics-url http://0.0.0.0:8080/metrics"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set +x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true

write_agentic_result_json "$RESULT_DIR"

stop_gpu_monitor
