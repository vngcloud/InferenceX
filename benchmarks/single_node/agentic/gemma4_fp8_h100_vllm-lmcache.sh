#!/usr/bin/env bash
# Gemma 4 31B FP8 on H100 via vLLM + LMCache — agentic trace replay.
#
# Uses LMCacheConnectorV1 (embedded in vLLM). Cache hits flow through
# vLLM's prefix-caching path and are visible at the vLLM /metrics endpoint,
# where process_agentic_result.py reads them as server_gpu_cache_hit_rate.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR
#
# Counterpart config keys:
#   gemma4-fp8-h100-2x-vllm-lmcache
#   gemma4-fp8-h100-1x-vllm-lmcache-gn00

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"

resolve_trace_source
install_agentic_deps

start_gpu_monitor

set -x
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --server-metrics-url http://0.0.0.0:${PORT}/metrics"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set +x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true

write_agentic_result_json "$RESULT_DIR"

stop_gpu_monitor
