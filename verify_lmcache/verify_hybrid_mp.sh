#!/usr/bin/env bash
# Verify that LMCache OFFLOADS a HYBRID-attention model on the NEW stack:
#   vLLM 0.23.0 + lmcache 0.5.0 + LMCacheMPConnector.   (cf. REPORT.md §9.6)
#
# This is the opposite result to verify_hybrid.sh (which proves the OLD pinned
# stack cannot). It uses the gold-standard isolation: store on one engine, then
# RESTORE on a brand-new engine whose GPU prefix cache is empty -- so any hit can
# only have come from LMCache.
#
#   ./verify_hybrid_mp.sh        # expects: warm fresh-engine external hit rate ~99%
set -euo pipefail

IMAGE="vllm-lmcache-mp:v0.23.0"          # built by Dockerfile.vllm_mp
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPU="${GPU:-3}"
CHUNK="${CHUNK:-528}"                      # unified attention block size N for Qwen3.5-4B
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== building $IMAGE (FROM vllm v0.23.0 + lmcache 0.5.0) ==="
docker build -f "$HERE/Dockerfile.vllm_mp" -t "$IMAGE" "$HERE" >/dev/null
docker rm -f lmcache-srv vllm-mp >/dev/null 2>&1 || true

wait_api(){ for i in $(seq 1 40); do
  curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"' && return 0
  docker ps -a --filter name=vllm-mp --format '{{.Status}}' | grep -q Exited && { echo "vLLM EXITED:"; docker logs vllm-mp 2>&1|grep -iE "ValueError:|Error"|tail -4; return 1; }
  sleep 10; done; return 1; }

start_vllm(){ # $1 = gpu-mem-util
  docker run -d --name vllm-mp --network host --ipc=host --gpus "\"device=${GPU}\"" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e LMCACHE_LOG_LEVEL=INFO \
    --entrypoint vllm "$IMAGE" \
    serve "$MODEL" --served-model-name "$MODEL" \
    --max-model-len 16384 --gpu-memory-utilization "$1" --max-num-seqs 32 \
    --mamba-cache-mode align --enable-prefix-caching --max-num-batched-tokens "$CHUNK" \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' >/dev/null
}

# 1) persistent LMCache MP server (shares IPC ns + GPU with vLLM for CUDA-IPC transfer)
echo "=== starting LMCache MP server (zmq :5555, chunk-size=$CHUNK) ==="
docker run -d --name lmcache-srv --network host --ipc=host --gpus "\"device=${GPU}\"" \
  -e LMCACHE_LOG_LEVEL=INFO --entrypoint lmcache "$IMAGE" \
  server --chunk-size "$CHUNK" --l1-size-gb 20 --eviction-policy LRU --port 5555 --http-port 8080 >/dev/null
sleep 5; docker logs lmcache-srv 2>&1 | grep -iE "ZMQ cache server is running" | tail -1

# 2) first engine -> COLD request -> LMCache stores
echo "=== starting vLLM (engine #1) ==="; start_vllm 0.85; wait_api || exit 1
REQ=$(mktemp); python3 - "$REQ" "$MODEL" <<'PY'
import json,sys
para="LMCache stores key-value tensors from prefill so repeated prefixes skip recompute; it chunks state into fixed blocks. "
body="".join(f"Reference section {i}: {para}\n" for i in range(140))
json.dump({"model":sys.argv[2],"messages":[{"role":"user","content":f"Context.\n{body}\nQ: reply with one word."}],"max_tokens":8,"temperature":0,"stream":False}, open(sys.argv[1],"w"))
PY
echo "--- COLD request ---"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @"$REQ" -o /dev/null -w '  cold latency=%{time_total}s\n'
sleep 3
echo "  external_cache after cold: $(curl -s http://localhost:8000/metrics | grep -E 'external_prefix_cache_(queries|hits)_total' | grep -v '^#' | tr '\n' ' ')"
echo "  server stores: $(docker logs lmcache-srv 2>&1 | grep -c 'Stored .* tokens') chunk(s)"

# 3) replace engine -> brand-new empty GPU cache, server keeps DRAM copy -> WARM hit can ONLY be LMCache
echo "=== replacing vLLM with a FRESH engine (empty GPU prefix cache) ==="
docker rm -f vllm-mp >/dev/null 2>&1; sleep 3
start_vllm 0.5; wait_api || exit 1   # lower util: server still pins GPU mem from engine #1
echo "--- WARM request on fresh engine ---"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @"$REQ" -o /dev/null -w '  warm latency=%{time_total}s\n'
sleep 3
echo "--- result ---"
curl -s http://localhost:8000/metrics | grep -E 'external_prefix_cache_(queries|hits)_total' | grep -v '^#' | sed 's/^/  /'
docker logs lmcache-srv 2>&1 | grep -iE "Retrieved .* tokens" | tail -1 | sed 's/^/  /'
docker logs vllm-mp 2>&1 | grep -oE "External prefix cache hit rate: [0-9.]+%" | tail -1 | sed 's/^/  vLLM logger: /'
echo "  => non-zero external hits on a fresh engine == LMCache restored the hybrid model's KV."
rm -f "$REQ"; docker rm -f vllm-mp lmcache-srv >/dev/null 2>&1 || true
echo "Done. GPU $GPU freed."
