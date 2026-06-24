#!/usr/bin/env bash
# Runs INSIDE the vllm/vllm-openai:v0.23.0 container.
# Starts a standalone LMCache MP server, then vLLM wired to it via LMCacheMPConnector.
# This is the deployment shape required for HYBRID-attention models (Qwen3.5/Qwen3-Next):
# the in-process LMCacheConnectorV1 cannot handle their heterogeneous KV layout.
set -e
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
CHUNK="${CHUNK:-528}"          # = vLLM unified attention block size N (multiple of N required)

echo "[boot] lmcache $(python3 -c 'import lmcache;print(lmcache.__version__)' 2>/dev/null | tail -1)"
echo "[boot] starting lmcache MP server (zmq :5555, chunk-size=$CHUNK)"
lmcache server --chunk-size "$CHUNK" --l1-size-gb 20 --eviction-policy LRU \
  --http-host 0.0.0.0 --http-port 8080 > /tmp/lmcache_server.log 2>&1 &
SRV=$!

# wait for the ZMQ server to be up
for i in $(seq 1 40); do
  grep -qiE "listening|started|serving|bound|fired|ready|MessageQueueServer|MPCacheServer" /tmp/lmcache_server.log && break
  kill -0 "$SRV" 2>/dev/null || { echo "[boot] lmcache server died:"; cat /tmp/lmcache_server.log; exit 1; }
  sleep 1
done
echo "[boot] lmcache server up; launching vLLM"

exec vllm serve "$MODEL" --served-model-name "$MODEL" \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --mamba-cache-mode align --enable-prefix-caching \
  --max-num-batched-tokens "$CHUNK" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}'
