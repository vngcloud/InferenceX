#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM v0.21.0 + bundled LMCache (0.4.5) on GPU 3.
# LMCache is wired in via vLLM's KV-connector interface (LMCacheConnectorV1).
set -euo pipefail

IMAGE="vllm/vllm-openai:v0.21.0"
NAME="vllm-lmcache"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST_PORT="${HOST_PORT:-8100}"
GPU="${GPU:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus "\"device=${GPU}\"" \
  --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HERE:/config" \
  -e LMCACHE_CONFIG_FILE=/config/lmcache_cpu.yaml \
  -e LMCACHE_LOG_LEVEL=INFO \
  -e PYTHONHASHSEED=0 \
  -p "${HOST_PORT}:8000" \
  -p "7001:7001" \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

echo "Started $NAME on http://localhost:${HOST_PORT}  (GPU $GPU, model $MODEL)"
echo "Follow logs:  docker logs -f $NAME"
