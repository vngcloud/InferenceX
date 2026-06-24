#!/usr/bin/env bash
# Serve Qwen3-8B with SGLang v0.5.12 + LMCache (0.5.0, baked into sglang-lmcache:v0.5.12).
# SGLang wires LMCache in via --enable-lmcache (layerwise CPU-offload connector).
set -euo pipefail

IMAGE="sglang-lmcache:v0.5.12"
NAME="sglang-lmcache"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST_PORT="${HOST_PORT:-30000}"
LMC_METRICS_PORT="${LMC_METRICS_PORT:-7011}"  # host port -> LMCache internal :7001
GPU="${GPU:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus "\"device=${GPU}\"" \
  --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HERE:/config" \
  -e LMCACHE_USE_EXPERIMENTAL=True \
  -e LMCACHE_CONFIG_FILE=/config/lmcache_cpu.yaml \
  -e LMCACHE_LOG_LEVEL=INFO \
  -e PYTHONHASHSEED=0 \
  -p "${HOST_PORT}:30000" \
  -p "${LMC_METRICS_PORT}:7001" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80 \
    --context-length 16384 \
    --enable-metrics \
    --enable-lmcache

echo "Started $NAME on http://localhost:${HOST_PORT}  (GPU $GPU, model $MODEL)"
echo "SGLang metrics:  http://localhost:${HOST_PORT}/metrics"
echo "LMCache metrics: http://localhost:${LMC_METRICS_PORT}/metrics"
echo "Follow logs:  docker logs -f $NAME"
