#!/usr/bin/env bash
set -euo pipefail
set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_CACHE="${CONTAINER_HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export MODEL_STORE_MOUNT="${MODEL_STORE_MOUNT:-/mnt/models}"
export MODEL_STORE="${MODEL_STORE:-/mnt/models}"
export PORT="${PORT:-8888}"

docker pull "$IMAGE"

BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b300-netperf_${FRAMEWORK}.sh"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-b300-netperf_00}"
RUN_ENV=(
  HF_TOKEN PORT
  EXP_NAME MODEL MODEL_PREFIX IMAGE FRAMEWORK PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB DURATION
  HICACHE_RATIO
  RESULT_DIR RESULT_FILENAME RUN_EVAL EVAL_ONLY
  GITHUB_WORKSPACE GITHUB_RUN_ID GITHUB_RUN_ATTEMPT RUNNER_NAME RUNNER_TYPE
)
ENV_ARGS=()
for name in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$name")
done

docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT

docker run --rm --init --gpus all --ipc=host --network host --shm-size=32g \
  -v "$GITHUB_WORKSPACE:/workspace" \
  -v /mnt/test-raid0:/mnt/test-raid0 \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -v "$MODEL_STORE_MOUNT:$MODEL_STORE:ro" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  "$BENCH_SCRIPT"
