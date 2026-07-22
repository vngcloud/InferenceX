#!/usr/bin/env bash
set -euo pipefail
set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export PORT="${PORT:-8888}"

docker pull "$IMAGE"

BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200_${FRAMEWORK}.sh"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-h200-greennode_01}"
RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE PORT
  EXP_NAME MODEL MODEL_PREFIX IMAGE FRAMEWORK PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB DURATION
  RESULT_DIR RESULT_FILENAME RUN_EVAL EVAL_ONLY
  GITHUB_WORKSPACE RUNNER_NAME RUNNER_TYPE
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
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  "$BENCH_SCRIPT"
