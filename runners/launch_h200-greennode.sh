#!/usr/bin/env bash
# Launch script for GreenNode H200 runners (plain Docker, no Slurm/enroot).
# Selected by benchmark-tmpl.yml via: bash ./runners/launch_${RUNNER_NAME%%_*}.sh
# where RUNNER_NAME like "h200-greennode_01" -> this script.

set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export PORT="${PORT:-8888}"

docker pull "$IMAGE"

RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE PORT RANDOM_RANGE_RATIO
  EXP_NAME MODEL MODEL_PREFIX ISL OSL MAX_MODEL_LEN
  MAX_NUM_BATCHED_TOKENS
  IMAGE FRAMEWORK BENCHMARK_CLIENT PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING NUM_SPECULATIVE_TOKENS DISAGG
  RUN_EVAL EVAL_ONLY SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  OFFLOADING TOTAL_CPU_DRAM_GB DURATION
  INPUT_FILE PUBLIC_DATASET CUSTOM_DATASET_TYPE TOKENIZER
  NO_FIXED_SCHEDULE NUM_WARMUP_SESSIONS REQUEST_COUNT STRIP_TRACE_DELAYS
  RESULT_DIR RESULT_FILENAME
  PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX
  RUNNER_NAME RUNNER_TYPE
)
ENV_ARGS=()
for v in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$v")
done

BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  BENCH_SCRIPT="${BENCH_BASE}.sh"
fi

# DCGM exporter sidecar. Runs --network host so AIPerf inside the model
# container (also host network) reaches GPU telemetry at localhost:9400/metrics.
# SYS_ADMIN needed for DCGM_FI_PROF_* metrics; port 9400 must be free
# (conflicts with any host-level/k8s dcgm-exporter). Torn down on script exit.
DCGM_IMAGE="${DCGM_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04}"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-greennode}"
docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" "$DCGM_IMAGE"
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT

docker run --rm \
  --init \
  --gpus all \
  --ipc=host \
  --network host \
  --shm-size=32g \
  -v "$GITHUB_WORKSPACE:/workspace" \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  "$BENCH_SCRIPT"
