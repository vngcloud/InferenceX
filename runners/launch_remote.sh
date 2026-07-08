#!/usr/bin/env bash
set -euo pipefail
set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"

RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE RANDOM_RANGE_RATIO
  EXP_NAME MODEL MODEL_PREFIX ISL OSL MAX_MODEL_LEN
  MAX_NUM_BATCHED_TOKENS
  IMAGE FRAMEWORK BENCHMARK_CLIENT PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING NUM_SPECULATIVE_TOKENS DISAGG
  RUN_EVAL EVAL_ONLY SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  OFFLOADING TOTAL_CPU_DRAM_GB DURATION
  INPUT_FILE PUBLIC_DATASET CUSTOM_DATASET_TYPE TOKENIZER WEKA_NUM_DATASET_ENTRIES
  REMOTE_URL REMOTE_ENDPOINT REMOTE_SERVER_METRICS_URL REMOTE_GPU_TELEMETRY_URL REMOTE_API_KEY
  RESULT_DIR RESULT_FILENAME
  PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX
  RUNNER_NAME RUNNER_TYPE
)
ENV_ARGS=()
for v in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$v")
done

# The pre-built full AIPerf image is distroless and runs as non-root UID 1000,
# so it can't write into the bind-mounted workspace (owned by the runner user).
# Map the container to the host runner's uid/gid so mkdir/results writes succeed
# and result files stay runner-owned for the upload/cleanup steps. HOME=/app is
# owned by 1000 and unwritable under the remapped uid, so point HOME at the
# writable workspace (matplotlib is the only remaining HOME writer).
docker run --rm \
  --init \
  --ipc=host \
  --network host \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -v "$GITHUB_WORKSPACE:/workspace" \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  benchmarks/single_node/agentic/_remote_replay.sh
