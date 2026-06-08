#!/usr/bin/env bash
# Launch script for GreenNode H200 runners (plain Docker, no Slurm/enroot).
# Selected by benchmark-tmpl.yml via: bash ./runners/launch_${RUNNER_NAME%%_*}.sh
# where RUNNER_NAME like "h200-greennode_00" → this script.
# The `_h100` token below is only a serving-script filename convention.

set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export PORT="${PORT:-8888}"

# Pull the image up front so timing/failure is visible.
docker pull "$IMAGE"

# All env vars from benchmark-tmpl.yml's env: block and the calling step
# need to reach the benchmark script inside the container.
RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE PORT RANDOM_RANGE_RATIO
  EXP_NAME MODEL MODEL_PREFIX ISL OSL MAX_MODEL_LEN
  MAX_NUM_BATCHED_TOKENS
  IMAGE FRAMEWORK BENCHMARK_CLIENT PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING NUM_SPECULATIVE_TOKENS DISAGG
  RUN_EVAL EVAL_ONLY SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  OFFLOADING TOTAL_CPU_DRAM_GB DURATION
  INPUT_FILE CUSTOM_DATASET_TYPE
  NO_FIXED_SCHEDULE NUM_WARMUP_SESSIONS REQUEST_COUNT STRIP_TRACE_DELAYS
  SEARCH_RECIPE CONCURRENCY_MIN CONCURRENCY_MAX SEARCH_MAX_ITERATIONS SLA_MS
  BENCHMARK_DURATION BENCHMARK_GRACE_PERIOD
  RESULT_DIR RESULT_FILENAME
  PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX
  RUNNER_NAME RUNNER_TYPE
)
ENV_ARGS=()
for v in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$v")
done

# Prefer a framework-tagged script (e.g. gemma4_fp8_h100_sglang.sh) so the
# same model can be benchmarked on multiple engines side by side; fall back
# to the historical engine-less name (e.g. gemma4_fp8_h100.sh) when no tagged
# script exists. No spec suffix here on purpose: scripts that support MTP
# (e.g. gemma4_fp8_h100.sh) branch internally on $SPEC_DECODING.
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h100"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  BENCH_SCRIPT="${BENCH_BASE}.sh"
fi

# Dataset-scoped MiniMax agentic configs keep 64k/128k/1l1 in exp-name for
# result traceability, but they reuse the same serving launcher.
if [[ ! -f "$BENCH_SCRIPT" && "${EXP_NAME%%_*}" == minimaxm2.5-agentic-* ]]; then
  BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}minimaxm2.5-agentic_${PRECISION}_h100"
  BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}.sh"
  if [[ ! -f "$BENCH_SCRIPT" ]]; then
    BENCH_SCRIPT="${BENCH_BASE}.sh"
  fi
fi

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Benchmark script not found: $BENCH_SCRIPT" >&2
  exit 1
fi

docker run --rm \
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
