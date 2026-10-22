#!/usr/bin/env bash
set -euo pipefail
set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export MODEL_STORE_MOUNT="${MODEL_STORE_MOUNT:-/mnt/models}"
export MODEL_STORE="${MODEL_STORE:-/models}"
export PORT="${PORT:-8888}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/mnt/uv-cache}"

docker pull "$IMAGE"

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "vllm" ]] && printf '' || printf "_%s" "$FRAMEWORK")
# Mirrors the SPEC_SUFFIX convention in launch_h200-cw.sh / launch_h200-nb.sh /
# launch_h200-dgxc-slurm.sh, extended to "draft_model" (external-draft methods
# like eagle3/dspark, as opposed to internal "mtp") since this pool's first
# fixed-seq-len draft-model recipe (gemma4 router+eagle3) needs a distinct
# script name from the no-spec-decoding baseline sharing the same precision.
case "$SPEC_DECODING" in
  mtp) SPEC_SUFFIX="_mtp" ;;
  draft_model) SPEC_SUFFIX="_specdec" ;;
  *) SPEC_SUFFIX="" ;;
esac
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200${FRAMEWORK_SUFFIX}"
BENCH_SCRIPT="${BENCH_BASE}${SPEC_SUFFIX}.sh"
# Fall back to the unsuffixed name when no spec-specific script exists (same
# idiom as launch_b300-nv.sh). This pool's twelve pre-existing mtp recipes
# (glm5.2, glm5.2dspark, glm5.2eagle*, glm5.2prod/pdeep/psymm, glm5.2edeep,
# glm5.2elmhead) all ship a single script that branches on SPEC_DECODING
# internally, so an unconditional SPEC_SUFFIX makes every one of them exit
# 127 on a missing file.
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  BENCH_SCRIPT="${BENCH_BASE}.sh"
fi
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-h200-greennode_01}"
RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE PORT
  EXP_NAME MODEL MODEL_PREFIX IMAGE FRAMEWORK PRECISION TP EP_SIZE DP_ATTENTION
  PP_SIZE DCP_SIZE PCP_SIZE
  CONC SPEC_DECODING SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO
  KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB DURATION
  HICACHE_RATIO
  RESULT_DIR RESULT_FILENAME RUN_EVAL EVAL_ONLY
  GITHUB_WORKSPACE RUNNER_NAME RUNNER_TYPE AIPERF_UV_CACHE_DIR
  # Agentic recipes need these inside the container: benchmark_lib.sh's
  # install_agentic_deps requires AIPERF_PYTHON_VERSION + INFMAX_CONTAINER_WORKSPACE,
  # build_replay_cmd requires the AIPERF_*/AGENTIC_* knobs, and
  # run_agentic_replay_and_write_outputs requires IS_MULTINODE/REQUIRE_POWER.
  INFMAX_CONTAINER_WORKSPACE IS_MULTINODE REQUIRE_POWER AIPERF_EXPERIMENTAL_FAST
)
ENV_ARGS=()
for name in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$name")
done
# runtime_settings.sh (sourced by the launch step) publishes the canonical list
# of agentic runtime env vars to forward; carry each into the container.
for name in ${INFERENCEX_RUNTIME_ENV_VARS:-}; do
  ENV_ARGS+=(-e "$name")
done

# A recipe opts into a non-default DCGM fieldset by shipping a CSV sidecar
# next to itself (same path, ".gpu_metrics.csv" instead of ".sh"). Verified
# live (Vietinbank DCGM pipeline, 5-min smoke, 0 errors): dcgm-exporter reads
# it via -f, and the recipe passes the same file to aiperf's --gpu-telemetry
# so the fields it parses always match what the exporter actually serves.
GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"
DCGM_MOUNT_ARGS=()
DCGM_CMD_ARGS=()
if [[ -f "$GPU_METRICS_CSV" ]]; then
  DCGM_MOUNT_ARGS=(-v "$GITHUB_WORKSPACE/$GPU_METRICS_CSV:/etc/dcgm-exporter/custom.csv:ro")
  DCGM_CMD_ARGS=(-f /etc/dcgm-exporter/custom.csv)
fi

docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" \
  "${DCGM_MOUNT_ARGS[@]}" \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04 \
  "${DCGM_CMD_ARGS[@]}"
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT

docker run --rm --init --gpus all --ipc=host --network host --shm-size=32g \
  -v "$GITHUB_WORKSPACE:/workspace" \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -v "$MODEL_STORE_MOUNT:$MODEL_STORE:ro" \
  -v "$AIPERF_UV_CACHE_DIR:$AIPERF_UV_CACHE_DIR" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  "$BENCH_SCRIPT"
