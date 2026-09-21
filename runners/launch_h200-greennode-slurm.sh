#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars GPU_COUNT

set -x

# Slurm path for h200-greennode_06 (partition "test"), sibling of
# launch_h200-greennode.sh's docker path. --gres=gpu:$GPU_COUNT with no
# --exclusive so a second allocation can hold the other half of the node.
SLURM_PARTITION="test"
SLURM_ACCOUNT="greennode"
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/mnt/uv-cache}"

# pyxis shares the host netns by default (no --container-unshare=net) -> two
# concurrent allocations on this node can't both hardcode PORT=8888.
free_tcp_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}
export PORT="${PORT:-$(free_tcp_port)}"

# enroot needs `docker://registry#path` for a non-default registry; auto-detect
# by whether the first path component looks like a host (has a dot/colon, or
# is localhost), same heuristic as launch_b200-nscale-slurm.sh.
enroot_uri_for_image() {
    local image_ref="$1"
    local first_component="${image_ref%%/*}"
    if [[ "$image_ref" == */* && (
        "$first_component" == *.* ||
        "$first_component" == *:* ||
        "$first_component" == "localhost"
    ) ]]; then
        printf 'docker://%s#%s\n' "$first_component" "${image_ref#*/}"
    else
        printf 'docker://%s\n' "$image_ref"
    fi
}

# Local disk, not /shared (NFS) -- much faster for large image imports.
SQUASH_CACHE_DIR="/mnt/containers"
mkdir -p "$SQUASH_CACHE_DIR"
SQUASH_FILE="${SQUASH_CACHE_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
DOCKER_IMAGE_URI="$(enroot_uri_for_image "$IMAGE")"
LOCK_FILE="${SQUASH_FILE}.lock"

# GH Actions cancel can SIGKILL us before EXIT traps run, leaking the
# allocation; clear any stale job from this runner name before requesting a new one.
scancel --name="$RUNNER_NAME" 2>/dev/null || true

salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --gres=gpu:"$GPU_COUNT" --time=180 --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: failed to resolve h200-greennode Slurm allocation" >&2
    exit 1
fi
trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

srun --jobid="$JOB_ID" bash -c "
    export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
    mkdir -p \$ENROOT_CACHE_PATH
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" $DOCKER_IMAGE_URI
    fi
"

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "vllm" ]] && printf '' || printf "_%s" "$FRAMEWORK")
case "$SPEC_DECODING" in
  mtp) SPEC_SUFFIX="_mtp" ;;
  draft_model) SPEC_SUFFIX="_specdec" ;;
  *) SPEC_SUFFIX="" ;;
esac
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200${FRAMEWORK_SUFFIX}"
BENCH_SCRIPT="${BENCH_BASE}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  BENCH_SCRIPT="${BENCH_BASE}.sh"
fi

# DCGM sidecar via srun --overlap in the same allocation (docker launcher
# uses a second `docker run -d` instead). Own port, same reason as above.
DCGM_PORT="$(free_tcp_port)"
DCGM_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04"
DCGM_IMAGE_URI="$(enroot_uri_for_image "$DCGM_IMAGE")"
DCGM_SQUASH="${SQUASH_CACHE_DIR}/$(echo "$DCGM_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
DCGM_LOCK_FILE="${DCGM_SQUASH}.lock"
srun --jobid="$JOB_ID" bash -c "
    export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
    mkdir -p \$ENROOT_CACHE_PATH
    exec 9>\"$DCGM_LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $DCGM_SQUASH'; exit 1; }
    if unsquashfs -l \"$DCGM_SQUASH\" > /dev/null 2>&1; then
        echo 'DCGM squash file already exists and is valid, skipping import'
    else
        rm -f \"$DCGM_SQUASH\"
        enroot import -o \"$DCGM_SQUASH\" $DCGM_IMAGE_URI
    fi
"

GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"
DCGM_MOUNT_ARG=""
DCGM_EXTRA_ARGS=""
if [[ -f "$GPU_METRICS_CSV" ]]; then
    DCGM_MOUNT_ARG=",$GITHUB_WORKSPACE/$GPU_METRICS_CSV:/etc/dcgm-exporter/custom.csv:ro"
    DCGM_EXTRA_ARGS="-f /etc/dcgm-exporter/custom.csv"
fi

srun --jobid="$JOB_ID" --overlap \
    --container-image="$DCGM_SQUASH" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace${DCGM_MOUNT_ARG}" \
    --no-container-mount-home \
    --container-remap-root \
    --no-container-entrypoint \
    dcgm-exporter -a ":$DCGM_PORT" $DCGM_EXTRA_ARGS &
DCGM_SRUN_PID=$!
trap 'rc=$?; kill "$DCGM_SRUN_PID" 2>/dev/null || true; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

export AIPERF_GPU_TELEMETRY_URL="http://localhost:${DCGM_PORT}/metrics"
if [[ -n "$DCGM_EXTRA_ARGS" ]]; then
    export AIPERF_GPU_TELEMETRY_METRICS_CSV="/etc/dcgm-exporter/custom.csv"
fi

srun --jobid="$JOB_ID" \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_UV_CACHE_DIR:$AIPERF_UV_CACHE_DIR" \
    --no-container-mount-home \
    --container-remap-root \
    --container-workdir=/workspace/ \
    --no-container-entrypoint --export=ALL,PORT="$PORT",AIPERF_GPU_TELEMETRY_URL,AIPERF_GPU_TELEMETRY_METRICS_CSV \
    bash "$BENCH_SCRIPT"

kill "$DCGM_SRUN_PID" 2>/dev/null || true
scancel "$JOB_ID"
