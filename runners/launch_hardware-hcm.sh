#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars GPU_COUNT

set -x

# Slurm path for hardware-hcm-8x_01/_02 and hardware-hcm-4x_01/_02 (on-node
# runners for the `hardware-hcm` partition: hgx-h200-01/hgx-h200-02, 8x H200
# each, on the research Slurm cluster). GPU_COUNT and the salloc below are
# shared as-is between the 8x and 4x runner identities on a box -- no
# per-flavor branching needed since Slurm's own gres accounting (no
# --exclusive) queues a job if not enough GPUs are free.
# Adapted from launch_h200-greennode-slurm.sh, which
# is for a *different* cluster/partition ("test"/"greennode" account) --
# copy-pasting that launcher's partition/account here silently allocated
# against an account (greennode) with zero associations on this cluster
# (`sacctmgr show account greennode` -> no users), so jobs would fail at
# salloc. Real jobs on hgx-h200-01/02 use account "dev" (verified via
# `sacct -j <id>` on manually-submitted jobs).
#
# No 1x-node redirect here (unlike launch_h200-greennode-slurm.sh): these are
# normal 8x boxes with a normal filesystem/user setup, so none of the
# workarounds in launch_h200-greennode-slurm-1x.sh (no passwd entry for the
# runner user, workspace shipped over srun stdin, etc, plus that script's
# hardcoded --nodelist for its own 1x node) apply here.
SLURM_PARTITION="hardware-hcm"
SLURM_ACCOUNT="dev"
# Pin the allocation to this box's own Slurm node name. Without --nodelist,
# salloc is free to land on either node in the partition -- caught for real
# when a concurrent hardware-hcm-2x_02 job (running on hgx-h200-02) was
# allocated hgx-h200-01 instead: every --container-mounts path below
# ($GITHUB_WORKSPACE, /mnt/hf_hub_cache, /mnt/containers) is local to the
# box the *runner* lives on, and srun executed on a sibling machine that
# doesn't have those files ("Communication connection failure" at task
# launch, job 863). The local hostname doesn't match Slurm's NodeName
# either (hgx-h200-001 locally vs hgx-h200-01 in Slurm), hence the map.
case "$(hostname -s)" in
  hgx-h200-001) SLURM_NODELIST="hgx-h200-01" ;;
  hgx-h200-02) SLURM_NODELIST="hgx-h200-02" ;;
  *)
    echo "ERROR: unrecognized hardware-hcm host $(hostname -s), add it to the map in $0" >&2
    exit 1
    ;;
esac
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/mnt/uv-cache}"

# pyxis shares the host netns by default (no --container-unshare=net) -> two
# concurrent allocations on this node can't both hardcode PORT=8888.
# benchmark-tmpl.yml sets PORT=8888 as a job-level default env var, so
# ${PORT:-...} never falls through to free_tcp_port() -- it's always
# already non-empty. Override unconditionally instead: this launcher must
# always pick its own free port, never trust an inherited default.
free_tcp_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}
export PORT="$(free_tcp_port)"

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
# /mnt/sqsh, not /mnt/containers: the latter is stackops-owned 755 on
# hgx-h200-02, so a non-stackops user cannot pre-import the (private-VCR)
# squashfs there; /mnt/sqsh is world-writable (mkdir a+rwX once) and the
# launcher user only needs to read the sqsh + create its lock file.
SQUASH_CACHE_DIR="/mnt/sqsh"
mkdir -p "$SQUASH_CACHE_DIR"
SQUASH_FILE="${SQUASH_CACHE_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
DOCKER_IMAGE_URI="$(enroot_uri_for_image "$IMAGE")"
LOCK_FILE="${SQUASH_FILE}.lock"

# GH Actions cancel can SIGKILL us before EXIT traps run, leaking the
# allocation; clear any stale job from this runner name before requesting a new one.
scancel --name="$RUNNER_NAME" 2>/dev/null || true

# hgx-h200-01/02: 192 CPUs / ~2TB RAM over 8 GPUs (confirmed via
# `sinfo -N -p hardware-hcm -o '%N %c %m %G'`) -- same 24 CPU/GPU ratio as
# h200-greennode, sized proportionally so two allocations can share a node.
salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --nodelist="$SLURM_NODELIST" \
    --gres=gpu:"$GPU_COUNT" \
    --cpus-per-task=$((GPU_COUNT * 24)) --mem=$((GPU_COUNT * 240))G \
    --time=180 --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: failed to resolve hardware-hcm Slurm allocation" >&2
    exit 1
fi
trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

srun --jobid="$JOB_ID" bash -c "
    export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
    mkdir -p \$ENROOT_CACHE_PATH
    export TMPDIR=/mnt/tmp
    mkdir -p \$TMPDIR
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
    export TMPDIR=/mnt/tmp
    mkdir -p \$TMPDIR
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

# --container-writable: enroot's squashfs mount is read-only by default,
# unlike docker's writable-overlay default. Various tools baked into these
# images write into their own rootfs at runtime with no override available
# (triton -> /root/.triton, vllm -> /root/.cache/vllm/*, etc.).
srun --jobid="$JOB_ID" \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_UV_CACHE_DIR:$AIPERF_UV_CACHE_DIR" \
    --no-container-mount-home \
    --container-remap-root \
    --container-writable \
    --container-workdir=/workspace/ \
    --no-container-entrypoint --export=ALL,PORT="$PORT",AIPERF_GPU_TELEMETRY_URL,AIPERF_GPU_TELEMETRY_METRICS_CSV \
    bash "$BENCH_SCRIPT"

kill "$DCGM_SRUN_PID" 2>/dev/null || true
scancel "$JOB_ID"
