#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars GPU_COUNT IMAGE GITHUB_WORKSPACE RUNNER_NAME

set -x

# 1-GPU slurm path: hoanq3-h200-1x-han-3-1 (1xH200, 16 CPU, ~118G RAM)
# in the same "test" partition as han-1. Exec'd by launch_h200-greennode-slurm.sh
# when GPU_COUNT=1. Differs from the han-1 path because on these nodes:
# - the runner user has no passwd entry and /mnt is root-only, so pyxis
#   (enroot.conf -> /mnt/enroot-*-$uid) cannot start a container. Use the
#   enroot CLI with its paths under /var/tmp instead.
# - the runner's workspace (on han-1's disk) is not visible, so ship it over
#   srun stdin and pull results back the same way.
# - enroot start does not pass the host env into the container, so the job env
#   is written to a 0600 file inside the shipped workspace and sourced there.
SLURM_PARTITION="test"
SLURM_ACCOUNT="greennode"
# Its sibling hoanq3-h200-1x-han-3 lacks nvidia-container-cli (enroot's GPU
# hook fails there), so pin the one node that works.
SLURM_NODELIST="hoanq3-h200-1x-han-3-1"
# Per uid: the runner user and people testing by hand share these nodes.
NODE_ROOT="/var/tmp/inferencex-$(id -u)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/mnt/uv-cache}"
# One GPU per node -> one job per node, so fixed ports cannot collide.
export PORT=8888
DCGM_PORT=9400
DCGM_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04"

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

scancel --name="$RUNNER_NAME" 2>/dev/null || true
salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --nodelist="$SLURM_NODELIST" --gres=gpu:1 --cpus-per-task=16 --mem=100G \
    --time=180 --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: failed to resolve h200-greennode 1x Slurm allocation" >&2
    exit 1
fi
WS="$NODE_ROOT/ws-$JOB_ID"
trap 'rc=$?; srun --jobid="$JOB_ID" --overlap rm -rf "$WS" 2>/dev/null || true; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

# Enroot env + a container per image (created once per node, reused). The
# image's /etc/rc is replaced so `enroot start name cmd...` runs cmd instead of
# the image ENTRYPOINT (vllm-openai's is `vllm serve`).
ENROOT_ENV="R=$NODE_ROOT/enroot; export ENROOT_DATA_PATH=\$R/data ENROOT_RUNTIME_PATH=\$R/run ENROOT_CACHE_PATH=\$R/cache ENROOT_TEMP_PATH=\$R/tmp; mkdir -p \$R/data \$R/run \$R/cache \$R/tmp"
ensure_container() {
    local image="$1" name
    name="$(echo "$image" | sed 's/[\/:@#.]/_/g')"
    srun --jobid="$JOB_ID" bash -c "
        set -e; $ENROOT_ENV
        exec 9>\$R/$name.lock; flock -w 1800 9
        # .ok marks a finished create: a half-unpacked container is redone.
        if [ ! -f \$R/$name.ok ]; then
            enroot remove -f '$name' 2>/dev/null || true
            rm -f \$R/$name.sqsh
            enroot import -o \$R/$name.sqsh $(enroot_uri_for_image "$image")
            enroot create -n '$name' \$R/$name.sqsh
            rm -f \$R/$name.sqsh
            printf 'cd /workspace 2>/dev/null || true\nexec \"\$@\"\n' > \$ENROOT_DATA_PATH/$name/etc/rc
            touch \$R/$name.ok
        fi
    " >&2 || return 1
    echo "$name"
}
IMAGE_CT="$(ensure_container "$IMAGE")"
DCGM_CT="$(ensure_container "$DCGM_IMAGE")"

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
GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"
DCGM_EXTRA_ARGS=""
DCGM_CSV_MOUNT=""
if [[ -f "$GPU_METRICS_CSV" ]]; then
    DCGM_EXTRA_ARGS="-f /etc/dcgm-exporter/custom.csv"
    DCGM_CSV_MOUNT="-m $NODE_ROOT/ws-$JOB_ID/$GPU_METRICS_CSV:/etc/dcgm-exporter/custom.csv"
    export AIPERF_GPU_TELEMETRY_METRICS_CSV="/etc/dcgm-exporter/custom.csv"
fi
export AIPERF_GPU_TELEMETRY_URL="http://localhost:${DCGM_PORT}/metrics"

# Job env for the container. Host/session vars stay behind so the image keeps
# its own PATH, CUDA and library settings. Holds HF_TOKEN -> 0600, removed by the trap.
ENV_FILE="$(mktemp)"
chmod 600 "$ENV_FILE"
export -p | grep -vE '^declare -x (PATH|HOME|PWD|OLDPWD|SHLVL|HOSTNAME|USER|LOGNAME|SHELL|TMPDIR|TERM|MAIL|_|LD_[A-Z_]*|CUDA_[A-Z_]*|NVIDIA_[A-Z_]*|SLURM_[A-Z_]*|ENROOT_[A-Z_]*|XDG_[A-Z_]*|DBUS_[A-Z_]*)=' > "$ENV_FILE"
echo "export INFMAX_CONTAINER_WORKSPACE=/workspace RESULT_DIR=/workspace/results HOME=/ixhome" >> "$ENV_FILE"

srun --jobid="$JOB_ID" bash -c "umask 077; mkdir -p $WS $NODE_ROOT/hf $NODE_ROOT/uv $NODE_ROOT/home && tar -xz -C $WS" \
    < <(tar -cz -C "$GITHUB_WORKSPACE" --exclude=.git --exclude=./results .)
srun --jobid="$JOB_ID" bash -c "cat > $WS/.ix_env" < "$ENV_FILE"
rm -f "$ENV_FILE"

MOUNTS="-m $WS:/workspace -m $NODE_ROOT/hf:$HF_HUB_CACHE -m $NODE_ROOT/uv:$AIPERF_UV_CACHE_DIR -m $NODE_ROOT/home:/ixhome"
START="$ENROOT_ENV; enroot start --rw -e NVIDIA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

srun --jobid="$JOB_ID" --overlap bash -c "$START $DCGM_CSV_MOUNT $DCGM_CT dcgm-exporter -a :$DCGM_PORT $DCGM_EXTRA_ARGS" &
DCGM_SRUN_PID=$!
trap 'rc=$?; kill "$DCGM_SRUN_PID" 2>/dev/null || true; srun --jobid="$JOB_ID" --overlap rm -rf "$WS" 2>/dev/null || true; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

set +e
srun --jobid="$JOB_ID" --overlap bash -c "$START $MOUNTS $IMAGE_CT bash -c 'source /workspace/.ix_env && rm -f /workspace/.ix_env && cd /workspace && bash $BENCH_SCRIPT'"
BENCH_RC=$?
set -e

# Results: results/ plus whatever the recipe wrote at the workspace root
# (RESULT_FILENAME.json, eval outputs). Extract over the runner's checkout.
srun --jobid="$JOB_ID" --overlap bash -c "cd $WS && tar -cz \$(find . -maxdepth 1 -newer $WS/benchmarks ! -name . ! -name .ix_env -printf '%P\n')" \
    | tar -xz -C "$GITHUB_WORKSPACE" || echo "WARN: result copy-back failed" >&2

exit "$BENCH_RC"
