#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars GPU_COUNT

set -x

# Slurm path for hcm-rtx6000pro_01/_02 (on-node runners for the
# `hcm-rtx6000pro` partition: hoanq3-rtx6000x2-slurm-1/-2, 2x RTX PRO 6000
# Blackwell Max-Q each, on the research Slurm cluster). Same shape as
# runners/launch_hardware-hcm.sh (holding allocation + pyxis/enroot, no
# Docker on these boxes), adapted for this fleet:
#   - partition hcm-rtx6000pro; jobs run under the default account `ai`
#     (verified via sacct on probe jobs; no --account flag, unlike
#     hardware-hcm's account `dev`).
#   - No /mnt big disk here: squash cache, HF hub cache, and uv cache live
#     under /data/thanglq5 (984G ext4 per node, world-writable, node-local --
#     every mount below is local to the box the runner lives on, which is why
#     the allocation below pins --nodelist).
#   - 48 CPUs / ~236G RAM over 2 GPUs -> 23 CPUs and 110G per GPU. 23, not 24:
#     the runner daemon itself is a 1-CPU sbatch job on the same node (see
#     docs/knowledge.md 1x-sbatch-runner-daemon), so a full 2-GPU job at
#     24 CPUs/GPU would exceed the node's 48 CPUs and queue forever. One
#     runner per node = at most one GH bench job at a time per box.
# The serving-YAML path (SERVE_CMD, optional ROUTER_* for a router sidecar,
# STORAGE_* for a storage sidecar) mirrors launch_hardware-hcm.sh.
SLURM_PARTITION="hcm-rtx6000pro"
# Pin the allocation to this box's own Slurm node name (same rationale as
# launch_hardware-hcm.sh: every --container-mounts path below is local to the
# box the *runner* lives on). Hostname and Slurm NodeName are identical on
# this fleet; the map exists so an unexpected host fails loudly.
case "$(hostname -s)" in
  hoanq3-rtx6000x2-slurm-1) SLURM_NODELIST="hoanq3-rtx6000x2-slurm-1" ;;
  hoanq3-rtx6000x2-slurm-2) SLURM_NODELIST="hoanq3-rtx6000x2-slurm-2" ;;
  *)
    echo "ERROR: unrecognized hcm-rtx6000pro host $(hostname -s), add it to the map in $0" >&2
    exit 1
    ;;
esac
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/data/thanglq5/hf_hub_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/data/thanglq5/uv-cache}"

# pyxis shares the host netns by default (no --container-unshare=net) -> two
# concurrent allocations on this node can't both hardcode PORT=8888.
# benchmark-tmpl.yml sets PORT=8888 as a job-level default env var, so
# ${PORT:-...} never falls through to free_tcp_port() -- it's always
# already non-empty. Override unconditionally instead: this launcher must
# always pick its own free port, never trust an inherited default.
free_tcp_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}
# Dynamo parses DYN_SYSTEM_PORT as i16 (<= 32767); ephemeral ports (32768+) crash it.
free_tcp_port_below_32768() {
    python3 -c 'import random, socket
for p in random.sample(range(20000, 32768), 2000):
    s = socket.socket()
    try:
        s.bind(("", p)); print(p); break
    except OSError:
        pass
    finally:
        s.close()'
}

# enroot needs `docker://registry#path` for a non-default registry; auto-detect
# by whether the first path component looks like a host (has a dot/colon, or
# is localhost), same heuristic as launch_hardware-hcm.sh.
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

# Local /data, not /shared (NFS, 93% full on this cluster) -- much faster for
# large image imports. A runner's .env may set IX_SQUASH_DIR.
SQUASH_CACHE_DIR="${IX_SQUASH_DIR:-/data/thanglq5/containers}"
mkdir -p "$SQUASH_CACHE_DIR"

# GH Actions cancel can SIGKILL us before EXIT traps run, leaking the
# allocation; clear any stale job from this runner name before requesting a new one.
scancel --name="$RUNNER_NAME" 2>/dev/null || true

# These nodes ship a minimal Slurm client set (sbatch/srun/squeue/scancel/sinfo/sacct)
# with no salloc, so the holding allocation is a sleeping batch job instead of
# `salloc --no-shell`: srun --jobid steps attach to it exactly the same way.
# Every srun --jobid below pins --ntasks=1 --cpus-per-task --mem: the launcher
# runs inside the runner daemon's own sbatch allocation, and a bare srun
# inherits the allocation's CPU count as its task count (one task per CPU,
# each step-cgroup memory-capped small -> mksquashfs OOM; same trap as
# launch_h200-greennode-slurm-1x.sh, fix 3ecbd58dc). The main bench srun also
# takes --overlap so it shares CPUs with the already-running sidecar steps
# instead of queueing behind the CPUs they hold.
# --time=490, not hardware-hcm's 180: a first run on a fresh node downloads the
# model inside this allocation (gemma-4 FP8 ~33G at ~2MB/s measured on node 02;
# run 39107314182 was SIGTERMed by the 180-min limit mid-download), and 490
# stays under the workflow's 500-min job timeout so GitHub cancels first.
sbatch --partition="$SLURM_PARTITION" \
    --nodelist="$SLURM_NODELIST" \
    --gres=gpu:"$GPU_COUNT" \
    --cpus-per-task=$((GPU_COUNT * 23)) --mem=$((GPU_COUNT * 110))G \
    --time=490 --job-name="$RUNNER_NAME" \
    --output=/dev/null --error=/dev/null \
    --wrap "sleep $((490 * 60))"
JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: failed to resolve hcm-rtx6000pro Slurm allocation" >&2
    exit 1
fi
# sbatch returns as soon as the job is queued, unlike salloc which blocks until
# the allocation is granted; srun --jobid needs the job RUNNING. Wait like
# salloc would, and fail with the pending reason on timeout (e.g. another
# user's job holding the GPUs this job asked for).
SECONDS=0
until [[ "$(squeue --jobs "$JOB_ID" -h -o %T 2>/dev/null)" == "RUNNING" ]]; do
    if [[ -z "$(squeue --jobs "$JOB_ID" -h -o %T 2>/dev/null)" ]]; then
        echo "ERROR: allocation $JOB_ID vanished before RUNNING" >&2
        exit 1
    fi
    if (( SECONDS > 600 )); then
        echo "ERROR: allocation $JOB_ID not RUNNING after 600s, reason: $(squeue --jobs "$JOB_ID" -h -o %r)" >&2
        scancel "$JOB_ID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

squash_path() {
    echo "${SQUASH_CACHE_DIR}/$(echo "$1" | sed 's/[\/:@#]/_/g').sqsh"
}
# Import <image> into the shared squash cache once per box (flock: concurrent jobs).
# Private registries (vcr.vngcloud.vn) authenticate through the runner user's
# ~/.config/enroot/.credentials (HOME=/data/thanglq5 on this fleet).
import_squash() {
    local image="$1" squash
    squash="$(squash_path "$image")"
    srun --jobid="$JOB_ID" --ntasks=1 --cpus-per-task=16 --mem=32G bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
        mkdir -p \$ENROOT_CACHE_PATH
        export TMPDIR=/data/thanglq5/tmp
        mkdir -p \$TMPDIR
        exec 9>\"${squash}.lock\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $squash'; exit 1; }
        if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$squash\"
            enroot import -o \"$squash\" $(enroot_uri_for_image "$image")
        fi
    " >&2
}
SQUASH_FILE="$(squash_path "$IMAGE")"
import_squash "$IMAGE"

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "vllm" ]] && printf '' || printf "_%s" "$FRAMEWORK")
case "$SPEC_DECODING" in
  mtp) SPEC_SUFFIX="_mtp" ;;
  draft_model) SPEC_SUFFIX="_specdec" ;;
  *) SPEC_SUFFIX="" ;;
esac
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_rtx6000pro${FRAMEWORK_SUFFIX}"
BENCH_SCRIPT="${BENCH_BASE}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    BENCH_SCRIPT="${BENCH_BASE}.sh"
fi

# Serving-YAML path: one generic recipe; the arm lives in SERVE_CMD & co.
if [[ -n "${SERVE_CMD:-}" ]]; then
    BENCH_SCRIPT="benchmarks/single_node/agentic/serve_generic.sh"
fi

# DCGM sidecar via srun --overlap in the same allocation. Own port, same
# reason as free_tcp_port above.
DCGM_PORT="$(free_tcp_port)"
DCGM_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04"
DCGM_SQUASH="$(squash_path "$DCGM_IMAGE")"
import_squash "$DCGM_IMAGE"

GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"
DCGM_MOUNT_ARG=""
DCGM_EXTRA_ARGS=""
if [[ -f "$GPU_METRICS_CSV" ]]; then
    DCGM_MOUNT_ARG=",$GITHUB_WORKSPACE/$GPU_METRICS_CSV:/etc/dcgm-exporter/custom.csv:ro"
    DCGM_EXTRA_ARGS="-f /etc/dcgm-exporter/custom.csv"
fi

srun --jobid="$JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 --mem=2G \
    --container-image="$DCGM_SQUASH" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace${DCGM_MOUNT_ARG}" \
    --no-container-mount-home \
    --container-remap-root \
    --no-container-entrypoint \
    dcgm-exporter -a ":$DCGM_PORT" $DCGM_EXTRA_ARGS &
DCGM_SRUN_PID=$!

# Storage sidecar (serving YAML with a storage: block; Mooncake master for HiCache L3).
# Non-GPU process: NVIDIA_VISIBLE_DEVICES=void skips the nvidia hook. Starts BEFORE the
# router / engine because the engine's HiCache backend connects to it at init. The engine
# env references its ports via {{storage_master_port}} / {{storage_meta_port}}
# (expanded by expand_storage_placeholders in benchmark_lib.sh).
STORAGE_SRUN_PID=""
if [[ -n "${STORAGE_CMD:-}" ]]; then
    export STORAGE_MASTER_PORT="$(free_tcp_port)"
    export STORAGE_META_PORT="$(free_tcp_port)"
    mkdir -p "$GITHUB_WORKSPACE/results"
    rm -f "$GITHUB_WORKSPACE/results/.storage_exited"
    STORAGE_SQUASH="$(squash_path "$STORAGE_IMAGE")"
    import_squash "$STORAGE_IMAGE"
    STORAGE_EXPANDED_CMD="${STORAGE_CMD//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    STORAGE_EXPANDED_CMD="${STORAGE_EXPANDED_CMD//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    (
        rc=0
        srun --jobid="$JOB_ID" --overlap --ntasks=1 --cpus-per-task=4 --mem=8G \
            --container-image="$STORAGE_SQUASH" \
            --container-mounts="$GITHUB_WORKSPACE:/workspace" \
            --no-container-mount-home \
            --container-remap-root \
            --container-writable \
            --container-workdir=/workspace/ \
            --no-container-entrypoint \
            --export=ALL,NVIDIA_VISIBLE_DEVICES=void,STORAGE_MASTER_PORT,STORAGE_META_PORT,STORAGE_ENV \
            bash -c "
                set -eo pipefail
                if [[ -n \"\${STORAGE_ENV:-}\" ]]; then
                    while IFS= read -r assignment; do
                        [[ -z \"\$assignment\" ]] && continue
                        assignment=\"\${assignment//\\{\\{storage_master_port\\}\\}/\$STORAGE_MASTER_PORT}\"
                        assignment=\"\${assignment//\\{\\{storage_meta_port\\}\\}/\$STORAGE_META_PORT}\"
                        export \"\$assignment\"
                    done <<< \"\$STORAGE_ENV\"
                fi
                exec $STORAGE_EXPANDED_CMD
            " \
            > "$GITHUB_WORKSPACE/results/storage.log" 2>&1 || rc=$?
        echo "exit $rc" > "$GITHUB_WORKSPACE/results/.storage_exited"
    ) &
    STORAGE_SRUN_PID=$!
    # Wait for the master's TCP port to open before releasing the engine. Fail fast if
    # the sidecar dies before readiness, otherwise the engine comes up without an L3
    # backend and the whole bench measures the wrong thing. Bash /dev/tcp opens from
    # the launcher's host netns (pyxis shares netns with the srun step, same localhost).
    SECONDS=0
    until (exec 9<>/dev/tcp/127.0.0.1/"$STORAGE_MASTER_PORT") 2>/dev/null; do
        exec 9>&- || true
        if [[ -f "$GITHUB_WORKSPACE/results/.storage_exited" ]]; then
            echo "ERROR: storage sidecar exited before readiness; see results/storage.log" >&2
            exit 1
        fi
        if (( SECONDS > 300 )); then
            echo "ERROR: storage sidecar did not open port $STORAGE_MASTER_PORT within 300s" >&2
            exit 1
        fi
        sleep 2
    done
    exec 9>&- || true
fi

# Router sidecar (serving YAML with a router: block, spec 2026-10-13 §4, 2026-10-14 §3).
#   dynamo profile: a Dynamo frontend on $PORT; frontend and engine find each other through
#   Dynamo's file discovery in the shared workspace.
#   generic profile: any router on $PORT; the engine listens on $ENGINE_PORT and the router
#   command reaches it as {{engine_url}} (expanded by serve_router.sh).
# pyxis shares the host netns, so everything talks over localhost.
ROUTER_SRUN_PID=""
if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/workspace/results/.dynkv DYN_SYSTEM_ENABLED=true
        # engine/bench only: exported after the router srun forks so the frontend does not inherit it
        DYN_SYSTEM_PORT="$(free_tcp_port_below_32768)"
    else
        export ENGINE_PORT="$(free_tcp_port)"
        # Serving-YAML replicas (spec 2026-10-16): one engine port per replica; the first is ENGINE_PORT.
        # ponytail: free ports are picked one by one (bind+close); a clash is possible but has not happened
        # with the single-port path either. Reserve them together if it ever does.
        ENGINE_PORTS="$ENGINE_PORT"
        for ((replica = 1; replica < ${REPLICAS:-1}; replica++)); do
            ENGINE_PORTS+=" $(free_tcp_port)"
        done
        export ENGINE_PORTS
    fi
    mkdir -p "$GITHUB_WORKSPACE/results"
    # A previous run's registrations would show up as ghost workers.
    rm -rf "$GITHUB_WORKSPACE/results/.dynkv" "$GITHUB_WORKSPACE/results/.router_exited"
    ROUTER_SQUASH="$(squash_path "$ROUTER_IMAGE")"
    import_squash "$ROUTER_IMAGE"
    (
        rc=0
        srun --jobid="$JOB_ID" --overlap --ntasks=1 --cpus-per-task=4 --mem=8G \
            --container-image="$ROUTER_SQUASH" \
            --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
            --no-container-mount-home \
            --container-remap-root \
            --container-writable \
            --container-workdir=/workspace/ \
            --no-container-entrypoint --export=ALL,PORT="$PORT" \
            bash benchmarks/single_node/agentic/serve_router.sh \
            > "$GITHUB_WORKSPACE/results/router.log" 2>&1 || rc=$?
        # serve_generic.sh's readiness wait fails fast on this marker.
        echo "exit $rc" > "$GITHUB_WORKSPACE/results/.router_exited"
    ) &
    ROUTER_SRUN_PID=$!
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_SYSTEM_PORT
    fi
fi
trap 'rc=$?; kill "$DCGM_SRUN_PID" $ROUTER_SRUN_PID $STORAGE_SRUN_PID 2>/dev/null || true; rm -rf "$GITHUB_WORKSPACE/results/.dynkv"; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM

export AIPERF_GPU_TELEMETRY_URL="http://localhost:${DCGM_PORT}/metrics"
if [[ -n "$DCGM_EXTRA_ARGS" ]]; then
    # aiperf runs in the bench container (only $GITHUB_WORKSPACE:/workspace mounted), not the DCGM sidecar: /etc/dcgm-exporter/custom.csv does not exist there.
    export AIPERF_GPU_TELEMETRY_METRICS_CSV="/workspace/$GPU_METRICS_CSV"
fi

# --container-writable: enroot's squashfs mount is read-only by default,
# unlike docker's writable-overlay default. Various tools baked into these
# images write into their own rootfs at runtime with no override available
# (triton -> /root/.triton, vllm -> /root/.cache/vllm/*, etc.).
srun --jobid="$JOB_ID" --overlap --ntasks=1 \
    --cpus-per-task=$((GPU_COUNT * 23 - 10)) --mem=$((GPU_COUNT * 90))G \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_UV_CACHE_DIR:$AIPERF_UV_CACHE_DIR" \
    --no-container-mount-home \
    --container-remap-root \
    --container-writable \
    --container-workdir=/workspace/ \
    --no-container-entrypoint --export=ALL,PORT="$PORT",AIPERF_GPU_TELEMETRY_URL,AIPERF_GPU_TELEMETRY_METRICS_CSV,STORAGE_MASTER_PORT,STORAGE_META_PORT \
    bash "$BENCH_SCRIPT"

# scancel ends the router step, so the subshell writes its .router_exited
# marker; killing it here would race (and lose) that write. The EXIT trap's
# kill still covers abrupt exits, where nobody reads the marker.
kill "$DCGM_SRUN_PID" 2>/dev/null || true
scancel "$JOB_ID"
wait 2>/dev/null || true
rm -rf "$GITHUB_WORKSPACE/results/.dynkv"
