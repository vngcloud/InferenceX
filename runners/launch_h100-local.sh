#!/usr/bin/env bash
set -euo pipefail

# Local H100 Docker launcher for PR-like smoke runs.
# Run from a synced InferenceX workspace on h100:
#   cd /mnt/users/thanglq5/InferenceX
#   bash runners/launch_h100-local.sh
#
# This mirrors benchmark-tmpl.yml -> runners/launch_*.sh as closely as practical
# without registering a GitHub self-hosted runner.

export GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"

export RUNNER_NAME="${RUNNER_NAME:-h100-local_0}"
export RUNNER_TYPE="${RUNNER_TYPE:-h100-local}"

export MODEL="${MODEL:-/mnt/models/google/gemma-4-31B-it}"
export MODEL_PREFIX="${MODEL_PREFIX:-gemma4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-31B-it}"
export IMAGE="${IMAGE:-vllm/vllm-openai:v0.21.0}"
export FRAMEWORK="${FRAMEWORK:-vllm}"
export PRECISION="${PRECISION:-bf16}"
export EXP_NAME="${EXP_NAME:-gemma4}"

export TP="${TP:-2}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export CONC="${CONC:-4}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export DISAGG="${DISAGG:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-fixed-seq-len}"
export SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-}"
export PORT="${PORT:-8888}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
export BENCHMARK_CLIENT="${BENCHMARK_CLIENT:-inferencex_native}"

if [[ -z "${RESULT_FILENAME:-}" ]]; then
    client_suffix=""
    if [[ "$BENCHMARK_CLIENT" == "aiperf" ]]; then
        client_suffix="_aiperf"
    fi
    export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}${client_suffix}_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_${RUNNER_NAME}"
fi

if [[ -z "${HF_HUB_CACHE_MOUNT:-}" ]]; then
    if [[ -d /mnt/hf_hub_cache ]]; then
        HF_HUB_CACHE_MOUNT=/mnt/hf_hub_cache
    elif [[ -d /home/ubuntu/hf_hub_cache ]]; then
        HF_HUB_CACHE_MOUNT=/home/ubuntu/hf_hub_cache
    else
        HF_HUB_CACHE_MOUNT=/mnt/hf_hub_cache
    fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache/}"

BENCHMARK_SCRIPT="${LOCAL_BENCHMARK_SCRIPT:-${EXP_NAME%%_*}_${PRECISION}_h100.sh}"
BENCHMARK_SCRIPT_PATH="benchmarks/single_node/${SCENARIO_SUBDIR}${BENCHMARK_SCRIPT}"

if [[ ! -f "$GITHUB_WORKSPACE/$BENCHMARK_SCRIPT_PATH" ]]; then
    echo "Benchmark script not found: $GITHUB_WORKSPACE/$BENCHMARK_SCRIPT_PATH" >&2
    echo "Set LOCAL_BENCHMARK_SCRIPT=<script.sh> if the default mapping is wrong." >&2
    exit 1
fi

mkdir -p "$GITHUB_WORKSPACE"

docker_mounts=(
    -v "$GITHUB_WORKSPACE:/workspace"
)

if [[ -d /mnt/models ]]; then
    docker_mounts+=(-v /mnt/models:/mnt/models)
fi

if [[ -d "$HF_HUB_CACHE_MOUNT" ]]; then
    docker_mounts+=(-v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE")
fi

if [[ "$BENCHMARK_CLIENT" == "aiperf" ]]; then
    # Serving images usually do not ship AIPerf. ensure_aiperf (benchmark_lib.sh)
    # installs it into an isolated in-container venv at runtime: from
    # AIPERF_SOURCE_DIR if mounted, else from PyPI (aiperf==$AIPERF_VERSION).
    # For local runs we auto-mount a source checkout when present so dev/offline
    # works without a network. Set AIPERF_FORCE_PYPI=1 to skip the mount and
    # exercise the PyPI path that real CI uses.
    if [[ "${AIPERF_FORCE_PYPI:-}" == "1" ]]; then
        echo "AIPERF_FORCE_PYPI=1: skipping source mount; ensure_aiperf will install aiperf==${AIPERF_VERSION:-0.9.0} from PyPI." >&2
        unset AIPERF_SOURCE_DIR
    else
        if [[ -z "${AIPERF_SOURCE_DIR:-}" ]]; then
            if [[ -f "$GITHUB_WORKSPACE/utils/aiperf/pyproject.toml" ]]; then
                AIPERF_SOURCE_DIR="$GITHUB_WORKSPACE/utils/aiperf"
            elif [[ -d "$GITHUB_WORKSPACE/../aiperf" ]]; then
                AIPERF_SOURCE_DIR="$GITHUB_WORKSPACE/../aiperf"
            elif [[ -d "/mnt/users/thanglq5/aiperf" ]]; then
                AIPERF_SOURCE_DIR="/mnt/users/thanglq5/aiperf"
            fi
        fi

        if [[ -n "${AIPERF_SOURCE_DIR:-}" && -f "$AIPERF_SOURCE_DIR/pyproject.toml" ]]; then
            docker_mounts+=(-v "$AIPERF_SOURCE_DIR:/aiperf")
            export AIPERF_SOURCE_DIR=/aiperf
        else
            # Not fatal: ensure_aiperf falls back to PyPI inside the container.
            echo "AIPerf source not found on host; ensure_aiperf will install aiperf==${AIPERF_VERSION:-0.9.0} from PyPI inside the container." >&2
            echo "Set AIPERF_SOURCE_DIR=/path/to/aiperf (or AIPERF_FORCE_PYPI=1) to control this." >&2
            unset AIPERF_SOURCE_DIR
        fi
    fi

    export AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-/workspace/.aiperf_mmap_cache}"
fi

docker_env=(
    -e HF_TOKEN
    -e HF_HUB_CACHE
    -e GITHUB_WORKSPACE=/workspace
    -e INFMAX_CONTAINER_WORKSPACE=/workspace
    -e MODEL
    -e MODEL_PREFIX
    -e SERVED_MODEL_NAME
    -e IMAGE
    -e FRAMEWORK
    -e PRECISION
    -e EXP_NAME
    -e TP
    -e EP_SIZE
    -e DP_ATTENTION
    -e CONC
    -e ISL
    -e OSL
    -e MAX_MODEL_LEN
    -e RANDOM_RANGE_RATIO
    -e SPEC_DECODING
    -e DISAGG
    -e RUN_EVAL
    -e EVAL_ONLY
    -e RUNNER_NAME
    -e RUNNER_TYPE
    -e RESULT_FILENAME
    -e RESULT_DIR
    -e PORT
    -e SCENARIO_TYPE
    -e SCENARIO_SUBDIR
    -e BENCHMARK_CLIENT
    -e PYTHONDONTWRITEBYTECODE
    -e PYTHONPYCACHEPREFIX
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
)

if [[ "$BENCHMARK_CLIENT" == "aiperf" ]]; then
    docker_env+=(
        -e AIPERF_SOURCE_DIR
        -e AIPERF_VERSION
        -e AIPERF_DATASET_MMAP_CACHE_DIR
        -e RANDOM_SEED="${RANDOM_SEED:-0}"
    )
fi

container_name="${CONTAINER_NAME:-inferencex-${RESULT_FILENAME}}"

set -x
docker run --rm --network=host --name "$container_name" \
    --runtime=nvidia --gpus=all --ipc=host --privileged \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    "${docker_mounts[@]}" \
    -w /workspace \
    "${docker_env[@]}" \
    --entrypoint=/bin/bash \
    "$IMAGE" \
    "$BENCHMARK_SCRIPT_PATH"
set +x

if [[ "$EVAL_ONLY" != "true" && ! -f "$GITHUB_WORKSPACE/${RESULT_FILENAME}.json" ]]; then
    echo "Run finished but result file is missing: $GITHUB_WORKSPACE/${RESULT_FILENAME}.json" >&2
    exit 1
fi

echo "Result: $GITHUB_WORKSPACE/${RESULT_FILENAME}.json"
echo "Server log: $GITHUB_WORKSPACE/server.log"
echo "GPU metrics: $GITHUB_WORKSPACE/gpu_metrics.csv"
if [[ "$BENCHMARK_CLIENT" == "aiperf" ]]; then
    echo "AIPerf artifacts: $GITHUB_WORKSPACE/${RESULT_FILENAME}_aiperf"
fi
