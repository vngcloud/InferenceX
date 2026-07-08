#!/usr/bin/env bash
set -euo pipefail
set -x

export AIPERF_DIR="${AIPERF_DIR:-${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake}"

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL CONC RESULT_DIR REMOTE_URL
check_remote_endpoints

PUBLIC_DATASET="${PUBLIC_DATASET:-}"
if [[ "${CUSTOM_DATASET_TYPE:-}" == "weka_trace" && -z "${INPUT_FILE:-}" && -z "$PUBLIC_DATASET" ]]; then
    PUBLIC_DATASET="semianalysis_cc_traces_weka_with_subagents_060826"
fi

if [[ -n "${INPUT_FILE:-}" ]]; then
    if [[ ! -e "$INPUT_FILE" ]]; then
        echo "Error: trace input path not found: $INPUT_FILE (cwd=$(pwd))" >&2
        exit 1
    fi
    TRACE_SOURCE_FLAG="--input-file $INPUT_FILE"
    if [[ -n "${CUSTOM_DATASET_TYPE:-}" ]]; then
        TRACE_SOURCE_FLAG+=" --custom-dataset-type $CUSTOM_DATASET_TYPE"
    fi
elif [[ -n "$PUBLIC_DATASET" ]]; then
    TRACE_SOURCE_FLAG="--public-dataset $PUBLIC_DATASET"
else
    echo "Error: one of INPUT_FILE or PUBLIC_DATASET is required" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
install_agentic_deps
build_replay_cmd "$RESULT_DIR"
if [[ "$AIPERF_USE_DOCKER" == "true" ]]; then
    build_docker_replay_args "$RESULT_DIR"
fi

echo "${REPLAY_CMD/${REMOTE_API_KEY:-EMPTY}/<redacted>}" > "$RESULT_DIR/benchmark_command.txt"

set +x
# A remote-replay run once hung silently for ~16 min until the runner itself
# was killed, with no client-side timeout or exit-code check to catch it.
# AIPERF_MAX_RUNTIME bounds that: dataset configuration can take up to
# AIPERF_DATASET_CONFIGURATION_TIMEOUT (1800s, see build_replay_cmd) plus the
# benchmark duration itself, so the default here leaves headroom above that.
AIPERF_MAX_RUNTIME="${AIPERF_MAX_RUNTIME:-2400}"
# Use the short flags -s/-k rather than --signal/--kill-after: the pre-built
# full AIPerf image is distroless and ships busybox timeout, which only accepts
# `timeout [-s SIG] [-k KILL_SECS] SECS PROG`. GNU timeout accepts these too, so
# this is portable across both the full-image and pip-install paths.
if [[ "$AIPERF_USE_DOCKER" == "true" ]]; then
    timeout -s TERM -k 60 "$AIPERF_MAX_RUNTIME" "${DOCKER_REPLAY_ARGS[@]}" 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
else
    timeout -s TERM -k 60 "$AIPERF_MAX_RUNTIME" $REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
fi
replay_exit="${PIPESTATUS[0]}"
set -x

if [[ "$replay_exit" -eq 124 ]]; then
    echo "Error: aiperf exceeded AIPERF_MAX_RUNTIME=${AIPERF_MAX_RUNTIME}s against REMOTE_URL=$REMOTE_URL and was killed." >&2
elif [[ "$replay_exit" -ne 0 ]]; then
    echo "WARNING: aiperf exited with code $replay_exit; attempting result aggregation anyway." >&2
fi

report_failed_request_abort "$RESULT_DIR"

write_agentic_result_json "$RESULT_DIR"

python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
