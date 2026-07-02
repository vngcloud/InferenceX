#!/usr/bin/env bash
set -euo pipefail
set -x

export AIPERF_DIR="${AIPERF_DIR:-${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake}"

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL CONC RESULT_DIR REMOTE_URL

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

echo "${REPLAY_CMD/${REMOTE_API_KEY:-EMPTY}/<redacted>}" > "$RESULT_DIR/benchmark_command.txt"

set +x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set -x

write_agentic_result_json "$RESULT_DIR"

python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
