#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
library="$repo_root/benchmarks/benchmark_lib.sh"
remote_replay="$repo_root/benchmarks/single_node/agentic/_remote_replay.sh"
remote_launcher="$repo_root/runners/launch_remote.sh"

unset HF_TOKEN
source "$library"

tmp_home="$(mktemp -d)"
trap 'rm -rf "$tmp_home"' EXIT

assert_contains() {
    [[ "$1" == *"$2"* ]] || { echo "missing: $2" >&2; return 1; }
}

assert_absent() {
    [[ "$1" != *"$2"* ]] || { echo "unexpected: $2" >&2; return 1; }
}

launch_args_file="$tmp_home/launch-args"
docker() {
    printf '%s\n' "$*" > "$LAUNCH_ARGS_FILE"
}
export -f docker
LAUNCH_ARGS_FILE="$launch_args_file" GITHUB_WORKSPACE="$repo_root" \
    IMAGE=aiperf:test HF_HUB_CACHE=/tmp/hf-cache \
    FIXED_SCHEDULE=true MAX_CONTEXT_LENGTH=100000 \
    bash "$remote_launcher" >/dev/null 2>&1
launch_args="$(<"$launch_args_file")"
assert_contains "$launch_args" "-e FIXED_SCHEDULE"
assert_contains "$launch_args" "-e MAX_CONTEXT_LENGTH"
rm -f "$launch_args_file"

original_pythonpath="${PYTHONPATH:-}"
AIPERF_DIR="$repo_root/utils/aiperf-mooncake"
FIXED_SCHEDULE=true
aiperf() { :; }
install_agentic_deps
assert_contains ":${PYTHONPATH:-}:" ":$AIPERF_DIR/src:"
resolved_aiperf="$(python3 -c 'import aiperf; print(aiperf.__file__)')"
assert_contains "$resolved_aiperf" "$AIPERF_DIR/src/aiperf/"
unset -f aiperf
PYTHONPATH="$original_pythonpath"

curl() { return 0; }
timeout() {
    printf '%s\n' "$*" > "$TIMEOUT_ARGS_FILE"
}
python3() {
    if [[ "${1:-}" == "-c" ]]; then
        command python3 "$@"
    fi
}
aiperf() { :; }
export -f curl timeout python3 aiperf

run_runtime_check() {
    local fixed_schedule="$1" duration="$2" override="$3" expected="$4"
    local timeout_args_file="$tmp_home/timeout-$fixed_schedule-$duration-$override"
    local result_dir="$tmp_home/result-$fixed_schedule-$duration-$override"

    env AIPERF_MAX_RUNTIME="$override" \
        TIMEOUT_ARGS_FILE="$timeout_args_file" \
        INFMAX_CONTAINER_WORKSPACE="$repo_root" \
        MODEL=z-ai/glm-5.2 CONC=13 RESULT_DIR="$result_dir" \
        REMOTE_URL=https://replay.invalid REMOTE_API_KEY=test-secret \
        TOKENIZER=zai-org/GLM-5.2 INPUT_FILE="$repo_root/benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions" \
        CUSTOM_DATASET_TYPE=weka_trace FIXED_SCHEDULE="$fixed_schedule" \
        DURATION="$duration" bash "$remote_replay" >/dev/null 2>&1

    assert_contains "$(<"$timeout_args_file")" "-s TERM -k 60 $expected "
}

run_runtime_check true 3000 "" 4890
run_runtime_check true 60 "" 1950
run_runtime_check false 90 "" 2400
run_runtime_check true 3000 1234 1234

public_dataset_args_file="$tmp_home/timeout-public-dataset"
env AIPERF_MAX_RUNTIME=123 \
    TIMEOUT_ARGS_FILE="$public_dataset_args_file" \
    INFMAX_CONTAINER_WORKSPACE="$repo_root" \
    MODEL=z-ai/glm-5.2 CONC=2 RESULT_DIR="$tmp_home/result-public-dataset" \
    REMOTE_URL=https://replay.invalid REMOTE_API_KEY=test-secret \
    TOKENIZER=zai-org/GLM-5.2 PUBLIC_DATASET=weka_hf \
    HF_WEKA_REPO=semianalysisai/cc-traces-weka-062126 \
    CUSTOM_DATASET_TYPE=weka_trace FIXED_SCHEDULE=false \
    DURATION=60 bash "$remote_replay" >/dev/null 2>&1
public_dataset_args="$(<"$public_dataset_args_file")"
assert_contains "$public_dataset_args" "--public-dataset weka_hf"
assert_contains "$public_dataset_args" "--hf-weka-repo semianalysisai/cc-traces-weka-062126"
unset -f curl timeout python3 aiperf

MODEL="z-ai/glm-5.2"
CONC=999
REMOTE_URL="https://replay.invalid"
REMOTE_API_KEY="test-secret"
TOKENIZER="zai-org/GLM-5.2"
DURATION=60
INPUT_FILE="/workspace/benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions"
CUSTOM_DATASET_TYPE="weka_trace"
TRACE_SOURCE_FLAG="--input-file $INPUT_FILE --custom-dataset-type $CUSTOM_DATASET_TYPE"
FIXED_SCHEDULE=true
MAX_CONTEXT_LENGTH=100000

build_replay_cmd /tmp/replay-result
fixed_cmd="$REPLAY_CMD"

for required in \
    "aiperf profile" \
    "--url $REMOTE_URL" \
    "--endpoint /v1/chat/completions" \
    "--endpoint-type chat" \
    "--streaming" \
    "--model $MODEL" \
    "--api-key $REMOTE_API_KEY" \
    "--tokenizer $TOKENIZER" \
    "--tokenizer-trust-remote-code" \
    "--input-file $INPUT_FILE" \
    "--custom-dataset-type weka_trace" \
    "--fixed-schedule" \
    "--benchmark-duration 60" \
    "--extra-inputs ignore_eos:true" \
    "--random-seed 42" \
    "--slice-duration 1" \
    "--max-context-length 100000" \
    "--output-artifact-dir /tmp/replay-result/trace_replay"
do
    assert_contains "$fixed_cmd" "$required"
done

for forbidden in \
    "--scenario" \
    "--use-think-time-only" \
    "--benchmark-grace-period" \
    "--concurrency" \
    "--use-server-token-count" \
    "--failed-request-threshold" \
    "--trajectory-start" \
    "--num-dataset-entries" \
    "--unsafe-override" \
    "--cache-bust" \
    "--warmup"
do
    assert_absent "$fixed_cmd" "$forbidden"
done

MAX_CONTEXT_LENGTH=""
build_replay_cmd /tmp/replay-result
assert_absent "$REPLAY_CMD" "--max-context-length"

FIXED_SCHEDULE=false
build_replay_cmd /tmp/replay-result
assert_contains "$REPLAY_CMD" "--scenario inferencex-agentx-mvp"
assert_contains "$REPLAY_CMD" "--concurrency $CONC"
assert_absent "$REPLAY_CMD" "--fixed-schedule"
assert_absent "$REPLAY_CMD" "--failed-request-threshold"

HOME="$tmp_home"
AIPERF_DOCKER_IMAGE="aiperf:test"
build_docker_replay_args /tmp/replay-result
docker_args="${DOCKER_REPLAY_ARGS[*]}"
assert_contains "$docker_args" "-v $INPUT_FILE:$INPUT_FILE:ro"

! grep -q -- '--failed-request-threshold 0.05' "$library"
! grep -q -- 'report_failed_request_abort' "$library" "$remote_replay"

echo "replay command construction: PASS"
