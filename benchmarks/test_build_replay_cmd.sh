#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
library="$repo_root/benchmarks/benchmark_lib.sh"
remote_replay="$repo_root/benchmarks/single_node/agentic/_remote_replay.sh"

unset HF_TOKEN
source "$library"

assert_contains() {
    [[ "$1" == *"$2"* ]] || { echo "missing: $2" >&2; return 1; }
}

assert_absent() {
    [[ "$1" != *"$2"* ]] || { echo "unexpected: $2" >&2; return 1; }
}

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

tmp_home="$(mktemp -d)"
trap 'rm -rf "$tmp_home"' EXIT
HOME="$tmp_home"
AIPERF_DOCKER_IMAGE="aiperf:test"
build_docker_replay_args /tmp/replay-result
docker_args="${DOCKER_REPLAY_ARGS[*]}"
assert_contains "$docker_args" "-v $INPUT_FILE:$INPUT_FILE:ro"

! grep -q -- '--failed-request-threshold 0.05' "$library"
! grep -q -- 'report_failed_request_abort' "$library" "$remote_replay"

echo "replay command construction: PASS"
