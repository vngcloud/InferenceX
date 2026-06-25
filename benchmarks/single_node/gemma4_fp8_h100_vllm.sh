#!/usr/bin/env bash
set -euo pipefail
#
# Baseline agentic-replay launcher: RedHatAI/gemma-4-31B-it-FP8-dynamic (fp8, TP=1) on vLLM.
# No LMCache — paired with gemma4-lmcache_fp8_h100_vllm.sh to measure LMCache CPU KV-offload delta.
#
# Stack: vllm/vllm-openai:v0.23.0
#
# FP8 model: RedHatAI/gemma-4-31B-it-FP8-dynamic uses compressed-tensors quantization.
# vLLM auto-detects it from quantization_config.json — do NOT pass --quantization.
# --dtype bfloat16 sets the compute/activation dtype around the fp8 weights.
#
# Prefix-caching is enabled (--enable-prefix-caching) so prefix hit rates from
# server_metrics_export.json are comparable to the LMCache counterpart.

source "$(dirname "$0")/../benchmark_lib.sh"

export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"

check_env_vars \
    MODEL \
    TP \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-RedHatAI/gemma-4-31B-it-FP8-dynamic}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# An optional "#N" suffix on the input-file path replays only the first N records.
trace_limit=""
if [[ "$INPUT_FILE" == *"#"* ]]; then
    trace_limit="${INPUT_FILE##*#}"
    INPUT_FILE="${INPUT_FILE%#*}"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: trace input file not found: $INPUT_FILE (cwd=$(pwd))" >&2
    exit 1
fi

if [[ -n "$trace_limit" ]]; then
    head -n "$trace_limit" "$INPUT_FILE" > /workspace/_trace_subset.jsonl
    INPUT_FILE=/workspace/_trace_subset.jsonl
    echo "Subset trace to first $trace_limit records -> $INPUT_FILE"
fi

if [[ "${STRIP_TRACE_DELAYS:-}" == "true" || "${STRIP_TRACE_DELAYS:-}" == "1" ]]; then
    python3 -c '
import json, sys
with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("delay", None)
        fout.write(json.dumps(rec) + "\n")
' "$INPUT_FILE" /workspace/_trace_nodelay.jsonl
    INPUT_FILE=/workspace/_trace_nodelay.jsonl
    echo "Stripped per-turn delays for capacity sweep -> $INPUT_FILE"
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--served-model-name "$SERVED_MODEL_NAME" \
--tensor-parallel-size "$TP" \
--dtype bfloat16 \
--gpu-memory-utilization 0.90 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--enable-prefix-caching \
--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
set +x

STOP_ARGS=()
DURATION_OVERRIDE="${BENCHMARK_DURATION:-${DURATION:-}}"
if [[ -n "$DURATION_OVERRIDE" ]]; then
    STOP_ARGS+=(--benchmark-duration "$DURATION_OVERRIDE")
    echo "Duration-based replay of $INPUT_FILE: benchmark-duration=${DURATION_OVERRIDE}s at concurrency $CONC"
else
    if [[ -z "${REQUEST_COUNT:-}" ]]; then
        REQUEST_COUNT=$(grep -c . "$INPUT_FILE")
    fi
    STOP_ARGS+=(--request-count "$REQUEST_COUNT")
    echo "Replaying trace $INPUT_FILE: request-count=$REQUEST_COUNT at concurrency $CONC"
fi

REPLAY_ARGS=()
if [[ "${NO_FIXED_SCHEDULE:-true}" == "true" || "${NO_FIXED_SCHEDULE:-true}" == "1" ]]; then
    REPLAY_ARGS+=(--no-fixed-schedule)
fi
REPLAY_ARGS+=(--inter-turn-delay-cap-seconds "${INTER_TURN_DELAY_CAP_SECONDS:-60}")
REPLAY_ARGS+=(--dataset-sampling-strategy "${DATASET_SAMPLING_STRATEGY:-shuffle}")
REPLAY_ARGS+=(--warmup-request-count "${WARMUP_REQUEST_COUNT:-20}")
REPLAY_ARGS+=(--workers-max "${WORKERS_MAX:-200}")
REPLAY_ARGS+=(--benchmark-grace-period "${BENCHMARK_GRACE_PERIOD:-120}")
REPLAY_ARGS+=(--extra-inputs "ignore_eos:${IGNORE_EOS:-true}")
REPLAY_ARGS+=(--extra-inputs "temperature:${TEMPERATURE:-0}")
if [[ -n "${NUM_WARMUP_SESSIONS:-}" ]]; then
    REPLAY_ARGS+=(--num-warmup-sessions "$NUM_WARMUP_SESSIONS")
fi
if [[ -n "${GOODPUT:-}" ]]; then
    REPLAY_ARGS+=(--goodput "$GOODPUT")
fi
if [[ -n "${TOKENIZER:-}" ]]; then
    REPLAY_ARGS+=(--tokenizer "$TOKENIZER")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    "${STOP_ARGS[@]}" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    "${REPLAY_ARGS[@]}"

stop_gpu_monitor
set +x
