#!/usr/bin/env bash
#
# Agentic-replay smoke launcher: Qwen/Qwen3-4B-Instruct-2507 (bf16, TP=1) on vLLM (H200), driven by
# a recorded mooncake_trace JSONL through official AIPerf. The trace is replayed
# once; --request-count equals the dataset record count and isl/osl do not apply
# (the trace defines per-request lengths). See docs/AIPERF_INTEGRATION.md and
# docs/adr/0001-agentic-on-official-aiperf.md.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

# Capture the dataset identity from the original trace path before any
# subsetting/stripping rewrites $INPUT_FILE (benchmark_lib.sh helper).
capture_workload "$INPUT_FILE"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# An optional "#N" suffix on the input-file path replays only the first N
# records (a low-resource subset of a large committed trace).
trace_limit=""
if [[ "$INPUT_FILE" == *"#"* ]]; then
    trace_limit="${INPUT_FILE##*#}"
    INPUT_FILE="${INPUT_FILE%#*}"
fi

# The trace JSONL path is repo-relative; the container runs with cwd=/workspace.
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: trace input file not found: $INPUT_FILE (cwd=$(pwd))" >&2
    exit 1
fi

if [[ -n "$trace_limit" ]]; then
    head -n "$trace_limit" "$INPUT_FILE" > /workspace/_trace_subset.jsonl
    INPUT_FILE=/workspace/_trace_subset.jsonl
    echo "Subset trace to first $trace_limit records -> $INPUT_FILE"
fi

# Mode 1 (capacity sweep): strip the recorded per-turn `delay` field so the run
# is driven purely by --concurrency back-pressure with zero think-time. AIPerf
# 0.9.0 honors mooncake_trace inter-turn delays even under --no-fixed-schedule
# (concurrency uses the request-rate strategy, which sleeps meta.delay_ms), and
# has no CLI flag to ignore them — so we drop the field at the source.
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
--gpu-memory-utilization 0.85 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Stop condition. Duration-based smoke: when DURATION (the dispatch
# duration-override) is set, cap the run at that wall-clock and let AIPerf decide
# how many requests fit — the adapter then skips exact request-count validation,
# so overflow/errored turns (e.g. trace turns longer than --max-model-len) are
# tolerated. Otherwise fall back to a fixed request count: an explicit env
# override (Mode 1 resample) or one request per dataset record (single replay).
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

# Agentic-replay methodology — matches the canonical AIPerf command in the docs:
# closed-loop concurrency WITH think-time (sessions sleep their recorded inter-turn
# delay, capped so one idle session can't stall the run), shuffled session order,
# warmup, ignore_eos + temperature 0 for controlled/deterministic generation.
# Every value is env-overridable; STRIP_TRACE_DELAYS=true (above) flips this to the
# zero-think-time capacity-sweep variant, NO_FIXED_SCHEDULE=false to timestamp replay.
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
# Goodput is reporting-only (does not affect the run) and is computed offline from
# the raw artifact per ADR-0002; pass GOODPUT="metric:val ..." to label it inline.
if [[ -n "${GOODPUT:-}" ]]; then
    REPLAY_ARGS+=(--goodput "$GOODPUT")
fi
# Explicit tokenizer (HF id) when the served model name isn't a valid tokenizer
# source; unset => aiperf defaults to the served model (the standard flow).
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
