#!/usr/bin/env bash
#
# Agentic-replay Mode 1 launcher: MiniMaxAI/MiniMax-M2.5 on single-node SGLang
# prefill/decode (PD) disaggregation. The GreenNode H200 runner exposes all 8
# GPUs to this container; this launcher splits them into a 4-GPU prefill worker
# (GPUs 0-3) and a 4-GPU decode worker (GPUs 4-7), fronts them with the SGLang
# PD router, and replays a recorded mooncake_trace JSONL through official AIPerf.
#
# Serving knobs mirror the colocated launcher (minimaxm2.5-agentic_fp8_h100_sglang.sh)
# so disagg vs non-disagg TTFT/TPOT are directly comparable.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMaxAI/MiniMax-M2.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-147456}"

# Per-worker parallelism: default each PD worker to the sweep's TP/EP. For a
# 4P4D split on 8xH200 the config sets TP=4 EP=4 -> 4 GPUs prefill + 4 decode.
PREFILL_TP="${PREFILL_TP:-$TP}"
PREFILL_EP="${PREFILL_EP:-$EP_SIZE}"
DECODE_TP="${DECODE_TP:-$TP}"
DECODE_EP="${DECODE_EP:-$EP_SIZE}"

# GPU isolation: each launch_server process sees only its own GPUs, so both
# workers index them as 0..N-1 (no --base-gpu-id needed).
PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-4,5,6,7}"

PORT="${PORT:-8888}"                                  # router (client-facing)
PREFILL_PORT="${PREFILL_PORT:-8889}"
DECODE_PORT="${DECODE_PORT:-8890}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8998}"
DISAGG_TRANSFER_BACKEND="${DISAGG_TRANSFER_BACKEND:-mooncake}"

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

# Mode 1 (capacity sweep): strip recorded per-turn delays so the run is driven
# purely by --concurrency back-pressure with zero think-time.
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

PREFILL_LOG=/workspace/server-prefill.log
DECODE_LOG=/workspace/server-decode.log
ROUTER_LOG=/workspace/server.log

PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""

cleanup() {
    set +e
    stop_gpu_monitor
    for pid in "$ROUTER_PID" "$DECODE_PID" "$PREFILL_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$ROUTER_PID" "$DECODE_PID" "$PREFILL_PID"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

start_gpu_monitor

export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

# Serving args shared by both PD workers; identical to the colocated launcher.
COMMON_SGLANG_ARGS=(
  --model-path "$MODEL"
  --host 0.0.0.0
  --served-model-name "$SERVED_MODEL_NAME"
  --context-length "$MAX_MODEL_LEN"
  --tool-call-parser minimax-m2
  --reasoning-parser minimax
  --mem-fraction-static 0.85
  --page-size 64
  --chunked-prefill-size 16384
  --trust-remote-code
  --disaggregation-transfer-backend "$DISAGG_TRANSFER_BACKEND"
)

set -x
CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_VISIBLE_DEVICES" python3 -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --port "$PREFILL_PORT" \
  --tp "$PREFILL_TP" \
  --ep "$PREFILL_EP" \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$PREFILL_BOOTSTRAP_PORT" > "$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!

CUDA_VISIBLE_DEVICES="$DECODE_CUDA_VISIBLE_DEVICES" python3 -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --port "$DECODE_PORT" \
  --tp "$DECODE_TP" \
  --ep "$DECODE_EP" \
  --disaggregation-mode decode > "$DECODE_LOG" 2>&1 &
DECODE_PID=$!
set +x

wait_for_server_ready --port "$PREFILL_PORT" --server-log "$PREFILL_LOG" --server-pid "$PREFILL_PID"
wait_for_server_ready --port "$DECODE_PORT" --server-log "$DECODE_LOG" --server-pid "$DECODE_PID"

# PD router: no GPUs; routes one prefill + one decode worker. The trailing
# integer on --prefill is the prefill bootstrap port.
set -x
CUDA_VISIBLE_DEVICES="" python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy cache_aware \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --host 0.0.0.0 \
  --port "$PORT" > "$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
set +x

wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"

# Request count: explicit env override takes precedence; otherwise replay the
# trace exactly once, one request per dataset record.
if [[ -z "${REQUEST_COUNT:-}" ]]; then
    REQUEST_COUNT=$(grep -c . "$INPUT_FILE")
fi
echo "Replaying trace $INPUT_FILE through PD router: request-count=$REQUEST_COUNT at concurrency $CONC"

MODE1_ARGS=()
if [[ "${NO_FIXED_SCHEDULE:-}" == "true" || "${NO_FIXED_SCHEDULE:-}" == "1" ]]; then
    MODE1_ARGS+=(--no-fixed-schedule)
fi
if [[ -n "${NUM_WARMUP_SESSIONS:-}" ]]; then
    MODE1_ARGS+=(--num-warmup-sessions "$NUM_WARMUP_SESSIONS")
fi

run_client_benchmark \
    --model "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --concurrency "$CONC" \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type "$CUSTOM_DATASET_TYPE" \
    --request-count "$REQUEST_COUNT" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$ROUTER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    --extra-inputs ignore_eos:true \
    "${MODE1_ARGS[@]}"

trap - EXIT
cleanup
set +x
