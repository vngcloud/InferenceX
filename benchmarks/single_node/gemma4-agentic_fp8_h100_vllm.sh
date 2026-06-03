#!/usr/bin/env bash
#
# Agentic-replay launcher: google/gemma-4-31B-it (fp8, TP=2) on vLLM, driven by a
# recorded mooncake_trace JSONL through official AIPerf. Engine-vs-engine sibling of
# gemma4-agentic_fp8_h100_sglang.sh on the SAME model/hardware/trace (2x H100, TP=2,
# cache ON). The trace is replayed once; --request-count equals the dataset record
# count and isl/osl do not apply (the trace defines per-request lengths).
#
# fp8 recipe matches the existing gemma4_fp8_h100.sh bench: on-the-fly --quantization
# fp8 over google/gemma-4-31B-it + fp8_e4m3 KV cache. vLLM V1 keeps automatic prefix
# caching ON by default (NOT disabled here) — that is the "cache ON" requirement.
# See docs/AIPERF_INTEGRATION.md and docs/adr/0001-agentic-on-official-aiperf.md.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    MAX_MODEL_LEN \
    INPUT_FILE \
    CUSTOM_DATASET_TYPE \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-31B-it}"
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

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--served-model-name "$SERVED_MODEL_NAME" \
--tensor-parallel-size "$TP" \
--quantization fp8 \
--kv-cache-dtype fp8_e4m3 \
--gpu-memory-utilization 0.9 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Replay the trace exactly once: one request per dataset record.
REQUEST_COUNT=$(grep -c . "$INPUT_FILE")
echo "Replaying trace $INPUT_FILE: $REQUEST_COUNT records at concurrency $CONC"

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
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}"

stop_gpu_monitor
set +x
