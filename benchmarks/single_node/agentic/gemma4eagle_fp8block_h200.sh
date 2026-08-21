#!/usr/bin/env bash
set -euo pipefail
set -x

# Gemma-4 31B FP8-block vLLM AGENTIC recipe, EAGLE3 speculative decoding,
# 2-way data parallel (tp1 x dp2 = 2 GPUs).
#
# Agentic-scenario sibling of the fixed-seq-len twin
# (../fixed_seq_len/gemma4eagle_fp8block_h200.sh): same checkpoint, image,
# runner, DP2 topology and EAGLE3 draft, but replays the SemiAnalysis Claude
# Code trace corpus via aiperf instead of synthetic fixed-length prompts. Real
# repeated coding context is where EAGLE3 acceptance is actually meaningful --
# the fixed-seq-len run only measured a random-token floor.
#
# TWO CONTEXT KNOBS, DELIBERATELY DECOUPLED:
#   * server window  = --max-model-len 262144 (gemma-4's native 256k, from the
#     model config's max_position_embeddings). Gives the engine runway so a
#     surviving trace whose context GROWS across agentic turns does not 4xx or
#     get truncated mid-session.
#   * workload cap   = MAX_MODEL_LEN=131072 exported before build_replay_cmd,
#     which becomes aiperf's --max-context-length. Traces whose input exceeds
#     128k are filtered OUT of the replay set at entry. So we admit <=128k
#     prompts but let each session breathe up to 256k.
# The two are separate flags that merely default to the same variable; setting
# the serve window as a literal and MAX_MODEL_LEN only for the replay keeps them
# independent (see benchmark_lib.sh build_replay_cmd --max-context-length).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR DURATION

# GPU-resident KV (kv-offloading: none in the master config); assert the matrix
# did not hand us an offload backend this recipe would silently ignore.
require_agentic_kv_offload_none

# gemma-4's model-prefix (gemma4eagle) routes to the 256k-capped corpus by
# default; pin it explicitly so the dataset choice is auditable and matches the
# 128k workload cap below.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k

# Mandatory agentic telemetry: aiperf scrapes vLLM's Prometheus /metrics and the
# DCGM exporter the runner launcher starts on :9400.
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

DP_SIZE=2
DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
# Agentic scheduler headroom convention: admit up to 2x the offered concurrency.
MAX_NUM_SEQS=$((2 * CONC))

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi

# EAGLE3 draft head lives in a separate repo; pull it into the HF cache so the
# --speculative-config "model" id resolves offline like the target does.
hf download "$DRAFT_MODEL"

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export PYTHONNOUSERSITE=1
# EAGLE3 draft + target compile on a 256k window can exceed the default startup
# window; give the engine room.
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# --max-model-len 262144 is the SERVER window (see header). Do NOT wire
# MAX_MODEL_LEN into it -- that variable is reserved for the replay cap below.
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP_SIZE" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 262144 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Workload cap: admit only traces whose input <= 128k. build_replay_cmd reads
# MAX_MODEL_LEN as aiperf --max-context-length. Server window stays 256k.
export MAX_MODEL_LEN=131072

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
