#!/usr/bin/env bash
set -euo pipefail
set -x

# Gemma-4 31B FP8-block SGLang AGENTIC recipe, NGRAM (model-free) speculative
# decoding, 2-way data parallel (tp1 x dp2 = 2 GPUs).
#
# Agentic-scenario sibling of the fixed-seq-len twin
# (../fixed_seq_len/gemma4ngram_fp8block_h200_sglang.sh): same checkpoint,
# image, runner, DP2 topology and NGRAM speculation, but replays the
# SemiAnalysis Claude Code trace corpus via aiperf. NGRAM feeds on repeated
# n-grams, so real coding traces (heavy repeated context) are exactly where it
# pays off -- the fixed-seq-len run only saw a degenerate-output floor.
#
# NGRAM is used here (not EAGLE3) for the same reason as the fixed-seq-len twin:
# gemma-4 EAGLE3 is not in mainline SGLang, only the ThoughtWorks fork, so it
# cannot run on lmsysorg/sglang:v0.5.16-cu130. Cross-engine numbers are thus not
# apples-to-apples; the comparison is spec-on vs spec-off within each engine.
#
# TWO CONTEXT KNOBS, DELIBERATELY DECOUPLED (see the vLLM twin's header):
#   * server window = --context-length 262144 (gemma-4 native 256k) + a
#     deliberate --allow-auto-truncate so a session growing past 256k truncates
#     rather than erroring.
#   * workload cap  = MAX_MODEL_LEN=131072 exported before build_replay_cmd ->
#     aiperf --max-context-length; traces with input > 128k are filtered out.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR DURATION

require_agentic_kv_offload_none

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

DP_SIZE=2
# Agentic scheduler headroom convention: admit up to 2x the offered concurrency.
MAX_RUNNING_REQUESTS=$((2 * CONC))

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

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1
export NCCL_P2P_LEVEL=NVL

# NGRAM disables the overlap scheduler and mixed chunked prefill (documented);
# it is compatible with classic --dp (only --enable-dp-attention is forbidden).
# --context-length 262144 is the SERVER window; MAX_MODEL_LEN (the replay cap)
# is set only after startup. The ngram-* knobs are SGLang's documented defaults,
# pinned so the sweep is reproducible if defaults change.
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --dp "$DP_SIZE"
    --mem-fraction-static 0.88
    --kv-cache-dtype fp8_e4m3
    --context-length 262144
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --chunked-prefill-size 8192
    --allow-auto-truncate
    --enable-metrics
    --enable-cache-report
    --speculative-algorithm NGRAM
    --speculative-num-draft-tokens 8
    --speculative-ngram-min-bfs-breadth 1
    --speculative-ngram-max-bfs-breadth 10
    --speculative-ngram-match-type BFS
    --speculative-ngram-max-trie-depth 18
    --speculative-ngram-capacity 10000000
)

# See gemma4_fp8block_h200_sglang.sh: SGLang keeps its own parser registry with
# its own spellings, so probe rather than fail server startup outright.
if python3 -m sglang.launch_server --help 2>&1 | grep -q gemma4; then
    SGLANG_CMD+=(--tool-call-parser gemma4 --reasoning-parser gemma4)
else
    echo "NOTE: no 'gemma4' parser registered in this sglang image; serving without tool/reasoning parsers."
fi

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Workload cap: admit only traces whose input <= 128k (aiperf --max-context-length).
export MAX_MODEL_LEN=131072

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
