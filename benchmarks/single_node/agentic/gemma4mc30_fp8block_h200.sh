#!/usr/bin/env bash
set -euo pipefail
set -x

# Gemma-4 31B FP8-block vLLM MOONCAKE-TRACE recipe, EAGLE3 speculative
# decoding, 2-way data parallel (tp1 x dp2 = 2 GPUs).
#
# Same server topology as ../fixed_seq_len/gemma4eagle_fp8block_h200.sh and
# ./gemma4eagle_fp8block_h200.sh (Actions run 32392516834): two DP replicas of
# the FP8-block checkpoint behind one --data-parallel-size 2 engine, each
# verifying RedHat's EAGLE3 speculator. The only thing that changes is the
# WORKLOAD.
#
# WORKLOAD: utils/agentic/datasets/traces/gemma4_prod_cache30_910.jsonl -- 910
# single-turn requests fitted to the measured production ISL distribution
# (mean 27,755 / p50 34,919 / p90 42,244 / max 44,220), carrying `hash_ids`
# tiled at aiperf's 512-token mooncake block size so that 30% of all prefill
# tokens are a re-sent shared prefix. That 30% is an INFINITE-CACHE upper
# bound: what the engine actually hits depends on how many of those blocks
# survive vLLM's LRU eviction at the concurrency being measured, which is the
# number this sweep exists to find. Reuse in this trace is scattered, not
# consecutive (median gap 22 requests, ~594k intervening prefill tokens), so
# the hit rate is a real cache-residency measurement rather than a
# back-to-back-request artifact.
#
# The three sibling levels (cache60 / cache90 / coldfloor) live beside this
# file in the same traces/ directory. They get their own model-prefix +
# recipe when they are swept, following the gemma4emnbt{2k,4k,8k} pattern --
# the matrix has no per-config dataset field.
#
# WHY NOT the agentic-coding Weka path: build_replay_cmd pins the
# inferencex-agentx-mvp scenario, which locks --cache-bust first_turn_prefix
# and replays multi-turn trajectory trees. Cache-busting is the exact opposite
# of what a prefix-reuse trace measures, so this recipe calls
# build_mooncake_replay_cmd instead. It reuses the same aiperf venv, the same
# server-metrics/GPU-telemetry scrape and the same result aggregation and
# validation as every other agentic recipe.
#
# CONTEXT: 131,072 (128k), the same admitted-workload cap the sibling agentic
# recipes use, rather than gemma-4's native 256k. The trace's longest request
# is 44,171 input + 974 output = 45,145 tokens, so every record fits with ~3x
# headroom. Note the window does NOT trade against KV capacity: vLLM sizes the
# KV pool from whatever GPU memory is left after weights and peak activation
# under --gpu-memory-utilization, not from --max-model-len; the window only has
# to be small enough that the pool can hold one full-length sequence. How much
# of the trace's shared prefix survives eviction is therefore set by the pool
# size logged below, not by this number.
#
# There is no client-side cap to match it: aiperf's --max-context-length is
# Weka-only and is rejected outright for mooncake traces.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR DURATION

# GPU-resident KV (kv-offloading: none in the master config); assert the matrix
# did not hand us an offload backend this recipe would silently ignore.
require_agentic_kv_offload_none

# Local JSONL, not a HuggingFace corpus -- resolve_trace_source() is
# deliberately NOT called. Path is relative to the repo checkout, which the
# runner bind-mounts at /workspace.
export MOONCAKE_TRACE_FILE="utils/agentic/datasets/traces/gemma4_prod_cache30_910.jsonl"

# Mandatory agentic telemetry: aiperf scrapes vLLM's Prometheus /metrics and
# the DCGM exporter the runner launcher starts on :9400. vllm:prefix_cache_*
# comes from the former and is the primary read-out of this run.
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

DP_SIZE=2
DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SERVER_MAX_MODEL_LEN=131072

# GPU pinning, same convention as the gemma4emnbt* wrappers. h200-greennode_03
# hosts another tenant on GPUs 4-7, and this recipe's --data-parallel-size 2
# would otherwise take "the first DP_SIZE visible cards" -- correct today, but
# only by accident of enumeration order. Pin it so the engine provably cannot
# allocate outside 0,1 no matter what else is on the box. Override with GPU_IDS
# if 0,1 are busy (the gemma4routerspec recipe uses 2,3 on this same host for
# exactly that reason).
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
# Agentic scheduler headroom convention: admit up to 2x the offered concurrency.
MAX_NUM_SEQS=$((2 * CONC))

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# Unfiltered, so the log records the whole box: which cards the other tenant is
# using and how much memory is already committed there. CUDA_VISIBLE_DEVICES
# does not affect nvidia-smi, so this stays a full-box view on purpose.
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

install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export PYTHONNOUSERSITE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Prefix caching is ON by default in vLLM v1 and is load-bearing here; state it
# explicitly so a future default flip cannot silently turn this run into a
# cold-cache run that still reports a "cache30" filename.
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP_SIZE" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$SERVER_MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# How many tokens of KV the engine actually got. This is the single number that
# decides how much of the trace's 30% shared prefix survives eviction, so put it
# in the job log next to the result instead of leaving it buried in server.log.
grep -iE "GPU KV cache size|maximum concurrency" "$SERVER_LOG" | tail -n 10 || true

build_mooncake_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"

# EAGLE3 acceptance and prefix-cache hit rate, straight from the engine. aiperf
# records the Prometheus series into its own artifacts, but nothing echoes the
# engine's own log lines back to the Actions console.
echo "===== EAGLE3 spec-decode acceptance ====="
grep -iE "SpecDecoding metrics|acceptance" "$SERVER_LOG" | tail -n 20 \
    || echo "(no acceptance lines found in $SERVER_LOG)"
echo "===== vLLM prefix cache hit rate ====="
grep -iE "prefix cache hit rate|Prefix cache" "$SERVER_LOG" | tail -n 20 \
    || echo "(no prefix-cache lines found in $SERVER_LOG)"
