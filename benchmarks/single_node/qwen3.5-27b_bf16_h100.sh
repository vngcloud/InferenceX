#!/usr/bin/env bash
# Qwen3.5-27B (dense) BF16 on a single H100 via vLLM, benchmarked through
# official AIPerf.
#
# Qwen3.5-27B is the dense sibling of the Qwen3.5-397B-A17B MoE. This serves
# the base model (Qwen/Qwen3.5-27B) in bfloat16 precision without quantization.
#
# Arch is Qwen3_5ForConditionalGeneration (a vision-language model). This
# bench drives text-only random-token workloads; serving flags
# (reasoning/tool parsers) are kept to match the production serve config.
#
# Supports AIPerf native Bayesian-Optimization search (--search-recipe) over a
# [CONCURRENCY_MIN, CONCURRENCY_MAX] range, optionally with duration-based
# measurement (BENCHMARK_DURATION). In search mode CONC carries the upper
# search bound (CONCURRENCY_MAX) so the server is sized for the largest
# concurrency AIPerf's BO may probe; the adapter records the single winning
# point. The vLLM running-batch cap (--max-num-seqs) is taken from
# MAX_NUM_SEQS when set, otherwise defaults to CONC, so an experiment can pin
# the batch (e.g. 256) independently of the offered-concurrency search range.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Respect whatever the matrix pins (workflow computes ISL+OSL+256); fall back
# to a self-consistent default if unset.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Running-batch cap. Defaults to CONC (server sized for the largest probed
# concurrency); an experiment may pin it lower via MAX_NUM_SEQS.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$CONC}"

cat > config.yaml << EOF
max-model-len: $MAX_MODEL_LEN
max-num-batched-tokens: $MAX_NUM_BATCHED_TOKENS
EOF

export PYTHONNOUSERSITE=1
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor

set -x
vllm serve "$MODEL" --host=0.0.0.0 --port=$PORT \
    --config config.yaml \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --max-num-seqs="$MAX_NUM_SEQS" > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# Optional AIPerf native BO search recipe (config-driven via env). When set, the
# adapter delegates to `aiperf --search-recipe` over [CONCURRENCY_MIN,
# CONCURRENCY_MAX] and records the winning point AIPerf's BO selects.
SEARCH_ARGS=()
if [[ -n "${SEARCH_RECIPE:-}" ]]; then
    SEARCH_ARGS+=(--search-recipe "$SEARCH_RECIPE" --concurrency-min "${CONCURRENCY_MIN}" --concurrency-max "${CONCURRENCY_MAX}")
    if [[ -n "${SLA_MS:-}" ]]; then SEARCH_ARGS+=(--sla-ms "$SLA_MS"); fi
    if [[ -n "${SEARCH_MAX_ITERATIONS:-}" ]]; then SEARCH_ARGS+=(--search-max-iterations "$SEARCH_MAX_ITERATIONS"); fi
fi
# Optional duration-based measurement (config-driven). When set, each BO-probed
# concurrency is measured for BENCHMARK_DURATION seconds instead of a fixed
# request count; BENCHMARK_GRACE_PERIOD must exceed one request's decode time.
if [[ -n "${BENCHMARK_DURATION:-}" ]]; then
    SEARCH_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
    if [[ -n "${BENCHMARK_GRACE_PERIOD:-}" ]]; then
        SEARCH_ARGS+=(--benchmark-grace-period "$BENCHMARK_GRACE_PERIOD")
    fi
fi

run_client_benchmark \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --endpoint-type chat \
    --isl "$ISL" \
    --osl "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --bench-serving-dir "${INFMAX_CONTAINER_WORKSPACE:-$(pwd)}" \
    --trust-remote-code \
    --server-pid "$SERVER_PID" \
    --random-seed "${RANDOM_SEED:-0}" \
    "${SEARCH_ARGS[@]}"

stop_gpu_monitor
set +x
