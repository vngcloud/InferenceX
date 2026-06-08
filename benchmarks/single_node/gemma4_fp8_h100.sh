#!/usr/bin/env bash
# Gemma 4 31B FP8 on H100 via vLLM, with optional MTP speculative decoding.
#
# Behaviour switched by SPEC_DECODING (set by the matrix config):
#   "mtp"   → enable native Gemma 4 MTP via vLLM v1 --speculative-config,
#             using Google's official `<model>-assistant` drafter. The
#             assistant is a Q-only decoder that shares the target's KV
#             cache; no separate vLLM process. Requires vLLM ≥ v0.22
#             (PR vllm-project/vllm#41745 merged 2026-05-08).
#   "none"  → plain vLLM serve, no speculative decoding (baseline).
#
# Lab test for MEP-0006 in maas-project/meps/.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    SPEC_DECODING

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# MTP assistant follows the `<target>-assistant` HF naming pattern used
# by Google for the released Gemma 4 drafters (E2B/E4B/26B confirmed in
# vLLM PR #41745 test plan; 31B presumed by extension).
ASSISTANT_MODEL=""
if [ "$SPEC_DECODING" = "mtp" ]; then
    ASSISTANT_MODEL="${MODEL}-assistant"
    if [[ "$ASSISTANT_MODEL" != /* ]]; then hf download "$ASSISTANT_MODEL"; fi
fi

# Gemma 4 31B supports 128K natively. Default to ISL+OSL+headroom; respect
# whatever the matrix passes via $MAX_MODEL_LEN (workflow computes ISL+OSL+256).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( ISL + OSL + 256 ))}"

# Gemma 4 is multimodal — its vision encoder emits 2496 tokens per image
# (max_tokens_per_mm_item). vLLM refuses to start if max-num-batched-tokens
# is smaller than that, so 2048 (a reasonable text-only default) fails.
# 8192 fits MM items and matches our 8k1k workload — used as the fallback
# when the matrix doesn't pin a value via $MAX_NUM_BATCHED_TOKENS.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

cat > config.yaml << EOF
max-model-len: $MAX_MODEL_LEN
max-num-batched-tokens: $MAX_NUM_BATCHED_TOKENS
EOF

export PYTHONNOUSERSITE=1
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Long-context AIPerf path for the GreenNode H200 runs. Keep this separate from
# the historical native-client Gemma sweeps above: those characterize on-the-fly
# fp8 quantization, while this path serves the pre-quantized block-FP8 checkpoint
# through the OpenAI-compatible chat endpoint.
if [ "${BENCHMARK_CLIENT:-inferencex_native}" = "aiperf" ]; then
    TOKENIZER_MODEL="${TOKENIZER_MODEL:-RedHatAI/gemma-4-31B-it-FP8-block}"
    AIPERF_SERVED_MODEL_NAME="${AIPERF_SERVED_MODEL_NAME:-google/gemma-4-31b-it}"
    GEMMA4_AIPERF_MAX_MODEL_LEN="${GEMMA4_AIPERF_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"

    SEARCH_ARGS=()
    if [[ -n "${SEARCH_RECIPE:-}" ]]; then
        SEARCH_ARGS+=(--search-recipe "$SEARCH_RECIPE" --concurrency-min "${CONCURRENCY_MIN}" --concurrency-max "${CONCURRENCY_MAX}")
        if [[ -n "${SLA_MS:-}" ]]; then SEARCH_ARGS+=(--sla-ms "$SLA_MS"); fi
        if [[ -n "${SEARCH_MAX_ITERATIONS:-}" ]]; then SEARCH_ARGS+=(--search-max-iterations "$SEARCH_MAX_ITERATIONS"); fi
    fi
    if [[ -n "${BENCHMARK_DURATION:-}" ]]; then
        SEARCH_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
        if [[ -n "${BENCHMARK_GRACE_PERIOD:-}" ]]; then
            SEARCH_ARGS+=(--benchmark-grace-period "$BENCHMARK_GRACE_PERIOD")
        fi
    fi

    start_gpu_monitor

    set -x
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER_MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --served-model-name google/gemma-4-31b-it gemma-4-31b-it \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.92 \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len "$GEMMA4_AIPERF_MAX_MODEL_LEN" \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!

    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

    run_client_benchmark \
        --model "$AIPERF_SERVED_MODEL_NAME" \
        --tokenizer "$TOKENIZER_MODEL" \
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
        --server-pid "$SERVER_PID" \
        --random-seed "${RANDOM_SEED:-0}" \
        --extra-inputs ignore_eos:true \
        "${SEARCH_ARGS[@]}"
    BENCHMARK_EXIT_CODE=$?

    stop_gpu_monitor
    set +x
    exit "$BENCHMARK_EXIT_CODE"
fi

# num_speculative_tokens (N) comes from the matrix via $NUM_SPECULATIVE_TOKENS.
# Falls back to N=2 to match Gemma 4's native MTP drafter depth — kept for
# backwards-compat with configs that don't pin a value.
NUM_SPEC_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"

# Build spec-decode flag set as an array — safest way to thread JSON
# containing both single and double quotes through bash word-splitting.
SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
    SPEC_JSON="{\"model\":\"${ASSISTANT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
    SPEC_ARGS=(--speculative-config "$SPEC_JSON")
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --host=0.0.0.0 --port=$PORT \
    --config config.yaml \
    --quantization=fp8 \
    --kv-cache-dtype=fp8_e4m3 \
    --gpu-memory-utilization=0.9 \
    --tensor-parallel-size=$TP \
    --max-num-seqs=$CONC \
    --trust-remote-code \
    "${SPEC_ARGS[@]}" > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# Acceptance-rate metrics already land in server.log (vLLM logs
# `SpecDecoding metrics: ...` every ~10 s during inference) which is
# uploaded by the workflow's existing Upload server logs step. No
# extra curl needed.

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
