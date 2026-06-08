#!/usr/bin/env bash
# Gemma 4 31B FP8 on 2x H100 via SGLang (baseline, no speculative decoding).
#
# Counterpart to the vLLM gemma4_fp8_h100.sh for an engine-to-engine compare.
# Expects a pre-quantized compressed-tensors fp8 checkpoint (e.g.
# RedHatAI/gemma-4-31B-it-FP8-dynamic) whose quantization_config ignores the
# vision tower (ignore: re:.*vision.*). We deliberately do NOT pass
# --quantization: SGLang auto-detects compressed-tensors from the checkpoint
# and honours that ignore list, keeping the vision encoder in bf16. Passing
# --quantization fp8 (on-the-fly) instead quantizes the vision tower too and
# crashes in triton_scaled_mm (gemma4_vision.py scale_b shape assertion).
# --kv-cache-dtype fp8_e4m3 mirrors the vLLM bench's KV-cache choice.
#
# Selected by runners/launch_h100-greennode.sh for framework=sglang configs
# whose model-prefix is gemma4 (e.g. gemma4-fp8-h100-2x-sglang).

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

nvidia-smi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MAX_SEQ_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 20))}"
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_SEQ_LEN="$EVAL_MAX_MODEL_LEN"
fi

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN, TP: $TP"

if [ "${BENCHMARK_CLIENT:-inferencex_native}" = "aiperf" ]; then
    AIPERF_MODEL_NAME="${AIPERF_MODEL_NAME:-google/gemma-4-31b-it}"
    TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL}"
    SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-$CONC}"
    if (( SERVER_MAX_RUNNING_REQUESTS > 256 )); then
        SERVER_MAX_RUNNING_REQUESTS=256
    fi

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
    python3 -m sglang.launch_server \
      --model "$MODEL" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --served-model-name "$AIPERF_MODEL_NAME" \
      --tp "$TP" \
      --tool-call-parser gemma4 \
      --kv-cache-dtype fp8_e4m3 \
      --mem-fraction-static 0.90 \
      --chunked-prefill-size 8192 \
      --context-length "$MAX_SEQ_LEN" \
      --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" \
      --cuda-graph-max-bs "$SERVER_MAX_RUNNING_REQUESTS" \
      --decode-log-interval 1 \
      --trust-remote-code \
      > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!

    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

    run_client_benchmark \
        --model "$AIPERF_MODEL_NAME" \
        --tokenizer "$TOKENIZER_MODEL" \
        --port "$PORT" \
        --backend sglang-oai \
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

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --context-length "$MAX_SEQ_LEN" \
  --cuda-graph-max-bs "$CONC" \
  --decode-log-interval 1 \
  --trust-remote-code \
  > "$SERVER_LOG" 2>&1 &

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
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
