#!/usr/bin/env bash
# Gemma 4 31B FP8 on 2x H100 via SGLang (baseline, no speculative decoding).
#
# Counterpart to the vLLM gemma4_fp8_h100.sh so the two engines can be
# compared apples-to-apples on the same model/hardware. google/gemma-4-31B-it
# ships as a bf16 checkpoint; --quantization fp8 does on-the-fly fp8 (matching
# the vLLM bench's --quantization=fp8), and --kv-cache-dtype fp8_e4m3 mirrors
# the vLLM KV-cache choice.
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
MAX_SEQ_LEN=$((ISL + OSL + 20))
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_SEQ_LEN="$EVAL_MAX_MODEL_LEN"
fi

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN, TP: $TP"

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP" \
  --quantization fp8 \
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
