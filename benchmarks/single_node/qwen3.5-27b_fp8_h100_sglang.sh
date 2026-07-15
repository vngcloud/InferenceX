#!/usr/bin/env bash
# Qwen3.5-27B (dense) FP8 on a single H100 via SGLang.
#
# SGLang counterpart to qwen3.5-27b_fp8_h100.sh (vLLM) for an engine-to-engine
# comparison on the same checkpoint/hardware. Adapted from the 397B-A17B
# qwen3.5_fp8_h100.sh: dense single-GPU (no expert parallelism, no multi-GPU
# allreduce fusion).
#
# Qwen/Qwen3.5-27B-FP8 is a pre-quantized block-fp8 checkpoint whose
# quantization_config.modules_to_not_convert already excludes the vision tower
# (model.visual.*), lm_head, embeddings and the linear-attn conv1d/in_proj
# layers. So we pass NO --quantization: SGLang auto-detects the checkpoint's
# fp8 config and honours that ignore list (LLM fp8, vision bf16). Forcing
# --quantization fp8 (on-the-fly) would quantize the vision tower and crash in
# triton_scaled_mm (the gemma4 failure mode).

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
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --max-running-requests 64 \
  --chunked-prefill-size 8192 \
  --decode-log-interval 1 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs "$CONC" \
  --context-length "$MAX_SEQ_LEN" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --stream-interval 50 \
  --tokenizer-worker-num 6 \
  --mamba-ssm-dtype bfloat16 \
  --disable-radix-cache \
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
