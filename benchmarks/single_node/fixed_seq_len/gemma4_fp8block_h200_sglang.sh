#!/usr/bin/env bash

# Gemma-4 31B FP8-block (RedHatAI/gemma-4-31B-it-FP8-block) single-GPU SGLang
# recipe, fixed-seq-len. Framework twin of gemma4_fp8block_h200.sh: same
# checkpoint, same ISL/OSL sweep, same fp8_e4m3 KV cache and the same
# per-request admission limit, so the two runs are directly comparable at equal
# CONC. Note --mem-fraction-static is NOT a rename of vLLM's
# --gpu-memory-utilization: SGLang's fraction covers only the static pool
# (weights + KV) and leaves the remainder for CUDA graphs and activations, so
# copying vLLM's 0.92 across would OOM during graph capture. 0.88 matches the
# other single-node SGLang recipes in this directory.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
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

SERVER_LOG=/workspace/server.log

export PYTHONNOUSERSITE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# Radix cache is left enabled on purpose: vLLM has prefix caching on by
# default, so disabling it here would break the framework comparison. With
# RANDOM_RANGE_RATIO the prompts share almost no prefix, so it is near-neutral.
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --mem-fraction-static 0.88
    --kv-cache-dtype fp8_e4m3
    --context-length "$MAX_MODEL_LEN"
    --max-running-requests "$CONC"
    --chunked-prefill-size 8192
    --enable-metrics
)

# SGLang keeps its own tool/reasoning parser registry with its own spellings
# (glm47, glm45, deepseekv4 ...), so vLLM's "gemma4" is not guaranteed to be
# registered in this image. The throughput sweep hits /v1/completions with
# --ignore-eos and never parses a tool call, so these flags only matter to the
# gsm8k eval leg -- probe for them rather than fail server startup outright.
if python3 -m sglang.launch_server --help 2>&1 | grep -q gemma4; then
    SGLANG_CMD+=(--tool-call-parser gemma4 --reasoning-parser gemma4)
else
    echo "NOTE: no 'gemma4' parser registered in this sglang image; serving without tool/reasoning parsers."
fi

set -x
printf '%q ' "${SGLANG_CMD[@]}" | tee /workspace/sglang_command.txt
printf '\n' | tee -a /workspace/sglang_command.txt

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &

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
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
