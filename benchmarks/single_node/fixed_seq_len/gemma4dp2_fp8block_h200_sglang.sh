#!/usr/bin/env bash

# Gemma-4 31B FP8-block SGLang recipe, fixed-seq-len, 2-way data parallel.
#
# Data parallel *replicas*, not DP attention: gemma-4-31B is dense (plain GQA,
# no MoE and no MLA), so --enable-dp-attention has no KV-cache duplication to
# eliminate and is deliberately NOT passed. --dp-size here is SGLang's classic
# DP mode -- it launches DP_SIZE independent TP groups behind one HTTP server
# with built-in load balancing, so the box uses TP * DP_SIZE GPUs (4 at TP2).
#
# DP_SIZE is a constant rather than an env var because nvidia-master.yaml has
# no DP-size field (dp-attn is a bool, and the shared convention wires dp=tp);
# the "dp2" in this file's name and in the gemma4dp2 model-prefix is what
# records the topology. Everything else is byte-identical to
# gemma4_fp8block_h200_sglang.sh so the DP ladder is directly comparable to the
# TP1 and TP2 ladders at equal CONC.

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

DP_SIZE=2

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

# --max-running-requests is per DP rank, so leaving it at CONC keeps it
# non-binding: the load balancer hands each replica ~CONC/DP_SIZE in steady
# state, and a rank that transiently receives more is not throttled. Matching
# the TP recipes' intent (no server-side queueing below the client's
# --max-concurrency) rather than their literal per-rank number.
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
    --context-length "$MAX_MODEL_LEN"
    --max-running-requests "$CONC"
    --chunked-prefill-size 8192
    --enable-metrics
)

# See gemma4_fp8block_h200_sglang.sh: SGLang keeps its own parser registry with
# its own spellings, so probe rather than fail server startup outright.
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
