#!/usr/bin/env bash

# Gemma-4 31B FP8-block vLLM recipe, fixed-seq-len, 2-way data parallel.
#
# Data parallel *replicas*, not DP attention: gemma-4-31B is dense (plain GQA,
# no MoE and no MLA), so vLLM's implicit DP-attention path -- which only
# applies to MLA models -- is not in play. --data-parallel-size here launches
# DP_SIZE independent engine cores behind one API server, using
# TP * DP_SIZE GPUs (2 at TP1).
#
# DP_SIZE is a constant rather than an env var because nvidia-master.yaml has
# no DP-size field (dp-attn is a bool, and the shared convention wires dp=tp);
# the "dp2" in this file's name and in the gemma4dp2 model-prefix is what
# records the topology. Everything else is byte-identical to
# gemma4_fp8block_h200.sh so the DP ladder is directly comparable to the TP1
# and TP2 ladders at equal CONC.

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

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# --max-num-seqs is per DP rank, so leaving it at CONC keeps it non-binding:
# the API server spreads requests across replicas and a rank that transiently
# receives more than CONC/DP_SIZE is not throttled. Matches the TP recipes'
# intent (no server-side queueing below the client's --max-concurrency).
set -x
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP_SIZE" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

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
