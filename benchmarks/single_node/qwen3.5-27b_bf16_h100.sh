#!/usr/bin/env bash
# Qwen3.5-27B (dense) BF16 on a single H100 via vLLM.
#
# Qwen3.5-27B is the dense sibling of the Qwen3.5-397B-A17B MoE. This serves
# the base model (Qwen/Qwen3.5-27B) in bfloat16 precision without quantization.
#
# Arch is Qwen3_5ForConditionalGeneration (a vision-language model). This
# bench drives text-only random-token workloads; serving flags
# (reasoning/tool parsers) are kept to match the production serve config.
#
# Lab bench: TP=1, single H100 (from h100-2x runner), max throughput search
# across conc {8-32} for isl 8k x osl 1k via Bayesian Optimization.

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

start_gpu_monitor

set -x
vllm serve "$MODEL" --host=0.0.0.0 --port=$PORT \
    --config config.yaml \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --max-num-seqs=$CONC > $SERVER_LOG 2>&1 &

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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
