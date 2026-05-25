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

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

cat > config.yaml << EOF
max-model-len: $MAX_MODEL_LEN
max-num-batched-tokens: 2048
EOF

export PYTHONNOUSERSITE=1
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Build spec-decode flag set as an array — safest way to thread JSON
# containing both single and double quotes through bash word-splitting.
SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
    SPEC_JSON="{\"model\":\"${ASSISTANT_MODEL}\",\"num_speculative_tokens\":2}"
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

# Capture vLLM's spec-decode acceptance metrics for later analysis. These
# only exist when SPEC_DECODING=mtp; harmless to call either way.
curl -s "http://127.0.0.1:${PORT}/metrics" \
    | grep -E "spec_decode|speculative" \
    > /workspace/spec_metrics_${RESULT_FILENAME}.txt 2>/dev/null || true

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
