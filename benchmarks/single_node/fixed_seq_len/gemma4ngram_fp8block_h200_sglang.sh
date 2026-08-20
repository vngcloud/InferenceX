#!/usr/bin/env bash

# Gemma-4 31B FP8-block SGLang recipe, fixed-seq-len, NGRAM (model-free)
# speculative decoding, 2-way data parallel (tp1 x dp2 = 2 GPUs).
#
# Spec-decoding sibling of gemma4dp2_fp8block_h200_sglang.sh: identical
# checkpoint, image, runner, 8k1k workload and DP2 topology, plus NGRAM
# speculation. NGRAM is used here instead of EAGLE3 deliberately: gemma-4 EAGLE3
# is NOT in mainline SGLang (only in the ThoughtWorks fork), so it cannot run on
# our pinned lmsysorg/sglang:v0.5.16-cu130. NGRAM needs no draft model -- it
# proposes tokens from an n-gram cache of previously generated text -- so it runs
# on the upstream image today. This is a DIFFERENT spec method than the vLLM
# EAGLE3 twin, so cross-engine numbers are not apples-to-apples; the comparison
# is spec-on vs spec-off within each engine, plus each engine's best available
# spec path on its released image.
#
# Classic DP (--dp 2, no --enable-dp-attention) is compatible with NGRAM: the
# docs only forbid NGRAM with --enable-dp-attention, which this dense model never
# uses. Each DP replica runs its own independent NGRAM engine.
#
# DP_SIZE is a constant (see gemma4dp2_fp8block_h200_sglang.sh); the "ngram" in
# this file's name and the gemma4ngram model-prefix record the variant and select
# the recipe path (this runner appends no spec-suffix).
#
# NOTE ON ACCEPTANCE: fixed-seq-len feeds *random* token content, which is close
# to a worst case for NGRAM -- an n-gram cache cannot predict unrepeated random
# tokens, so acceptance (and any speedup) will be low here. --use-chat-template
# makes the structural tokens predictable, but a faithful measurement needs the
# agentic (real-trace) scenario, where repeated context makes NGRAM effective.
# Treat the fixed-seq-len number as a conservative floor.

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

# NGRAM disables the overlap scheduler and mixed chunked prefill (documented), so
# --chunked-prefill-size is kept but is no longer "mixed". --speculative-num-draft-tokens
# sets the parallel verification width; the ngram-* knobs are SGLang's documented
# defaults, pinned explicitly so the sweep is reproducible if defaults change.
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
    --speculative-algorithm NGRAM
    --speculative-num-draft-tokens 8
    --speculative-ngram-min-bfs-breadth 1
    --speculative-ngram-max-bfs-breadth 10
    --speculative-ngram-match-type BFS
    --speculative-ngram-max-trie-depth 18
    --speculative-ngram-capacity 10000000
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

# --use-chat-template: see the acceptance note in the header.
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
    --trust-remote-code \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
