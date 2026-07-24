#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

# Fixed-seq-len (8k1k) twin of the agentic recipe
# agentic/glm5.2_fp4_b300-netperf_sglang.sh: same TP8 + DP-attention + HiCache r1
# serving stack, but driven by a fixed ISL/OSL throughput sweep instead of the
# agentic Weka replay. No sglang-router here — the router only exists to give the
# agentic replay correlation-id prefix affinity; a random fixed-seq-len workload
# has no shared prefix, so requests go straight to the DP-aware scheduler on $PORT.
check_env_vars MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME

if [ "$DP_ATTENTION" != "true" ]; then
    echo "Error: this recipe requires DP_ATTENTION=true" >&2
    exit 1
fi

export MODEL_PATH=/mnt/models/nvidia/GLM-5.2-NVFP4
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export SGLANG_TIMEOUT_KEEP_ALIVE=900

nvidia-smi

RESULT_DIR="${RESULT_DIR:-$PWD}"
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
CONTEXT_LENGTH=$((ISL + OSL + 20))

# ponytail: floor max-running-requests at TP so CONC=1 doesn't underflow to 0
# per-DP-rank (2*CONC is divided across the 8 DP ranks). Above CONC=4 the 2*CONC
# term wins, matching the agentic run's MAX_RUNNING_REQUESTS=2*CONC.
MAX_RUNNING_REQUESTS=$(( 2 * CONC > TP ? 2 * CONC : TP ))

start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --dp "$TP"
    --enable-dp-attention
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    --quantization modelopt_fp4
    --chunked-prefill-size 32768
    --tool-call-parser glm47
    --reasoning-parser glm45
    --mem-fraction-static 0.92
    --schedule-policy lpm
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --watchdog-timeout 1800
    --context-length "$CONTEXT_LENGTH"
    --enable-metrics
    --enable-cache-report
    --enable-hierarchical-cache
    --hicache-ratio 1.0
    --hicache-write-policy write_back
    --hicache-io-backend direct
    --hicache-mem-layout page_first_direct
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

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
    --result-dir "$RESULT_DIR/" \
    --use-chat-template

stop_gpu_monitor
set +x
