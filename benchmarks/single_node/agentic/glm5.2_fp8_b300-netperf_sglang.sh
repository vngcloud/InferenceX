#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION EP_SIZE SPEC_DECODING HICACHE_RATIO
require_agentic_kv_offload_backend hicache

if [ "$SPEC_DECODING" != "mtp" ]; then
    echo "Error: this recipe requires SPEC_DECODING=mtp" >&2
    exit 1
fi

export MODEL_PATH=/mnt/models/zai-org/GLM-5.2-FP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
HICACHE_L3=0
HICACHE_BACKEND=""
if [[ "${KV_OFFLOAD_BACKEND_METADATA:-}" == *l3-nixl-posix* ]]; then
    HICACHE_L3=1
    HICACHE_BACKEND=nixl
    export SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX
    export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="/mnt/test-raid0/hicache/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-c${CONC}"
elif [[ "${KV_OFFLOAD_BACKEND_METADATA:-}" == *l3-mooncake-ssd* ]]; then
    HICACHE_L3=1
    HICACHE_BACKEND=mooncake
    MOONCAKE_SSD_DIR="/mnt/test-raid0/mooncake/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-c${CONC}"
    mkdir -p "$MOONCAKE_SSD_DIR"
    mooncake_master --rpc_port=50051 --metrics_port=9003 --enable_offload=true \
        --eviction_high_watermark_ratio=0.95 --logtostderr=true \
        > "$RESULT_DIR/mooncake_master.log" 2>&1 &
    for _ in {1..30}; do
        if curl -fsS http://127.0.0.1:9003/metrics >/dev/null; then
            break
        fi
        sleep 1
    done
    curl -fsS http://127.0.0.1:9003/metrics >/dev/null
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size 8192
    --mem-fraction-static 0.88
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --watchdog-timeout 1800
    --enable-metrics
    --allow-auto-truncate
    --enable-cache-report
    --schedule-policy lpm
    --kv-cache-dtype fp8_e4m3
    --bf16-gemm-backend cutedsl
    --max-prefill-tokens 8192
    --cuda-graph-max-bs 256
    --enable-hierarchical-cache
    --hicache-write-policy write_back
    --hicache-io-backend direct
    --hicache-mem-layout page_first_direct
    --hicache-ratio "$HICACHE_RATIO"
    --enable-flashinfer-allreduce-fusion
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)
if [ "$HICACHE_L3" -eq 1 ]; then
    if [ "$HICACHE_BACKEND" = "nixl" ]; then
        SGLANG_CMD+=(
            --hicache-storage-backend nixl
            --hicache-storage-prefetch-policy timeout
            --hicache-storage-backend-extra-config '{"use_direct_io":false,"use_uring":"false","l3_cleaner_high_watermark":40.0,"l3_cleaner_low_watermark":30.0}'
        )
    else
        SGLANG_CMD+=(
            --hicache-storage-backend mooncake
            --hicache-storage-prefetch-policy timeout
            --hicache-storage-backend-extra-config "{\"master_server_address\":\"127.0.0.1:50051\",\"metadata_server\":\"P2PHANDSHAKE\",\"local_hostname\":\"127.0.0.1\",\"protocol\":\"tcp\",\"global_segment_size\":\"1gb\",\"enable_ssd_offload\":true,\"ssd_offload_path\":\"$MOONCAKE_SSD_DIR\"}"
        )
    fi
fi

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
