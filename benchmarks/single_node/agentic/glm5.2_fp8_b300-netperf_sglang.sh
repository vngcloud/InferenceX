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
# The GPU KV pool holds ~2.73M tokens ~= 17 mean-sized agentic contexts (157k tokens each).
# A blind 2*CONC admits 64 at CCU 32, which drove token usage to 0.99 with 3.3M pending
# tokens against a 2.73M pool and killed the server on a cuBLAS GEMM. Cap admission so
# pending work cannot run that far past the pool.
MAX_RUNNING_REQUESTS_CAP="${MAX_RUNNING_REQUESTS_CAP:-32}"
if [ "$MAX_RUNNING_REQUESTS" -gt "$MAX_RUNNING_REQUESTS_CAP" ]; then
    MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS_CAP"
fi
# Larger == more conservative admission (server_args.py:717). Default 1.0 over-admits here.
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-2.0}"
HICACHE_L3=0
HICACHE_BACKEND=""
HICACHE_WRITE_POLICY=write_back
if [[ "${KV_OFFLOAD_BACKEND_METADATA:-}" == *l3-nixl-posix* ]]; then
    HICACHE_L3=1
    HICACHE_BACKEND=nixl
    export SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX
    export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="/mnt/test-raid0/hicache/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-c${CONC}"
elif [[ "${KV_OFFLOAD_BACKEND_METADATA:-}" == *l3-mooncake-ssd* ]]; then
    HICACHE_L3=1
    HICACHE_BACKEND=mooncake
    # write_through pushes every page to L2+L3 as it is produced; write_back only pushes on
    # eviction, which never populates the store enough to measure L3.
    HICACHE_WRITE_POLICY=write_through
    # DRAM tier for the store. Must SATURATE inside the run: SSD is only written via eviction,
    # so a segment the run never fills leaves SSD at 0 B. Derived, not hardcoded.
    # Fill rate with write_through is ~0.31 GB/min per unit concurrency (measured: 272 GB in
    # ~28 min at CONC 32, run 30141767135). Target crossing the 0.85 watermark at 20% of the
    # window, leaving ~80% with offload active:
    #     gb = 0.31 * CONC * (DURATION/60) * 0.20 / 0.85  ==  31 * CONC * DURATION / 25500
    # See docs/HICACHE_L3_SMOKE_TEST.md §3.
    if [ -z "${MOONCAKE_GLOBAL_SEGMENT_SIZE:-}" ]; then
        SEG_GB=$(( 31 * CONC * DURATION / 25500 ))
        [ "$SEG_GB" -lt 4 ] && SEG_GB=4        # 1gb proved pathological; never go that small
        [ "$SEG_GB" -gt 512 ] && SEG_GB=512    # host RAM safety ceiling
        MOONCAKE_GLOBAL_SEGMENT_SIZE="${SEG_GB}gb"
    fi
    echo "mooncake: global_segment_size=$MOONCAKE_GLOBAL_SEGMENT_SIZE (CONC=$CONC DURATION=$DURATION)"
    MOONCAKE_PREFETCH_POLICY="${MOONCAKE_PREFETCH_POLICY:-wait_complete}"
    MOONCAKE_SSD_DIR="/mnt/test-raid0/mooncake/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-c${CONC}"
    mkdir -p "$MOONCAKE_SSD_DIR"
    # offload_on_evict is required: --enable_offload alone mounts the disk segment but never
    # writes to it, because offload happens via eviction. Without it every eviction attempt
    # fails (the master will not drop the only replica), so SSD stays at 0 B and the master
    # spins on EVICT-TRIGGER. See docs/HICACHE_L3_MOONCAKE_PLAN.md.
    mooncake_master --rpc_port=50051 --metrics_port=9003 --enable_offload=true \
        --offload_on_evict=true \
        --eviction_high_watermark_ratio=0.85 --eviction_ratio=0.10 --logtostderr=true \
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
    --schedule-conservativeness "$SCHEDULE_CONSERVATIVENESS"
    --kv-cache-dtype fp8_e4m3
    --bf16-gemm-backend cutedsl
    --max-prefill-tokens 8192
    --cuda-graph-max-bs 256
    --enable-hierarchical-cache
    --hicache-write-policy "$HICACHE_WRITE_POLICY"
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
            # wait_complete blocks the batch until a prefetch fully lands (no timeout escape,
            # all-reduced across TP workers). global_segment_size is the cluster-wide DRAM
            # segment and is divided by tp_size, so "1gb" gave each of 8 ranks only 128 MB.
            --hicache-storage-prefetch-policy "$MOONCAKE_PREFETCH_POLICY"
            --hicache-storage-backend-extra-config "{\"master_server_address\":\"127.0.0.1:50051\",\"metadata_server\":\"P2PHANDSHAKE\",\"local_hostname\":\"127.0.0.1\",\"protocol\":\"tcp\",\"global_segment_size\":\"$MOONCAKE_GLOBAL_SEGMENT_SIZE\",\"enable_ssd_offload\":true,\"ssd_offload_path\":\"$MOONCAKE_SSD_DIR\",\"prefetch_threshold\":64}"
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
