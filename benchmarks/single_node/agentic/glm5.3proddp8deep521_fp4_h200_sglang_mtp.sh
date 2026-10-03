#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3-W4AFP8 prod-exact DP8 port of glm5.2proddp8deep519_fp4_h200_sglang.sh
# (reference run 34508949422, v0.5.19, h200-greennode_06, 2026-09-11 ICT): same
# TP8/DP8/EP8 DPA + MTP-chain + hicache-128/rank + deepep + dfs-weight server
# config, moved to the hardware-hcm slurm pool with four forced deltas:
#   1. weights = /data/hf-cache/GLM-5.3-W4AFP8, a plain HF snapshot dir staged
#      on both hgx nodes (= PhalaCloud/GLM-5.3-W4AFP8@03179e95); no
#      snapshot_download — the slurm launcher mounts /data/hf-cache 1:1.
#   2. image lmsysorg/sglang:v0.5.21 (was v0.5.19).
#   3. router runs in-container via python3 -m sglang_router.launch_router with
#      the prod sidecar's args (cache_aware + dp-aware, cache-threshold 0.3,
#      balance-abs 100000 / rel 2.0, tree 64MiB, eviction 300s, timeout 900s,
#      retry 2) — enroot/pyxis has no docker socket, so the 5.2 arm's VCR
#      sglang-router-patched:v0.5.18 sidecar cannot run here. Router metrics
#      port is PORT+1000 (not the 5.2 arm's PORT+10000): the slurm launcher
#      picks a dynamic PORT that can exceed 55535 (r2 fix 038e065d).
#   4. --numa-node 0 1 2 3 0 1 2 3: hardware-hcm hosts are 4 NUMA nodes of
#      ~504G each (lscpu; GPU0-2->N0, GPU3->N1, GPU4-6->N2, GPU7->N3), not
#      han-1's 2 sockets. The ported 0 0 0 0 1 1 1 1 numa_set_preferred 4
#      ranks (4x ~154G KV+indexer host pools) onto each 504G zone; the kernel
#      OOM-killed ranks silently mid cudaHostRegister (5 smokes exit-137,
#      manual repro jobs 1579/1581, faulthandler silent = SIGKILL not SEGV,
#      global MemAvailable stayed ~1T because zones 2/3 sat empty). 2
#      ranks/node = ~308G leaves headroom; manual boot job 1584 green,
#      oom_kill delta 0.
# GLM-5.3 arch == GLM-5.2 arch field-for-field (GlmMoeDsaForCausalLM, 78
# layers / 256 experts / top-8, DSA index_topk 2048, 1 nextn layer), so every
# serve flag ports unchanged.
# --hicache-size is PER-RANK GB: 8 ranks * 128 = 1024 GB host RAM total. The
# 8x-tier salloc caps the job at 1920G (GPU_COUNT*240G) on a ~2002G node, and
# the pool's available-cpu-dram-mib (983_040, sized for the 4x tier)
# under-reports the budget in TOTAL_CPU_DRAM_GB; the binding guards are
# sglang's host free-RAM check (~1900G free / 8 ranks ~= 237G/rank >= 128) and
# the salloc cap itself, both of which hold alongside the ~39G/rank DSA
# indexer host pool.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING PORT
require_agentic_kv_offload_backend hicache

MODEL_DIR=/data/hf-cache/GLM-5.3-W4AFP8
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: $MODEL_DIR/config.json not found — stage PhalaCloud/GLM-5.3-W4AFP8 on the node first" >&2; exit 1; }
export MODEL_PATH="$MODEL_DIR"
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
# The slurm launcher exports its own DCGM sidecar port; the docker path uses 9400.
export AIPERF_GPU_TELEMETRY_URL="${AIPERF_GPU_TELEMETRY_URL:-http://localhost:9400/metrics}"

# Prod env parity
export SGLANG_DP_USE_GATHERV=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_ENABLE_METRICS_DP_ATTENTION=1
# hardware-hcm fabric: the 8 p2p RoCE links are Intel E810 (irdma), which
# NVSHMEM's IBRC transport cannot probe — DeepEP low-latency bring-up segfaults
# at main_nvshmem/.../ibrc.cpp:314 "NULL value" (smoke run 37134667065). A
# single-node TP8 job has no remote PEs, so disable the remote transport
# entirely; same env nguyennvc's dispatch-l3 GLM-5.3-W4AFP8 tp8/ep8 DeepEP
# engines run with on these exact nodes.
export NVSHMEM_REMOTE_TRANSPORT=none
# The silent smoke deaths were per-NUMA-zone exhaustion (delta 4), not the
# registration call: the default single 128G cudaHostRegister is fine once
# the zones have headroom (manual boot 1584), so registration chunking stays
# at the sglang default. Print native fault stacks if anything still dies.
export PYTHONFAULTHANDLER=1

CACHE_ARGS=(
  --enable-hierarchical-cache
  --hicache-size 128
  --hicache-io-backend direct
  --hicache-write-policy write_back
)

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

SGLANG_BACKEND_PORT="$PORT"
ROUTER_ENABLED=false
if [ -n "${ROUTER_METADATA:-}" ]; then
  ROUTER_ENABLED=true
  SGLANG_BACKEND_PORT=$((PORT + 1))
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

# Slurm can hand us GPUs while the previous job's processes still hold VRAM
# (runs 36322826747/36322829133 died at boot on "memory capacity is
# unbalanced"). Wait up to 10 min for every visible GPU to drop under 2 GiB.
for _ in $(seq 60); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {busy=1} END {exit busy}' && break
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -lt 256 ] && MAX_RUNNING_REQUESTS=256
PARALLEL_ARGS=(--tp-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
  [ "$MAX_RUNNING_REQUESTS" -lt "$TP" ] && MAX_RUNNING_REQUESTS=$TP
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp 8
    --ep "$EP_SIZE"
    --enable-dp-attention
    --enable-dp-attention-local-control-broadcast
    --enable-dp-lm-head
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    --numa-node 0 1 2 3 0 1 2 3
  )
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --moe-a2a-backend deepep
  --chunked-prefill-size 32768
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.75
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --context-length 300000
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  --schedule-policy dfs-weight
  --enable-prefill-delayer
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

# 64MB stack, as the prod containers and nguyennvc's dispatch-l3 engines run
# (the sbatch/enroot default is 8MB).
ulimit -s 65536

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prod router args (mirror the live glm-52-fp8-h200-8x-router sidecar), run
# in-container: no docker socket under enroot/pyxis.
if [ "$ROUTER_ENABLED" = "true" ]; then
  ROUTER_LOG="$RESULT_DIR/router.log"
  python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy cache_aware \
    --dp-aware \
    --cache-threshold 0.3 \
    --balance-abs-threshold 100000 \
    --balance-rel-threshold 2.0 \
    --max-tree-size 67108864 \
    --eviction-interval-secs 300 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$((PORT + 1000))" \
    --health-check-interval-secs 15 \
    --health-check-timeout-secs 10 \
    --health-failure-threshold 5 \
    --request-timeout-secs 900 \
    --retry-max-retries 2 > "$ROUTER_LOG" 2>&1 &
  ROUTER_PID=$!
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
