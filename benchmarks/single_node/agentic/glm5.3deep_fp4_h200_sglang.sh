#!/usr/bin/env bash
set -euo pipefail
set -x

# GLM-5.3 on the glm5.2deep recipe: byte-for-byte the same serving command as
# glm5.2deep_fp4_h200_sglang.sh with only the checkpoint swapped, so the pair
# reads the model delta and nothing else. GLM-5.3 is the same architecture as
# 5.2 (GlmMoeDsaForCausalLM / glm_moe_dsa, 78 layers, index_topk 2048,
# num_nextn_predict_layers 1), so every flag below -- w4afp8, the DSA prefill
# backend, DeepEP ep8 under DP-attention, EAGLE 3/1/4 off the built-in MTP
# layer -- carries over unchanged. Boot verified on h200-greennode_03
# (2026-09-02): ready in 15m40s, 16/16 concurrent requests, 100k-token prefill
# in 18.8 s, MTP accept len 2.05-2.75, 126 GiB/143 GiB used at mem-fraction
# 0.80 (17 GiB headroom, vs the 9.8 GiB that OOMed glm5.2deep at 0.85).
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING

CACHE_ARGS=()
if [ "$KV_OFFLOADING" = "dram" ]; then
  require_agentic_kv_offload_backend hicache
  CACHE_ARGS=(--enable-hierarchical-cache)
  if [ -n "${HICACHE_RATIO:-}" ]; then
    CACHE_ARGS+=(--hicache-ratio "$HICACHE_RATIO")
  else
    CACHE_ARGS+=(--hicache-size 128)
  fi
else
  require_agentic_kv_offload_none
fi

SPEC_ARGS=()
if [ "$SPEC_DECODING" = "mtp" ]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

export MODEL_PATH=/models/GLM-5.3-W4AFP8
# $MODEL is zai-org/GLM-5.3-FP8, which does not exist on the Hub (only
# zai-org/GLM-5.3 and zai-org/GLM-5.3-BF16 do), and aiperf's dataset manager
# loads a tokenizer by --model unless --tokenizer overrides it. Point it at the
# checkpoint's own tokenizer instead of a name that 404s -- it is also the
# exactly-correct tokenizer, not a same-family stand-in.
export AIPERF_TOKENIZER=$MODEL_PATH
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
# Run 32585352758 OOMed at c16/c24 on a 2.38-2.53 GiB DeepEP dispatch buffer with
# 5.4 GiB sitting reserved-but-unallocated. Expandable segments let the caching
# allocator grow a segment instead of needing that much contiguous free space.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
  USE_SGLANG_ROUTER=true
  SGLANG_BACKEND_PORT=$((PORT + 1))
  SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:$SGLANG_BACKEND_PORT/metrics"

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MAX_RUNNING_REQUESTS=$((2 * CONC))
CHUNKED_PREFILL_SIZE=8192
PARALLEL_ARGS=(--tp-size "$TP")
GRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
  [ "$MAX_RUNNING_REQUESTS" -lt "$TP" ] && MAX_RUNNING_REQUESTS=$TP
  CHUNKED_PREFILL_SIZE=32768
  PARALLEL_ARGS=(
    --tp "$TP"
    --dp "$TP"
    # Passed explicitly only so RESULT_FILENAME records ep8 instead of ep1:
    # overrides.py:_a2a_ep_size forces ep_size = tp_size for every A2A-spanning
    # backend, DeepEP included, whether or not --ep is given.
    --ep "$EP_SIZE"
    --enable-dp-attention
    # dp == tp here, so attn_tp_size is 1: local-control-broadcast drops the
    # all-ranks gloo control sync per scheduler iteration, and dp-lm-head
    # drops the cross-DP logits all-gather (logits_processor.py gates its own
    # all-gather on attn_tp_size > 1, so a group of 1 skips both).
    --enable-dp-attention-local-control-broadcast
    --enable-dp-lm-head
    --tokenizer-worker-num "$TP"
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
  )
else
  CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
  [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
  GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

SGLANG_CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --quantization w4afp8
  --disable-shared-experts-fusion
  # deepep_mode defaults to auto, which DeepEPMode.resolve splits per batch:
  # NORMAL for extend, LOW_LATENCY for decode -- the right split for one
  # aggregated server serving both phases, so no explicit mode.
  # --disable-shared-experts-fusion above stays: only Waterfill (forces it on)
  # and the flashinfer A2A backend (forces it off) override it, neither in play.
  # It OOMed, as predicted: run 32585352758 died at c16/c24 ~2 min into serving.
  # Measured on that run -- DeepEP's graph capture costs 10.65 GB vs glm5.2dplm's
  # 3.74 GB (a2a buffers get pinned into the graph pool), leaving only 9.82 GB
  # free at ready, and ~7.4 GiB of the crash footprint is non-PyTorch DeepEP
  # NVSHMEM/IPC allocated outside mem-fraction-static entirely. 0.80 frees
  # 6.99 GiB for a 2.53 GiB alloc (2.8x margin) and costs ~16% of the KV pool,
  # which is close to free here: dplm c24 already peaks at token usage 1.00 and
  # cache hit is at 99.76% of the workload's theoretical ceiling, so the device
  # pool is not the binding constraint. The glm5.2dplm pair is no longer exact
  # -- dplm ran 0.85. Do NOT "match prod" with max-running-requests 256: graphs
  # are captured up to max_running_requests/dp_size, so 256 means bs 32 (20
  # graphs, ~35 GB) instead of bs 6, and peak observed #running-req is 5-6.
  --moe-a2a-backend deepep
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.80
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  "${GRAPH_ARGS[@]}"
  --context-length 500000
  --kv-cache-dtype fp8_e4m3
  # Prod's prefill kernel. Without this, overrides.py auto-detects
  # flashmla_kv for fp8_e4m3 on Hopper (trtllm only from SM100), so the
  # native FP8 SM90 sparse-prefill path never runs in the sibling recipes.
  # Prefill-only on purpose: dsa_backend.py rejects it as a decode backend
  # and auto-detect leaves decode on flashmla_kv, which is what prod does.
  --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics
  --enable-cache-report
  "${CACHE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  --schedule-policy lpm
  --served-model-name "$MODEL"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
  python3 -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy cache_aware \
    --request-id-headers x-correlation-id \
    --dp-aware \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
    --connect-timeout-secs 900 \
    --request-timeout-secs 14400 \
    --disable-health-check \
    --disable-retries > "$ROUTER_LOG" 2>&1 &
  ROUTER_PID=$!
  wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
