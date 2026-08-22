#!/usr/bin/env bash
set -euo pipefail
set -x

# glm5.2deep_fp4_h200_sglang.sh plus --speculative-skip-dp-mlp-sync, so the
# pair with glm5.2deep reads that one flag and nothing else. Everything the
# stacked arm carries -- the two DP-attention sync flags, flashmla_sparse_q8,
# --moe-a2a-backend deepep with --ep 8, dp == tp, mem-fraction 0.85, lpm,
# hicache ratio from the config, EAGLE 3/1/4 -- is unchanged.
#
# What the flag does: scheduler.py:3111-3116 runs an extra DP-wide
# maybe_prepare_mlp_sync_batch before merging a new batch whenever spec
# decoding and DP attention are both on. This skips it. The NOTE at that
# branch says the sync is what keeps prefill and decode batches from mixing
# under spec + dp-attn, so expect batch composition -- and therefore accept
# length -- to move, not just the per-iteration sync cost. Read accept
# len/rate alongside throughput or the delta is unattributable.
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
    # Inside the mtp branch on purpose: speculative_hook.py:132 asserts
    # speculative_algorithm == "EAGLE" whenever this flag is set, so passing it
    # unconditionally would abort boot on any spec-off run of this recipe.
    --speculative-skip-dp-mlp-sync
  )
fi

export MODEL_PATH=/models/PhalaCloud/GLM-5.2-W4AFP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics

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
  # ponytail: kept mem-fraction-static at 0.85 to hold the pair with glm5.2dplm,
  # but DeepEP allocates its NVLink dispatch buffers outside the static fraction
  # and nothing auto-reduces it. If the warmup request OOMs, drop to 0.75 (what
  # glm5.2edeep/glm5.2pdeep use) and record that the pair is no longer exact.
  --moe-a2a-backend deepep
  --host 0.0.0.0
  --port "$SGLANG_BACKEND_PORT"
  "${PARALLEL_ARGS[@]}"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --mem-fraction-static 0.85
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
