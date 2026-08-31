#!/usr/bin/env bash
set -euo pipefail
set -x

# glm5.2deep (tp8/dp8/ep8 DeepEP + DP-attention, EAGLE 3/1/4 fixed) with two
# additions and nothing else:
#   1. an in-container apply of sglang PR #35898 over the installed sglang, and
#   2. --speculative-adaptive with the boot depth raised to 7/8.
# Everything else -- the DeepEP flags, the three DP-attention sync flags,
# --dsa-prefill-backend flashmla_sparse_q8, hicache ratio, mem-fraction 0.80,
# lpm, --disable-shared-experts-fusion -- is byte-identical to glm5.2deep, so
# this arm pairs directly against the glm5.2deep fixed-depth twin at conc
# [8, 16, 24].
#
# WHY THE PATCH. Stock v0.5.18 hard-disables --speculative-adaptive under
# --enable-dp-attention: adaptive_spec_params.py:adaptive_unsupported_reason
# rejects enable_dp_attention ("tier decisions are not synchronized across DP
# ranks"). The rejection is a logger.warning plus a SILENT fallback to static
# params, not an error. PR #35898 (ours, open upstream, fork
# thangquang09/sglang exp/adaptive-spec-dp-attention) removes that guard and
# adds cross-rank tier consensus over the existing MLP-sync all_gather (each DP
# rank votes num_steps, tier = min() over ranks, all ranks activate in
# lockstep so num_draft_tokens never diverges and the NCCL collective cannot
# hang). The patch here is the 5 runtime files of that PR (test file dropped);
# it applies cleanly on v0.5.18 -- 4/5 files are identical between the PR base
# and the v0.5.18 tag, and the schedule_batch.py hunk applies at offset -9.
# Because the guard fails silently, "speculative_adaptive=True" in the
# server_args dump AND the "AdaptiveSpeculativeParams initialized" line are the
# only proof the feature actually ran; the "Switch adaptive runtime state"
# lines are the proof the dp8 consensus fired without deadlocking. Check all
# three before reading the numbers.
#
# This is an ENABLEMENT run, not a prod-win claim. On a high-acceptance
# workload adaptive holds its tier and ties fixed-depth (zero-overhead
# plumbing); it only downshifts when acceptance degrades or batch grows. The
# cc-traces trace is hash-encoded tokens, which understates draft acceptance,
# so any adaptive-over-fixed edge here is a best case that does not transfer to
# real prod traffic.
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING

# Apply PR #35898 over the installed sglang before anything imports it. The
# recipe runs inside the sglang container, so /sgl-workspace/sglang is the
# editable install and the patch takes effect on import. Idempotent-guarded so
# a re-run of the same container does not fail on an already-applied hunk.
ADAPTIVE_DP_PATCH="$(dirname "$0")/../../patches/adaptive-dp-35898.patch"
if [ ! -f "$ADAPTIVE_DP_PATCH" ]; then
  echo "FATAL: adaptive-dp patch not found at $ADAPTIVE_DP_PATCH" >&2
  exit 1
fi
# Resolve to an absolute path BEFORE the pushd below: the container runs with
# -w /workspace and the path above is relative to it, so it would not resolve
# once the cwd changes into the sglang tree.
ADAPTIVE_DP_PATCH="$(readlink -f "$ADAPTIVE_DP_PATCH")"
pushd /sgl-workspace/sglang >/dev/null
if git apply --check -p1 "$ADAPTIVE_DP_PATCH" 2>/dev/null; then
  git apply -p1 --verbose "$ADAPTIVE_DP_PATCH"
  echo "adaptive-dp patch applied"
elif git apply --check -p1 -R "$ADAPTIVE_DP_PATCH" 2>/dev/null; then
  echo "adaptive-dp patch already applied (reverse-check passed), skipping"
else
  echo "FATAL: adaptive-dp patch neither applies nor is already applied" >&2
  exit 1
fi
popd >/dev/null

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
    # Boot at the widest candidate. No --speculative-adaptive-config is passed,
    # so DEFAULT_ADAPTIVE_CONFIG applies (candidate union {0, 1, 3, 7}) and
    # speculative_hook.py hard-raises unless boot num-steps is in that union --
    # 3 boots but the fixed twin already covers 3, and booting below the widest
    # tier risks the boot-sized shared logits buffer; 7 makes max_rows the
    # ceiling by construction. topk 1 and num-draft-tokens = max(candidate)+1 = 8
    # are both preconditions of adaptive, not free choices.
    --speculative-num-steps 7
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8
    --speculative-adaptive
  )
fi

export MODEL_PATH=/models/PhalaCloud/GLM-5.2-W4AFP8
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
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
    --ep "$EP_SIZE"
    --enable-dp-attention
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
