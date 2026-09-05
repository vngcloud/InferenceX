#!/usr/bin/env bash
set -euo pipefail
set -x

# Low-latency (TP-only) recipe for GLM-5.2 on H200: byte-identical to
# glm5.2ll7cp_fp4_h200_sglang.sh plus one flag, --enable-cp-decode-attn-tp,
# i.e. prefill CP with the replicated-decode mitigation turned on. Kept as
# its own file to get its own MODEL_PREFIX, since the conc ladder overlaps
# ll7cp's and agg_bmk.json tells arms apart by prefix. Driven with dp-attn
# false, so the DP branch below is dead code here: no router,
# chunked-prefill 8192, --tp-size only, --cuda-graph-max-bs min(2*CONC, 64).
#
# NOTE (v0.5.18, unchanged from ll7cp): the DSA override in sglang's
# arg_groups/overrides.py forces attn_cp_size = tp/dp (=8 here), so the CLI
# --attn-cp-size 2 below is overridden to 8 and attn_tp becomes 1 (attention
# fully replicated). Prefill CUDA graphs are disabled; decode keeps its
# graphs.
#
# WHY THIS ARM. Run 33961987006 (ll7cp) vs 33249899782 (ll7) measured mean
# ITL 10.91 vs 9.14 ms at conc 4 (+19.4%) and 15.36 vs 13.75 ms at conc 8
# (+11.7%), with total throughput down 6.6% and 2.4%. Decode adds no CP
# collectives on this path, so that cost is redundant work, not
# communication: with attn_tp=1 every rank runs the full decode attention
# instead of 1/8 of it. --enable-cp-decode-attn-tp slices the replicated
# attention linears (q_b_proj, o_proj, w_kc/w_vc and their scales) to the
# local CP partition on non-prefill-CP forwards, which is exactly that
# 8x redundancy. This arm prices the mitigation.
#
# PRECONDITIONS VERIFIED against the v0.5.18 tag (all four must hold or the
# flag is a silent no-op or a hard abort):
#   1. server_args.py:5270 aborts unless model_arch is in
#      CP_DECODE_ATTN_TP_SUPPORTED_ARCHS; GLM-5.2 reports
#      GlmMoeDsaForCausalLM, which is on that list.
#   2. deepseek_v2.py:maybe_use_decode_attn_tp returns a bare yield when
#      q_lora_rank is None. GLM-5.2's config.json has q_lora_rank 2048, so
#      the slicing path is live.
#   3. cp_decode_attn_tp.py:_activate asserts shape[dim] % decode_tp_size.
#      64 attention heads over decode_tp_size 8 is 8 heads/rank, exact.
#   4. The context only slices when neither is_cp_v2_active nor
#      dsa_use_prefill_cp holds, so prefill CP keeps all heads and only
#      decode is sharded. The two flags do not fight.
#
# NOT EXPECTED TO FIX conc 12. ll7cp's conc-12 blowup (TTFT p90 7353 vs
# 4681 ms) is memory, not compute: replication cuts the KV pool to 967,424
# from 1,173,696 tokens/rank, prefix hit falls .914 from .942 and the
# evicted prefixes get recomputed. This flag does not shrink the resident
# weights -- _slice_cache holds the full tensor alongside the slice so
# _restore can put it back, and dim-1 slices call .contiguous(), so the
# footprint is flat-to-slightly-up. Hence the ladder below is [4, 8]: the
# two points where the regression it targets actually lives.
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION DP_ATTENTION SPEC_DECODING

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
    --speculative-num-steps 7
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8
  )
fi

# Prefill context parallelism, carried over from ll7cp unchanged. NOTE: the
# DSA override forces attn_cp_size = tp/dp regardless of the CLI value, so
# --attn-cp-size 2 below resolves to 8 on this TP-only profile; kept at 2 to
# document intent and stay correct if the override ever stops forcing it.
# --enable-cp-decode-attn-tp is the one variable under test vs ll7cp.
CP_ARGS=(
  --attn-cp-size 2
  --enable-prefill-cp
  --cp-strategy interleave
  --enable-cp-decode-attn-tp
)

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
  "${CP_ARGS[@]}"
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
