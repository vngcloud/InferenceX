#!/usr/bin/env bash
set -eo pipefail
set -x

# Gemma-4-31B customer-service (tau2) agentic arm, SGLang MTP4 variant — the
# engine-side A/B partner of benchmarks/single_node/agentic/
# gemma4tau2_fp8_h200_mtp.sh (vLLM MTP4, refs/bench/gemma4tau2-mtp-vllm): same
# private tau2 corpus replay (thangquang09/customer-service-agent-traces,
# mooncake_trace), same agentic-coding scenario, same conc ladder, single
# replica on one GPU. Only the engine differs:
#   vLLM v0.30.0 (mtp speculative-config)   SGLang patched v0.5.21 image B
#   prefix caching ON (default)             radix cache ON (SWA-aware; no
#                                           --disable-radix-cache)
#   FA4 auto (VLLM_ATTENTION_BACKEND is     --attention-backend fa4
#   unknown on v0.30.0, ignored)
#   --max-num-seqs 64                       --max-running-requests capped at 32
#                                           (FROZEN_KV_MTP SWA pool aborts at
#                                           64 slots, ~107 GiB)
#   fcfs (vLLM has no policy knob)          --schedule-policy hrrn (token-based
#                                           aging; cache-aware; falls back to
#                                           fcfs above a 128-deep queue)
# Plus the tuned knobs today's gemma4sba probes selected (probes c/e/f + tuned
# smokes + tuned sweep, refs/bench/gemma4-sglang-tuned @ e46c9d30):
#   --mem-fraction-static 0.92, SGLANG_SWA_EVICTION_INTERVAL=32,
#   --chunked-prefill-size 8192, NEXTN->FROZEN_KV_MTP auto-promote with the
#   google/gemma-4-31B-it-assistant draft at num-steps 4 / topk 1 / 5 draft
#   tokens (= vLLM num_speculative_tokens 4).
# KV fix (run 37802254543 postmortem): default --swa-full-tokens-ratio 0.8
# sized the SWA pool 105,728 tok (90.75 GB, ~10% used) and starved the
# full-attention pool to 132,161 tok -> 98-100% usage + retraction at c16+.
# 0.2 shifts the split toward the full pool (tau2 needs ~14k tok/req of
# full-attn KV; 16 conc ~= 224k); SWA worst case stays far above the
# window+page admission floor.
# Image: ghcr.io/noridom1/sglang:v0.5.21-gemma4-fa4-swapool-3a7d5ad =
# release/v0.5.21 + sglang#42019 (Gemma4 FA4 on SM90 head-dim 512; stock falls
# back to Triton) + the SWA-KV-pool fix for FA under FROZEN_KV_MTP
# (Noridom1/sglang gemma4-v0.5.21-fa4-swapool @ 3a7d5ad64).
# context-length 262144 + BF16 KV (auto) match the vLLM arm for A/B parity.
#
# Required env (harness-provided): MODEL TP CONC KV_OFFLOADING RESULT_DIR
# DURATION PORT. The ambient HF_TOKEN must be able to read the private tau2
# dataset and the gated google/gemma-4-31b-it tokenizer.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION PORT
require_agentic_kv_offload_none

# $MODEL is the un-quantized HF id (aiperf tokenizer + wire name); the engine
# serves the RedHatAI FP8-block quant behind it, MTP draft assistant included.
WEIGHTS="RedHatAI/gemma-4-31B-it-FP8-block"
DRAFT="google/gemma-4-31B-it-assistant"
TAU2_REPO="thangquang09/customer-service-agent-traces"

# aiperf hits the single SGLang replica on $PORT; its /metrics (registered with
# the sglang: prometheus prefix, needs --enable-metrics) feeds the required
# server-metrics gate.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

# benchmark_lib.sh reassigns AIPERF_UV_CACHE_DIR to an ephemeral /tmp dir at
# source time, so every dispatch cold-downloads aiperf's deps from PyPI (~1 GB,
# painfully slow on this box). Restore the launcher's persistent mount so c2 and
# reruns hit the uv cache instead of re-downloading.
if [ -d /mnt/uv-cache ]; then
    export AIPERF_UV_CACHE_DIR=/mnt/uv-cache
fi
install_agentic_deps
nvidia-smi

# Model weights + MTP draft are public -> ambient HF_TOKEN. Idempotent: a cache
# hit is a no-op. Serve by HF id so SGLang resolves from the mounted
# HF_HUB_CACHE.
"$AIPERF_HF_CLI" download "$WEIGHTS"
"$AIPERF_HF_CLI" download "$DRAFT"

# The tau2 corpus is a PRIVATE repo; the ambient HF_TOKEN (the official CI token)
# must be able to read it. Resolve the snapshot dir via snapshot_download (returns
# just the path) instead of parsing `hf download` stdout, which newer CLIs decorate
# with a "✓ Downloaded / path: ..." banner.
TAU2_DIR=$("$AIPERF_PYTHON" -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], repo_type='dataset'))" \
    "$TAU2_REPO")
mapfile -t TAU2_FILES < <(find "$TAU2_DIR" -maxdepth 2 -name '*.jsonl' | sort)
if [ "${#TAU2_FILES[@]}" -ne 1 ]; then
    echo "Error: expected exactly one .jsonl in $TAU2_DIR, found ${#TAU2_FILES[@]}: ${TAU2_FILES[*]}" >&2
    exit 1
fi
TAU2_FILE="${TAU2_FILES[0]}"
echo "tau2 mooncake_trace input: $TAU2_FILE"

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# FROZEN_KV_MTP sizes the SWA pool from the worst case at --max-running-requests
# (draft SWA layers included). At 64 that is ~107 GiB and startup aborts, so MTP
# caps slots at 32 — the same cap the tuned gemma4sba sweep runs with. The tau2
# A/B ladder tops out at 32, so the cap only bites future wider dispatches.
MAX_RUNNING="$CONC"
if [ "$CONC" -gt 32 ]; then
    MAX_RUNNING=32
fi
# Tuned knob from today's probes: engine default is 128; 32 kept the SWA pool
# inside the mem-fraction-static budget at 32 slots without eviction stalls.
export SGLANG_SWA_EVICTION_INTERVAL=32

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$WEIGHTS"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --mem-fraction-static 0.92
    --swa-full-tokens-ratio 0.2
    --context-length 262144
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size 8192
    --attention-backend fa4
    --schedule-policy hrrn
    --enable-metrics
    --enable-cache-report
    --speculative-algorithm NEXTN
    --speculative-draft-model-path "$DRAFT"
    --speculative-num-steps 4
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 5
    --tool-call-parser gemma4
    --reasoning-parser gemma4
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Prove the two image patches took effect: FA4 selected (not the Triton
# fallback), and the Gemma4 draft promoted NEXTN -> FROZEN_KV_MTP.
echo "===== tau2 arm (sglang MTP4 + hrrn) ====="
echo "schedule_policy=hrrn radix=on(max-running=$MAX_RUNNING) swa_evict=$SGLANG_SWA_EVICTION_INTERVAL"
grep -iE "attention.backend|fa4|triton|FROZEN_KV_MTP|speculative" "$SERVER_LOG" | head -20 || true
echo "========================================="

# Replay input override: swap the Weka public-dataset default for the local
# mooncake_trace file. build_replay_cmd appends $TRACE_SOURCE_FLAG verbatim, so
# we set it directly instead of calling the Weka-only resolve_trace_source.
export TRACE_SOURCE_FLAG="--custom-dataset-type mooncake_trace --input-file $TAU2_FILE"
# The local --input-file corpus is unpinned; runs are stamped submission_valid:
# false, which is expected and acceptable.
export AIPERF_UNSAFE_OVERRIDE=true
# Dual stop: 20-minute cap OR one full pass over the 16,798-turn pool, whichever
# fires first -- the cap is what keeps low-CCU smoke points from running forever.
# This is a deliberate recipe-specific override, not a discarded caller input:
# the matrix generator hardcodes duration=3600 for every agentic-coding row
# (infx/matrix/generate.py DEFAULT_AGENTIC_DURATION_SECONDS), with no per-arm
# field to set it. DURATION feeds --benchmark-duration inside build_replay_cmd.
export DURATION=1200

build_replay_cmd "$RESULT_DIR"
# Override the two Weka-shaped assumptions build_replay_cmd hardcodes:
#  - entry cap 393 (with-subagents corpus) -> the full tau2 corpus (1,576 sessions;
#    the loader treats it as min(cap, available)).
#  - add the turn-count half of the dual stop; --request-count and
#    --benchmark-duration coexist, stopping at whichever fires first.
REPLAY_CMD="${REPLAY_CMD/--num-dataset-entries 393/--num-dataset-entries 1576}"
if [[ "$REPLAY_CMD" != *"--num-dataset-entries 1576"* ]]; then
    echo "Error: build_replay_cmd no longer emits '--num-dataset-entries 393'; the tau2 entry-cap override missed. Refresh it before running." >&2
    exit 1
fi
REPLAY_CMD+=" --request-count 16798"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
