#!/usr/bin/env bash

# Gemma-4 31B FP8-block vLLM recipe, fixed-seq-len, EAGLE3 speculative decoding,
# 2-way data parallel (tp1 x dp2 = 2 GPUs).
#
# Spec-decoding sibling of gemma4dp2_fp8block_h200.sh: identical checkpoint,
# image, runner, 8k1k workload and DP2 topology, but adds a --speculative-config
# pointing at RedHat's EAGLE3 speculator, which is trained for this exact target
# checkpoint. A 2B EAGLE3 draft proposes tokens that the 31B target verifies in
# parallel; output is unchanged, decode throughput/TPOT improve when the draft's
# proposals are accepted. gemma-4 EAGLE3 was upstreamed in vLLM ~v0.22.0, so the
# pinned v0.25.0 image supports it with no image change.
#
# DP_SIZE is a constant (see gemma4dp2_fp8block_h200.sh) because
# nvidia-master.yaml has no DP-size field; the "eagle" in this file's name and in
# the gemma4eagle model-prefix records the spec-decoding variant, and the recipe
# path is selected from that prefix (this runner appends no spec-suffix).
#
# NOTE ON ACCEPTANCE: the fixed-seq-len workload feeds *random* token content,
# which understates real EAGLE3 gains -- a draft head cannot predict random
# tokens. --use-chat-template wraps prompts in the tokenizer's chat structure
# (the same mitigation the dsv4 MTP recipes use via --dsv4) so at least the
# structural tokens are predictable, but a faithful acceptance-rate measurement
# needs the agentic (real-trace) scenario. Treat the fixed-seq-len number as a
# conservative floor, not the achievable ceiling.

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
DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

# --max-num-batched-tokens is parameterized (default 8192, the original literal)
# so the gemma4emnbt* wrapper recipes can sweep it without adding a matrix field.
# Left unset, this reproduces the historical DP2 behavior byte-for-byte.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Optional GPU pinning. When GPU_IDS is set (the MNBT sweep sets it to "0,1" on
# h200-greennode_03, whose GPUs 4-7 host another tenant), restrict this server to
# those cards and scope the GPU monitor to match, so we provably never touch the
# other tenant's GPUs. Unset => original behavior (DP2 grabs the first DP_SIZE
# visible cards; the monitor watches all).
if [[ -n "${GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

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

# EAGLE3 draft head lives in a separate repo; pull it into the HF cache so the
# --speculative-config "model" id resolves offline like the target does.
hf download "$DRAFT_MODEL"

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

if [[ -n "${GPU_IDS:-}" ]]; then
    start_gpu_monitor --gpu-ids "$GPU_IDS"
else
    start_gpu_monitor
fi

# --max-num-seqs is per DP rank (see gemma4dp2_fp8block_h200.sh); leaving it at
# CONC keeps it non-binding. --speculative-config carries the EAGLE3 draft; each
# DP replica loads its own copy of the draft and verifies independently.
set -x
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP_SIZE" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# --use-chat-template: see the acceptance note in the header. Spec-decoding
# throughput on raw random tokens is misleadingly low; the chat template is the
# fixed-seq-len mitigation.
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

# --- EAGLE3 spec-decode acceptance (DP2-safe) --------------------------------
# Under --data-parallel-size 2 the stdout "SpecDecoding metrics" lines are
# unreliable (vLLM's stat logger is per-engine and the DP front-end path can
# suppress them), so we read acceptance from the Prometheus /metrics endpoint
# instead. vLLM exposes spec_decode counters per engine (create_metric_per_engine,
# labelled engine="N"): vllm:spec_decode_num_drafts / _num_draft_tokens /
# _num_accepted_tokens. We scrape once after the run (counters are cumulative for
# this fresh container) and aggregate across engines. This is read-only telemetry
# and changes no serving flag, so it never perturbs the measured config.
echo "===== EAGLE3 spec-decode acceptance (from /metrics) ====="
python3 - "$PORT" <<'PY' || echo "(spec-decode metrics scrape failed)"
import sys, re, urllib.request
port = sys.argv[1]
try:
    text = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10).read().decode()
except Exception as e:
    print(f"(could not fetch /metrics: {e})"); sys.exit(0)
lines = [l for l in text.splitlines() if l.startswith("vllm:spec_decode_")]
print("\n".join(lines) if lines else "(no vllm:spec_decode_* series found)")
def total(name):
    pat = r'^%s(?:_total)?(?:\{[^}]*\})?\s+([0-9eE.+-]+)$' % re.escape(name)
    return sum(float(m.group(1)) for m in re.finditer(pat, text, re.M))
drafts = total("vllm:spec_decode_num_drafts")
draft_tokens = total("vllm:spec_decode_num_draft_tokens")
accepted = total("vllm:spec_decode_num_accepted_tokens")
print(f"[spec] drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} accepted={accepted:.0f}")
if draft_tokens > 0:
    print(f"[spec] draft acceptance rate = {accepted/draft_tokens:.4f}")
    if drafts > 0:
        print(f"[spec] mean acceptance length = {1 + accepted/drafts:.4f}")
else:
    print("[spec] no draft tokens recorded (spec decoding inactive or metrics unavailable)")
PY

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
