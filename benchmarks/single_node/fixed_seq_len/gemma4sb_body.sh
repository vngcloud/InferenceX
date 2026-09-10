#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM -- STAGE 5 of the optimization
# roadmap: SPEED-Bench (real text, stratified by entropy) instead of the
# synthetic 8k1k counting sequence. SHARED BODY.
#
# This file is NOT dispatchable on its own. runners/launch_h200-greennode.sh
# builds the recipe path from the matrix model-prefix, so every
# (arm x category x eos-mode) cell needs its own filename. The sixteen
# dispatchable wrappers each set three variables and source this body, so the
# cell coordinates are provably the only thing that differs across arms.
#
#   SB_ARM        base | e3 | dflash | mtp      -- which speculator, if any
#   SB_CATEGORY   low_entropy | mixed | high_entropy
#   SB_IGNORE_EOS 1 | 0                          -- default 1
#
# WHY A NEW STAGE INSTEAD OF MORE ARMS ON STAGE 1-3.
# Stages 0-3 measured on utils/bench_serving/benchmark_serving.py's random
# dataset, whose prompts are a counting sequence
# (benchmark_serving.py:319: prefix + [(offset + i + j) % vocab_size ...]).
# A drafter's agreement with the target on a counting sequence is a FLOOR, not
# a production estimate -- Stage 2 read acceptance 1.211 at conc 1 that way.
# SPEED-Bench is real text with a declared entropy stratification, so the
# acceptance numbers here are the ones that should drive the adoption call.
#
# Swapping the client swaps the workload, so NOTHING here is comparable to the
# Stage 0-3 tables. That is why SB_ARM=base exists: the gate denominator must
# be a no-spec arm measured on THIS dataset, at the same concurrency and
# category, or the ">=15%" comparison has a denominator from another workload.
#
# WHY THIS CANNOT USE run_benchmark_serving.
# The vendored client declares choices=["random"] for --dataset-name
# (utils/bench_serving/benchmark_serving.py:999), so it cannot address
# speed_bench at all. We call `vllm bench serve` directly out of the same
# vllm/vllm-openai:v0.28.0 image the server runs in. benchmarks/benchmark_lib.sh
# is deliberately NOT touched: it is shared by every other recipe in the repo.
# The precedent for this is benchmarks/single_node/speedbench/*.sh, which have
# been calling `vllm bench serve --dataset-name speed_bench` in production.
#
# PREFIX CACHING IS OFF, ON PURPOSE.
# Seven of the nine existing speedbench recipes pass --no-enable-prefix-caching
# (dsv4:239, dsr1:150, glm5:259, glm52:150, kimik2.5:276, qwen3.5:261,
# minimaxm3:191). With APC on, the `mixed` ignore-eos=0 control would replay the
# exact prompt set the ignore-eos=1 cell already prefilled, and its TTFT would
# collapse against a warm 8k prefix. Off, there is no cross-request KV reuse to
# contaminate anything, and no need for /reset_prefix_cache (which sits behind
# VLLM_SERVER_DEV_MODE and is one more thing that can be missing).
# This is a deliberate divergence from Stages 0-3, which ran with APC on at the
# v1 default. It is safe precisely because Stage 5 carries its own base arm.
#
# ACCEPTANCE COMES FROM PROMETHEUS, NOT FROM THE SERVER LOG.
# vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_drafts_total
# give AL = 1 + accepted/drafts directly. Three reasons this replaces the
# log-scraping used for Stage 2:
#   1. num_drafts counts draft EVENTS, so the formula is independent of
#      num_speculative_tokens -- one expression covers every depth.
#   2. No drain-phase bias. The last "SpecDecoding metrics" block in a Stage-2
#      server log read 1.06 at conc 64 where the whole run was 1.496.
#   3. The counters are exact totals, not a sum over 10s interval blocks.
# One job serves exactly one benchmark, so the counters would start at zero
# anyway; the before/after snapshot below costs four lines and removes the
# assumption entirely.
#
# NUM-PROMPTS. Pinning one category caps the pool at 512 rows (SPEED-Bench
# throughput_* configs are 1,536 rows = 512 x {low,mixed,high}; qualitative is
# 880 = 11 categories x ~80). So CONC*10 is clamped to [64, 512]: 320 at conc
# 32, 512 at conc 64. Every arm at a given concurrency therefore draws the
# identical prompt set, which is what makes arm-vs-arm acceptance comparable.
# Comparing acceptance ACROSS concurrencies is not supported by this design.
#
# NO WARMUP REQUESTS, matching the speedbench/ precedent (dsv4 issues none).
# vLLM captures CUDA graphs before the server reports ready, and 320-512
# prompts dilute any residual cold-start. All arms are treated identically.
#
# ISL LABELLING. The matrix key declares isl 8192 so process_result.py tags the
# row sensibly, but SPEED-Bench throughput_8k prompts are only approximately 8k.
# total_input_tokens in the result JSON is the authoritative number.
#
# ENGINE CONFIG is copied verbatim from gemma4v28_fp8block_h200.sh (Stage 1)
# apart from --no-enable-prefix-caching, the per-arm --speculative-config, and
# BENCH_GPU: 4 here instead of Stage 1's 6 (operator request, re-pinned after
# the first dispatch was canceled with no benchmark job completed; same node,
# same H200 SKU, so nothing measured changes). The dead
# VLLM_ATTENTION_BACKEND=FLASHINFER and the absence of --kv-cache-dtype are
# inherited deliberately: fp8 KV pins gemma4 to Triton
# on SM90 and costs +72% TTFT / +25% TPOT / -20% req/s. Power telemetry is not
# comparable on this branch. Pin the host with
# --runner-node-filter h200-greennode_07.
#
# ROADMAP GATE. Adopt a speculative arm only if it lifts output throughput by
# >=15% against the SB_ARM=base cell at the SAME concurrency and category,
# without worsening p99 TTFT or p99 ITL by more than 5%.
#
# Exploration-only: dispatched via e2e-tests.yml against a branch, never merged.

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

: "${SB_ARM:?set by the wrapper that sources this file}"
: "${SB_CATEGORY:?set by the wrapper that sources this file}"
SB_IGNORE_EOS="${SB_IGNORE_EOS:-1}"

BENCH_GPU=4
SB_CONFIG="${SB_CONFIG:-throughput_8k}"
SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"

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

# ---- Per-arm speculative configuration -------------------------------------
# Each drafter ships as its own repo; prefetch it so the --speculative-config
# "model" id resolves offline exactly the way the target does.
SPEC_ARGS=()
DRAFT_MODEL=""
case "$SB_ARM" in
    base)
        NUM_SPEC_TOKENS=0
        ;;
    e3)
        # Byte-identical to the Stage-2 arm: this exact JSON ran green in
        # Actions run 33041047279, so the drafter is known-loadable.
        DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.eagle3"
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
        SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"eagle3\"}")
        ;;
    dflash)
        # "dflash" is a valid SpeculativeMethod at tag v0.28.0
        # (vllm/config/speculative.py: DFlashModelTypes is folded into
        # EagleModelTypes, which is folded into SpeculativeMethod). The model
        # card still says "requires vllm nightly / PR #42095" and "Validated on
        # Nvidia H100, other hardware validation pending" -- the Literal being
        # accepted means the CONFIG parses, not that the checkpoint LOADS
        # against a multimodal Gemma-4 target. Preflight this arm on its own
        # before dispatching the sweep.
        DRAFT_MODEL="RedHatAI/gemma-4-31B-it-speculator.dflash"
        NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
        SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"dflash\"}")
        ;;
    mtp)
        # method MUST be "mtp", not "draft_model": the assistant checkpoint
        # consumes the target's hidden states (backbone_hidden_size 5376) and
        # is not a standalone LM. Getting this wrong is an init crash on a
        # multimodal target, not a slow run. See gemma4v28mtp_body.sh.
        DRAFT_MODEL="google/gemma-4-31B-it-assistant"
        : "${NUM_SPEC_TOKENS:?the mtp wrapper must pin the Stage-3 winning depth}"
        SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}")
        ;;
    *)
        echo "CRITICAL: unknown SB_ARM='$SB_ARM' (expected base|e3|dflash|mtp)" >&2
        exit 1
        ;;
esac

if [[ -n "$DRAFT_MODEL" && "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi

# ---- SPEED-Bench dataset ----------------------------------------------------
# Not auto-downloaded by vllm bench serve. --dataset-path is the DIRECTORY the
# prepare script writes {config}.jsonl into, not the file itself.
#
# The prepared throughput_8k.jsonl is COMMITTED alongside this script instead
# of being fetched per job. The runner's route to raw.githubusercontent.com and
# gutenberg.org crawls at ~12 kB/s: run 34492930930 died on BAMBOO
# meetingpred_16k.jsonl (5.43 MB) at aiohttp's 5-minute cap, and the map itself
# ran 6.84 s/example = ~3 h per cell. The committed file is byte-identical to
# NeMo's prepare.py --config throughput_8k with the hle branch disabled:
# cais/hle is GATED on the Hub and the runner token
# (INFERENCEX_OFFICIAL_RO_HF_TOKEN) has not accepted its terms, so the unpatched
# prep dies mid-map on the first hle row (run 34468639874: all sixteen cells
# failed at 513/1536). hle feeds ONLY the mixed category (268/512 rows);
# low_entropy (repobench / AdaLEval textsort / lca-code-completion) and
# high_entropy (BAMBOO / gutenberg) rows are fully resolved from public
# sources. That keeps the mx wrappers INVALID until someone accepts the gate
# on the token's account; the hi/lo cells this stage benches are unaffected.
# The network path stays as a fallback for other SB_CONFIG values and fails
# loudly on its own -- no silent bad data.
if [[ -f "$(dirname "$0")/speed_bench_${SB_CONFIG}.jsonl" ]]; then
    echo "=== Using committed SPEED-Bench dataset ($SB_CONFIG) ==="
    mkdir -p "$SPEEDBENCH_DIR"
    cp "$(dirname "$0")/speed_bench_${SB_CONFIG}.jsonl" "$SPEEDBENCH_DIR/$SB_CONFIG.jsonl"
else
    echo "=== Downloading SPEED-Bench dataset ($SB_CONFIG) ==="
    pip install -q datasets tiktoken pandas
    curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
      | sed 's|^    elif BenchmarkDataset\.HLE\.value in example\["source"\]:$|    elif False:  # stage-5 hle skip|' \
      | python3 - --config "$SB_CONFIG" --output_dir "$SPEEDBENCH_DIR"
fi

if [[ ! -f "$SPEEDBENCH_DIR/$SB_CONFIG.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed -- $SPEEDBENCH_DIR/$SB_CONFIG.jsonl not found" >&2
    exit 1
fi
wc -l "$SPEEDBENCH_DIR/$SB_CONFIG.jsonl"

# CONC*10 clamped to the single-category pool size [64, 512].
SB_NUM_PROMPTS=$((CONC * 10))
if (( SB_NUM_PROMPTS > 512 )); then SB_NUM_PROMPTS=512; fi
if (( SB_NUM_PROMPTS < 64 )); then SB_NUM_PROMPTS=64; fi

EOS_ARGS=()
if [[ "$SB_IGNORE_EOS" == "1" ]]; then EOS_ARGS=(--ignore-eos); fi

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL
export VLLM_ATTENTION_BACKEND=FLASHINFER

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    # Requested 64k serve ceiling; overrides the scenario-computed ~9472.
    MAX_MODEL_LEN=65536
fi

# Scientific-notation-safe counter read. dsv4's [0-9.]+ silently truncates a
# Prometheus value printed as 1.234e+06, which is reachable at 512 prompts x
# 1024 tokens x depth 8.
fetch_metric() {
    local v
    v=$(curl -s "http://localhost:${1}/metrics" \
        | grep -oP "${2}(\\{[^}]*\\})? \\K[0-9.eE+-]+" | head -1)
    printf '%s' "${v:-0}"
}

start_gpu_monitor

set -x
CUDA_VISIBLE_DEVICES="$BENCH_GPU" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --no-enable-prefix-caching \
    "${SPEC_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Stage-5 integrity check. Print the verdicts into the job log rather than
# inferring them later: if v0.28.0 resolves a different attention backend, or
# resolves the mtp assistant as method='draft_model', the comparison is void.
echo "===== stage 5 cell ====="
echo "arm=$SB_ARM category=$SB_CATEGORY ignore_eos=$SB_IGNORE_EOS conc=$CONC"
echo "draft=${DRAFT_MODEL:-none} num_speculative_tokens=$NUM_SPEC_TOKENS"
echo "dataset=$SB_CONFIG num_prompts=$SB_NUM_PROMPTS output_len=$OSL"
echo "===== attention backend selection ====="
grep -E "attention backend|FlashAttention version|kv_cache_dtype|Using .* backend" "$SERVER_LOG" || true
echo "===== prefix caching ====="
grep -E "prefix cach|enable_prefix_caching" "$SERVER_LOG" || true
echo "===== speculative decoding ====="
grep -E "[Ss]peculative|num_speculative_tokens|drafter|[Ee]agle|MTP|[Dd][Ff]lash" "$SERVER_LOG" | head -40 || true
echo "======================================="

# Flag-name audit. The speed-bench options are new in v0.28.0 and a wrong name
# fails the run instantly; dumping the accepted spelling makes the first
# failure self-diagnosing instead of requiring a second dispatch.
echo "===== vllm bench serve: speed-bench options ====="
vllm bench serve --help 2>&1 | grep -iE "speed.bench|ignore-eos|dataset-name|dataset-path" || true
echo "======================================="

ACC_BEFORE=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
DRF_BEFORE=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")
DRT_BEFORE=$(fetch_metric "$PORT" "vllm:spec_decode_num_draft_tokens_total")

vllm bench serve \
    --backend openai \
    --model "$MODEL" \
    --port "$PORT" \
    --dataset-name speed_bench \
    --dataset-path "$SPEEDBENCH_DIR" \
    --speed-bench-dataset-subset "$SB_CONFIG" \
    --speed-bench-category "$SB_CATEGORY" \
    --speed-bench-output-len "$OSL" \
    --num-prompts "$SB_NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --request-rate inf \
    "${EOS_ARGS[@]}" \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --save-result \
    --result-dir /workspace/ \
    --result-filename "${RESULT_FILENAME}.json" \
    --trust-remote-code

ACC_AFTER=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
DRF_AFTER=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")
DRT_AFTER=$(fetch_metric "$PORT" "vllm:spec_decode_num_draft_tokens_total")

set +x

D_ACC=$(awk "BEGIN{printf \"%d\", $ACC_AFTER - $ACC_BEFORE}")
D_DRF=$(awk "BEGIN{printf \"%d\", $DRF_AFTER - $DRF_BEFORE}")
D_DRT=$(awk "BEGIN{printf \"%d\", $DRT_AFTER - $DRT_BEFORE}")
# AL = mean accepted length per draft event, including the always-accepted
# target token. Independent of num_speculative_tokens.
AL=$(awk "BEGIN{if ($D_DRF>0) printf \"%.4f\", 1 + $D_ACC/$D_DRF; else printf \"NaN\"}")
# Per-drafted-token acceptance rate; NaN for the base arm.
ACC_RATE=$(awk "BEGIN{if ($D_DRT>0) printf \"%.4f\", $D_ACC/$D_DRT; else printf \"NaN\"}")

echo "===== stage 5 acceptance ====="
echo "arm=$SB_ARM category=$SB_CATEGORY conc=$CONC ignore_eos=$SB_IGNORE_EOS depth=$NUM_SPEC_TOKENS"
echo "accepted_tokens=$D_ACC draft_events=$D_DRF drafted_tokens=$D_DRT"
echo "acceptance_length=$AL acceptance_rate=$ACC_RATE"
echo "=============================="

# Same block into the server log, which the server_logs_* artifact uploads, and
# a machine-readable sidecar under results/ which the same artifact picks up via
# its results/*.log glob. agg_*.json cannot carry these: process_result.py only
# forwards keys ending in "ms".
{
    echo "===== stage 5 acceptance ====="
    echo "arm=$SB_ARM category=$SB_CATEGORY conc=$CONC ignore_eos=$SB_IGNORE_EOS depth=$NUM_SPEC_TOKENS"
    echo "accepted_tokens=$D_ACC draft_events=$D_DRF drafted_tokens=$D_DRT"
    echo "acceptance_length=$AL acceptance_rate=$ACC_RATE"
} >> "$SERVER_LOG"

mkdir -p "${RESULT_DIR:-/workspace/results}"
SB_ARM="$SB_ARM" SB_CATEGORY="$SB_CATEGORY" SB_IGNORE_EOS="$SB_IGNORE_EOS" \
SB_CONFIG="$SB_CONFIG" SB_NUM_PROMPTS="$SB_NUM_PROMPTS" \
NUM_SPEC_TOKENS="$NUM_SPEC_TOKENS" DRAFT_MODEL="${DRAFT_MODEL:-}" \
D_ACC="$D_ACC" D_DRF="$D_DRF" D_DRT="$D_DRT" AL="$AL" ACC_RATE="$ACC_RATE" \
python3 - <<'PYEOF'
import json, os
from pathlib import Path

result = Path("/workspace") / f"{os.environ['RESULT_FILENAME']}.json"
data = json.loads(result.read_text())

# process_result.py reads these two unconditionally (lines 132/134) and dies on
# a missing or null value. `vllm bench serve` normally writes both; backfill so
# a key rename upstream costs a rerun of the client, not of the whole job.
if data.get("max_concurrency") in (None, ""):
    data["max_concurrency"] = int(os.environ["CONC"])
if data.get("model_id") in (None, ""):
    data["model_id"] = os.environ["MODEL"]

cell = {
    "sb_arm": os.environ["SB_ARM"],
    "sb_category": os.environ["SB_CATEGORY"],
    "sb_ignore_eos": int(os.environ["SB_IGNORE_EOS"]),
    "sb_dataset_subset": os.environ["SB_CONFIG"],
    "sb_num_prompts": int(os.environ["SB_NUM_PROMPTS"]),
    "num_speculative_tokens": int(os.environ["NUM_SPEC_TOKENS"]),
    "draft_model": os.environ["DRAFT_MODEL"] or None,
    "spec_accepted_tokens": int(os.environ["D_ACC"]),
    "spec_draft_events": int(os.environ["D_DRF"]),
    "spec_drafted_tokens": int(os.environ["D_DRT"]),
    "acceptance_length": None if os.environ["AL"] == "NaN" else float(os.environ["AL"]),
    "acceptance_rate": None if os.environ["ACC_RATE"] == "NaN" else float(os.environ["ACC_RATE"]),
}
data.update(cell)
result.write_text(json.dumps(data, indent=2))

sidecar = Path(os.environ.get("RESULT_DIR", "/workspace/results"))
sidecar.mkdir(parents=True, exist_ok=True)
(sidecar / f"specdec_{os.environ['RESULT_FILENAME']}.log").write_text(
    json.dumps(cell, indent=2)
)
print("stage5 cell metadata written")
PYEOF

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
