#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: gemma-4-31b-it serving benchmark on the gateway bench — base vs MTP
#
# Server flags mirror the InferenceX gemma4 MTP run (vngcloud/InferenceX run
# 34502057305): RedHatAI/gemma-4-31B-it-FP8-block, TP1, max_model_len 65536, prefix
# caching off, chunked prefill 16384, MTP = google/gemma-4-31B-it-assistant × 4 tokens.
# The dataset is data/gemma-4-31b-it/bench/2026-09-18_2026-09-25/bench.jsonl instead of SPEED-Bench.
#
# Per arm: one `vllm serve`, a warmup, then for each sampling group × concurrency one
# `vllm bench serve --dataset-name custom --skip-chat-template --custom-output-len -1`
# (per-row prod output length) with spec-decode counters scraped before/after.
# Sampling groups run separately because bench serve only has global sampling flags.
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_gemma4_bench.sh
#   ./scripts/slurm.sh submit scripts/job_gemma4_bench.sh ARMS=mtp "CONCURRENCIES=1 8"
#   ./scripts/slurm.sh submit scripts/job_gemma4_bench.sh BY_CATEGORY=1 \
#       "ARMS=dflash:8 mtp:6" "REUSE_ARMS=<old results>/base:base <old results>/mtp:mtp4"
#
# Arms: `base`, `mtp` (MTP × NUM_SPEC_TOKENS), or `<method>:<k>` with method mtp|dflash —
# result dir `<method><k>` (e.g. mtp:6 → mtp6). REUSE_ARMS symlinks arms from an earlier
# run into this one (same dataset split required) so the summary compares them all.
#
# Results: $RESULTS_DIR/<arm>/<group>/c<conc>/{bench.json,spec_before.json,spec_after.json}
#          $RESULTS_DIR/summary.{json,md}
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gemma4-bench
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
# han-3-1 has a CUDA 12.4 driver; venv-vllm torch is cu130 ("driver too old").
#SBATCH --exclude=hoanq3-h200-1x-han-3-1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/gemma4-bench_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/gemma4-bench_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
BENCH_PATH="${BENCH_PATH:-data/gemma-4-31b-it/bench/2026-09-18_2026-09-25/bench.jsonl}"
MODEL="${MODEL:-RedHatAI/gemma-4-31B-it-FP8-block}"
MTP_MODEL="${MTP_MODEL:-google/gemma-4-31B-it-assistant}"
DFLASH_MODEL="${DFLASH_MODEL:-RedHatAI/gemma-4-31B-it-speculator.dflash}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
ARMS="${ARMS:-base mtp}"
REUSE_ARMS="${REUSE_ARMS:-}"              # "<src arm dir>:<arm name> ..." linked, not rerun
CONCURRENCIES="${CONCURRENCIES:-1 8 32 64}"
BENCH_GROUPS="${BENCH_GROUPS:-}"          # space-separated group slugs; empty = all
BY_CATEGORY="${BY_CATEGORY:-0}"           # 1 = one run per (category, sampling group)
PER_CONC="${PER_CONC:-8}"                 # requests per run = max(MIN_PROMPTS, conc × PER_CONC)
MIN_PROMPTS="${MIN_PROMPTS:-80}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PREFIX_CACHE="${PREFIX_CACHE:-0}"         # 0 = off, like the InferenceX run
IGNORE_EOS="${IGNORE_EOS:-0}"             # 1 = always generate output_tokens tokens
SEED="${SEED:-0}"
PORT="${PORT:-8888}"
RESULTS_DIR="${RESULTS_DIR:-/shared/phucnlt2/bench-results/gemma-4-31b-it/v1/${SLURM_JOB_ID}}"

# ── Environment ──────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/shared/phucnlt2/venv}"
source "$VENV_DIR/bin/activate"
mkdir -p /shared/phucnlt2/job-logs "$RESULTS_DIR"

export HOME=/shared/phucnlt2
export XDG_CACHE_HOME=/shared/phucnlt2/.cache
export HF_HOME=/shared/phucnlt2/hf-cache
export FLASHINFER_WORKSPACE_BASE=/shared/phucnlt2
export TORCHINDUCTOR_CACHE_DIR=/shared/phucnlt2/torchinductor-cache
export TRITON_CACHE_DIR=/shared/phucnlt2/triton-cache
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if [ -d /shared/phucnlt2/python-headers/usr/include/python3.10 ]; then
    PYINC="/shared/phucnlt2/python-headers/usr/include/python3.10:/shared/phucnlt2/python-headers/usr/include"
    export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${PYINC}:${CPLUS_INCLUDE_PATH:-}"
fi

# The vLLM server runs from venv-vllm with its own CUDA wheels (see job_calibrate_entropy.sh).
unset CUDA_HOME
# FlashInfer's sampler JIT-compiles against CUDA_HOME, which pip-only installs lack.
# Both arms use the same sampler, so the comparison is unaffected.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Same for DeepGEMM (default FP8-block GEMM): JIT needs a CUDA toolkit → prebuilt CUTLASS
# block-FP8 kernels instead, for both arms.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

VLLM_BIN="${VLLM_BIN:-/shared/phucnlt2/venv-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
    echo "✗ vLLM not found at $VLLM_BIN — run: ./scripts/slurm.sh setup-vllm"
    exit 1
fi
cd /shared/phucnlt2/speculative-training-pipeline
if [ ! -f "$BENCH_PATH" ]; then
    echo "✗ $BENCH_PATH missing — ./scripts/slurm.sh sync copies it from the local repo"
    exit 1
fi

MAX_CONC=$(echo $CONCURRENCIES | tr ' ' '\n' | sort -n | tail -1)

echo "=== Gemma 4 bench job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Bench:   $BENCH_PATH"
echo "Model:   $MODEL  (MTP: $MTP_MODEL × $NUM_SPEC_TOKENS)"
echo "Arms:    $ARMS   Concurrency: $CONCURRENCIES"
[ -n "$REUSE_ARMS" ] && echo "Reused:  $REUSE_ARMS"
echo "vLLM:    $($VLLM_BIN --version 2>/dev/null | tail -1)"
echo "Results: $RESULTS_DIR"
echo "Date:    $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

# ── Dataset split per sampling group ─────────────────────────────────────────
SPLIT_DIR="$RESULTS_DIR/dataset"
SPLIT_ARGS=()
[ "$BY_CATEGORY" = "1" ] && SPLIT_ARGS+=(--by-category)
spec-data gw-bench split --bench "$BENCH_PATH" --out "$SPLIT_DIR" --seed "$SEED" "${SPLIT_ARGS[@]}" || exit 1
cp "$(dirname "$BENCH_PATH")/manifest.json" "$RESULTS_DIR/bench_manifest.json" 2>/dev/null
# slug|rows|path|flags per line
mapfile -t GROUP_LINES < <(python -c "
import json, sys
for g in json.load(open('$SPLIT_DIR/groups.json')):
    print('|'.join([g['slug'], str(g['rows']), g['path'], ' '.join(g['flags'])]))
")

# ── Reused arms from an earlier run (same node, vLLM and split) ──────────────
for SPEC in $REUSE_ARMS; do
    SRC="${SPEC%:*}"; NAME="${SPEC##*:}"
    SRC_SPLIT="$(dirname "$SRC")/dataset/groups.json"
    # Paths differ per run; the split must match on slug, rows and flags.
    if ! python -c "
import json, sys
key = lambda p: [(g['slug'], g['rows'], g['flags']) for g in json.load(open(p))]
sys.exit(key('$SRC_SPLIT') != key('$SPLIT_DIR/groups.json'))
"; then
        echo "✗ REUSE_ARMS: $SRC_SPLIT differs from this run's split (or is missing)"
        exit 1
    fi
    ln -sfn "$SRC" "$RESULTS_DIR/$NAME"
    echo "[bench] reused $SRC → $NAME"
done

SERVER_PID=""
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
        kill -9 "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
trap stop_server EXIT

start_server() {
    local arm=$1 log=$2
    local method="${arm%%:*}" k="$NUM_SPEC_TOKENS"
    [[ "$arm" == *:* ]] && k="${arm##*:}"
    local args=(
        serve "$MODEL"
        --host 127.0.0.1 --port "$PORT"
        --tensor-parallel-size 1
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_CONC"
        --long-prefill-token-threshold 8192
        --enable-chunked-prefill
        --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
        --trust-remote-code
    )
    [ "$PREFIX_CACHE" = "1" ] || args+=(--no-enable-prefix-caching)
    case "$method" in
        base) ;;
        mtp) args+=(--speculative-config "{\"method\":\"mtp\",\"model\":\"$MTP_MODEL\",\"num_speculative_tokens\":$k}") ;;
        dflash) args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$k}") ;;
        *) echo "✗ unknown arm: $arm"; return 1 ;;
    esac
    echo "[bench] $VLLM_BIN ${args[*]}"
    "$VLLM_BIN" "${args[@]}" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 720); do   # ≤ 60 min (first run downloads weights)
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
            echo "[bench] server ready ($arm)"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "✗ server exited ($arm); last log lines:"
            tail -30 "$log"
            return 1
        fi
        sleep 5
    done
    echo "✗ server not ready after 60 min ($arm)"
    return 1
}

run_bench() {  # dataset num_prompts concurrency result_dir result_file [sampling flags...]
    local dataset=$1 n=$2 conc=$3 dir=$4 file=$5
    shift 5
    local extra=()
    [ "$IGNORE_EOS" = "1" ] && extra+=(--ignore-eos)
    "$VLLM_BIN" bench serve \
        --backend vllm --endpoint /v1/completions \
        --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
        --dataset-name custom --dataset-path "$dataset" \
        --skip-chat-template --custom-output-len -1 \
        --num-prompts "$n" --no-oversample --disable-shuffle --seed "$SEED" \
        --max-concurrency "$conc" --request-rate inf --num-warmups 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --save-detailed --result-dir "$dir" --result-filename "$file" \
        "${extra[@]}" "$@"
}

FAILED=0
for ARM in $ARMS; do
    echo "=== Arm: $ARM ==="
    ARM_DIR="$RESULTS_DIR/${ARM/:/}"
    mkdir -p "$ARM_DIR"
    if ! start_server "$ARM" "$ARM_DIR/server.log"; then
        FAILED=1
        stop_server
        continue
    fi

    # Warmup (compile/graph capture paths, not measured) on the largest group.
    IFS='|' read -r _ _ WARM_PATH _ <<< "${GROUP_LINES[0]}"
    run_bench "$WARM_PATH" "$NUM_WARMUPS" "$NUM_WARMUPS" "$ARM_DIR" warmup.json \
        > "$ARM_DIR/warmup.log" 2>&1 || echo "⚠ warmup failed (see $ARM_DIR/warmup.log)"

    for LINE in "${GROUP_LINES[@]}"; do
        IFS='|' read -r SLUG ROWS DATASET FLAGS <<< "$LINE"
        if [ -n "$BENCH_GROUPS" ] && ! [[ " $BENCH_GROUPS " == *" $SLUG "* ]]; then
            continue
        fi
        for CONC in $CONCURRENCIES; do
            N=$(( CONC * PER_CONC ))
            [ "$N" -lt "$MIN_PROMPTS" ] && N=$MIN_PROMPTS
            [ "$N" -gt "$ROWS" ] && N=$ROWS
            CELL="$ARM_DIR/$SLUG/c$CONC"
            mkdir -p "$CELL"
            echo "[bench] $ARM | $SLUG ($FLAGS) | conc=$CONC | n=$N  $(date +%T)"
            spec-data gw-bench scrape --port "$PORT" --out "$CELL/spec_before.json"
            # shellcheck disable=SC2086  # FLAGS is a flag list
            if run_bench "$DATASET" "$N" "$CONC" "$CELL" bench.json $FLAGS > "$CELL/bench.log" 2>&1; then
                spec-data gw-bench scrape --port "$PORT" --out "$CELL/spec_after.json"
                grep -E "Successful requests|Output token throughput|Median TPOT|Median TTFT" "$CELL/bench.log" | sed 's/^/    /'
            else
                echo "✗ bench failed: $CELL/bench.log"
                tail -15 "$CELL/bench.log"
                FAILED=1
            fi
        done
    done
    stop_server
done

echo "---"
spec-data gw-bench summarize --results-dir "$RESULTS_DIR"
echo "---"
if [ $FAILED -eq 0 ]; then
    echo "✓ Bench PASSED — $RESULTS_DIR/summary.md"
else
    echo "✗ Bench had failures — see logs under $RESULTS_DIR"
fi
echo "=== Complete: $(date) ==="
exit $FAILED
