#!/usr/bin/env bash
set -euo pipefail
set -x

# CPU runner for quality-eval benchmarks. No salloc, no Docker image, no
# squashfs — this box only drives benchmark scripts against an externally-
# managed inference endpoint (QUALITY_ENDPOINT).
#
# This script sets up the Python environments and clones the external
# benchmark repos (if not already present) before dispatching to the
# per-benchmark script under benchmarks/single_node/quality/.
#
# Venvs and cloned repos live under $QUALITY_CACHE_DIR (persistent across
# CI runs on a self-hosted runner).  Job output (results, logs) lives under
# $QUALITY_WORKSPACE (= $GITHUB_WORKSPACE) and is uploaded as artifacts.
#
# Required env (set by benchmark-tmpl.yml):
#   QUALITY_BENCHMARK_NAME  e.g. gpqa, mmlu_pro, hle, livecodebench, bfcl,
#                           scicode, swebench_pro, deepswe
#   QUALITY_ENDPOINT        e.g. https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1
#   QUALITY_API_KEY         API key for the endpoint
#   QUALITY_MODEL_NAME      e.g. openai/z-ai/glm-5.2
# Optional env:
#   RUN_ID, LIMIT (from EVAL_LIMIT), HF_TOKEN, SMOKE

# --- Paths ---------------------------------------------------------------
# Job output: per-run workspace (cleaned by GitHub Actions each run)
export QUALITY_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

# Persistent cache: venvs + cloned repos survive across runs.
# On a self-hosted runner $HOME is stable.  Fall back to /tmp for ephemeral CI.
export QUALITY_CACHE_DIR="${QUALITY_CACHE_DIR:-${HOME:-/tmp}/.quality-eval-cache}"
mkdir -p "$QUALITY_CACHE_DIR"

export RESULT_DIR="${QUALITY_WORKSPACE}/results"

BENCH_SCRIPT="benchmarks/single_node/quality/run_${QUALITY_BENCHMARK_NAME}.sh"

if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "ERROR: Unknown quality benchmark '${QUALITY_BENCHMARK_NAME}'" >&2
    echo "Expected script: $BENCH_SCRIPT" >&2
    exit 1
fi

# Map InferenceX's EVAL_LIMIT to the LIMIT env var that all benchmark
# scripts read for subset/smoke runs.
export LIMIT="${EVAL_LIMIT:-${LIMIT:-}}"

# Map NUM_CONCURRENT to per-benchmark concurrency env vars.
# Each benchmark script reads its own var; NUM_CONCURRENT is the unified knob.
if [[ -n "${NUM_CONCURRENT:-}" ]]; then
    export NUM_CONCURRENT="$NUM_CONCURRENT"       # gpqa, mmlu_pro, hle (lm-eval)
    export MULTIPROCESS="$NUM_CONCURRENT"           # livecodebench
    export NUM_THREADS="$NUM_CONCURRENT"            # bfcl
    export MAX_CONNECTIONS="$NUM_CONCURRENT"        # scicode
    export WORKERS="$NUM_CONCURRENT"                # swebench_pro
    export CCU="$NUM_CONCURRENT"                    # deepswe
fi

# Set RUN_ID from the experiment name if not already set.
export RUN_ID="${RUN_ID:-${EXP_NAME:-quality-eval}}"

# Ensure uv is available (GitHub Actions runners may not have it pre-installed).
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Per-benchmark environment setup
# ---------------------------------------------------------------------------
# Each function ensures the venv and external repo are ready.
# Venvs/repos live under $QUALITY_CACHE_DIR and are reused across runs.
# First run: creates everything from scratch (~5-15 min depending on benchmark).
# Subsequent runs: skips setup entirely (just exports paths).

setup_lmeval() {
    local VENV="$QUALITY_CACHE_DIR/.venv-lmeval"
    if [[ ! -x "$VENV/bin/lm-eval" ]] || ! "$VENV/bin/python" -c "import tenacity" 2>/dev/null; then
        echo "=== Setting up lm-eval venv ==="
        uv venv --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            "lm-eval[api]>=0.4.5" "openai>=1.59.0"
    fi
    export QUALITY_VENV="$VENV"
}

setup_livecodebench() {
    local LCB_DIR="$QUALITY_CACHE_DIR/LiveCodeBench"
    local VENV="$LCB_DIR/.venv-lcb"
    if [[ ! -d "$LCB_DIR" ]]; then
        echo "=== Cloning LiveCodeBench (first time) ==="
        git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "=== Setting up LiveCodeBench venv (first time) ==="
        uv venv --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$LCB_DIR"
    fi
    export QUALITY_LCB_VENV="$VENV"
    export QUALITY_LCB_DIR="$LCB_DIR"
}

setup_bfcl() {
    local BFCL_DIR="$QUALITY_CACHE_DIR/BFCL/berkeley-function-call-leaderboard"
    local VENV="$BFCL_DIR/.venv-bfcl"
    if [[ ! -d "$QUALITY_CACHE_DIR/BFCL" ]]; then
        echo "=== Cloning BFCL (first time) ==="
        git clone --depth 1 https://github.com/ShishirPatil/gorilla.git "$QUALITY_CACHE_DIR/BFCL"
    fi
    if [[ ! -x "$VENV/bin/bfcl" ]]; then
        echo "=== Setting up BFCL venv (first time) ==="
        uv venv --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$BFCL_DIR"
    fi
    export QUALITY_BFCL_VENV="$VENV"
    export QUALITY_BFCL_DIR="$BFCL_DIR"
}

setup_scicode() {
    local SCICODE_DIR="$QUALITY_CACHE_DIR/SciCode"
    local VENV="$QUALITY_CACHE_DIR/.venv-scicode"
    if [[ ! -d "$SCICODE_DIR" ]]; then
        echo "=== Cloning SciCode (first time) ==="
        git clone --depth 1 https://github.com/scicode-bench/SciCode.git "$SCICODE_DIR"
    fi
    if [[ ! -x "$VENV/bin/inspect" ]]; then
        echo "=== Setting up SciCode venv (first time) ==="
        uv venv --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$SCICODE_DIR"
    fi
    export QUALITY_SCICODE_VENV="$VENV"
    export QUALITY_SCICODE_DIR="$SCICODE_DIR"
}

setup_swebench_pro() {
    local SWEBENCH_DIR="$QUALITY_CACHE_DIR/SWE-bench_Pro-os"
    local VENV="$QUALITY_CACHE_DIR/.venv-swebenchpro"
    if [[ ! -d "$SWEBENCH_DIR" ]]; then
        echo "=== Cloning SWE-bench Pro (first time) ==="
        git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$SWEBENCH_DIR"
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "=== Setting up SWE-bench Pro venv (first time) ==="
        uv venv --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            -r "$SWEBENCH_DIR/requirements.txt" \
            "minisweagent" "litellm" "rich" "pyyaml"
    fi
    export QUALITY_SWEBENCHPRO_VENV="$VENV"
    export QUALITY_SWEBENCH_DIR="$SWEBENCH_DIR"
}

setup_deepswe() {
    local DEEPSWE_DIR="$QUALITY_CACHE_DIR/deep-swe"
    if [[ ! -d "$DEEPSWE_DIR" ]]; then
        echo "=== Cloning DeepSWE (first time) ==="
        git clone --depth 1 https://github.com/datacurve/deep-swe.git "$DEEPSWE_DIR"
    fi
    export QUALITY_DEEPSWE_DIR="$DEEPSWE_DIR"
}

# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
# After the benchmark script runs, copy result files from the per-benchmark
# output directory to the workspace root so that benchmark-tmpl.yml's
# upload-artifact step (which globs for results*.json, *.traj*, etc. at
# workspace root) and validate_scores.py can find them.
#
# Also creates meta_env.json with the model prefix for threshold validation.

collect_results() {
    local BENCH="$1"
    local OUT_BASE="$QUALITY_WORKSPACE/jobs/$RUN_ID/$BENCH"
    local DEST="$QUALITY_WORKSPACE"

    echo "=== Collecting results from $OUT_BASE ==="

    # Create meta_env.json for validate_scores.py
    local MODEL_PREFIX="${MODEL_PREFIX:-${EXP_NAME%%_*}}"
    cat > "$DEST/meta_env.json" <<EOF
{"infmax_model_prefix": "${MODEL_PREFIX}", "benchmark": "${BENCH}", "run_id": "${RUN_ID}"}
EOF

    # Copy result files to workspace root (flatten, don't preserve dir structure)
    # Patterns cover all 8 benchmarks' output formats.
    local COPIED=0

    # results*.json — lm-eval (GPQA, MMLU-Pro, HLE) + general
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'results*.json' -print0 2>/dev/null || true)

    # sample*.jsonl — lm-eval logged samples
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'sample*.jsonl' -print0 2>/dev/null || true)

    # eval_results*.json — SWE-bench Pro, SciCode
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'eval_results*.json' -print0 2>/dev/null || true)

    # predictions.jsonl, agent_preds.json — SWE-bench, agentic
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f \( -name 'predictions.jsonl' -o -name 'agent_preds.json' \) -print0 2>/dev/null || true)

    # swebench_report_*.json
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'swebench_report_*.json' -print0 2>/dev/null || true)

    # *.traj* — DeepSWE, SWE-bench trajectories
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.traj*' -print0 2>/dev/null || true)

    # BFCL score CSVs
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.csv' -print0 2>/dev/null || true)

    # LiveCodeBench result JSONs/JSONLs
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f \( -name '*.jsonl' -o -name 'lcb_results*.json' \) -print0 2>/dev/null || true)

    echo "  Copied $COPIED result file(s) to $DEST"
    if [[ "$COPIED" -eq 0 ]]; then
        echo "  WARNING: no result files found in $OUT_BASE" >&2
        # List what IS there for debugging
        find "$OUT_BASE" -type f 2>/dev/null | head -20 || echo "  (directory empty or missing)"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
echo "=== Quality-eval setup: ${QUALITY_BENCHMARK_NAME} ==="
echo "  Workspace (output) : $QUALITY_WORKSPACE"
echo "  Cache (venv/repos) : $QUALITY_CACHE_DIR"
echo "  Run ID             : $RUN_ID"
echo

case "${QUALITY_BENCHMARK_NAME}" in
    gpqa|mmlu_pro|hle)
        setup_lmeval
        ;;
    livecodebench)
        setup_livecodebench
        ;;
    bfcl)
        setup_bfcl
        ;;
    scicode)
        setup_scicode
        ;;
    swebench_pro)
        setup_swebench_pro
        ;;
    deepswe)
        setup_deepswe
        ;;
    *)
        echo "ERROR: Unknown quality benchmark '${QUALITY_BENCHMARK_NAME}'" >&2
        exit 1
        ;;
esac

echo "=== Dispatching to $BENCH_SCRIPT ==="
bash "$BENCH_SCRIPT"

echo "=== Collecting results for artifact upload ==="
collect_results "${QUALITY_BENCHMARK_NAME}"
