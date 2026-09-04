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
    if [[ ! -x "$VENV/bin/lm-eval" ]] || ! "$VENV/bin/python" -c "import tenacity, PIL" 2>/dev/null; then
        echo "=== Setting up lm-eval venv ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            "lm-eval[api]>=0.4.5" "openai>=1.59.0" "Pillow>=10.0.0"
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
    # Patch lcb_runner files that import HUMAN_PROMPT/AI_PROMPT without
    # try/except fallback (anthropic>=0.42 removed them).  code_generation.py
    # already has a fallback; self_repair.py and test_output_prediction.py do not.
    for f in lcb_runner/prompts/self_repair.py lcb_runner/prompts/test_output_prediction.py; do
        if [[ -f "$LCB_DIR/$f" ]] && ! grep -q "HUMAN_PROMPT = None" "$LCB_DIR/$f" 2>/dev/null; then
            echo "=== Patching $f for anthropic>=0.42 compatibility ==="
            python3 - "$LCB_DIR/$f" <<'PY'
import pathlib, sys, re
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = "from anthropic import HUMAN_PROMPT, AI_PROMPT"
new = """try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT = None
    AI_PROMPT = None"""
if old in src and "HUMAN_PROMPT = None" not in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
        fi
    done
    # Patch code_generation.py: load_dataset needs config name = release_version,
    # otherwise datasets looks for 'default' config which doesn't exist in cache.
    local CG_FILE="$LCB_DIR/lcb_runner/benchmarks/code_generation.py"
    if ! grep -q 'release_version, split=' "$CG_FILE" 2>/dev/null; then
        echo "=== Patching code_generation.py load_dataset config name ==="
        python3 - "$CG_FILE" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = 'load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)'
new = 'load_dataset("livecodebench/code_generation_lite", release_version, split="test", version_tag=release_version, trust_remote_code=True)'
if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
PY
    fi
    # Patch lm_styles.py to add z-ai/glm-5.2 as an OpenAIChat model
    # (LCB has a hardcoded LanguageModelStore dict; our model isn't in it)
    local LM_STYLES="$LCB_DIR/lcb_runner/lm_styles.py"
    if ! grep -q '"z-ai/glm-5.2"' "$LM_STYLES" 2>/dev/null; then
        echo "=== Patching lm_styles.py with z-ai/glm-5.2 model ==="
        python3 - "$LM_STYLES" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
entry = '''    LanguageModel(
        "z-ai/glm-5.2",
        "GLM-5.2",
        LMStyle.OpenAIChat,
        datetime(2024, 12, 1),
        "https://huggingface.co/z-ai",
    ),
'''
marker = "\n]\n\nLanguageModelStore"
idx = src.find(marker)
if idx == -1:
    raise SystemExit("marker not found")
src = src[:idx] + "\n" + entry + src[idx:]
p.write_text(src)
PY
    fi
    # Cache-bust: check livecodebench import works
    if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import lcb_runner" 2>/dev/null; then
        echo "=== Setting up LiveCodeBench venv (first time) ==="
        uv venv --clear --seed "$VENV"
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
    # Inject z-ai/glm-5.2 model config if not already present
    local MC_FILE="$BFCL_DIR/bfcl_eval/constants/model_config.py"
    if ! grep -q "z-ai/glm-5.2-FC" "$MC_FILE" 2>/dev/null; then
        echo "=== Patching BFCL model_config.py with z-ai/glm-5.2 ==="
        python3 - "$MC_FILE" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
block = '''    "z-ai/glm-5.2-FC": ModelConfig(
        model_name="z-ai/glm-5.2",
        display_name="GLM-5.2 (FC, OpenAI-compatible)",
        url="https://tokenplan.api.greennode.ai",
        org="z-ai",
        license="Proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "z-ai/glm-5.2-PROMPT": ModelConfig(
        model_name="z-ai/glm-5.2",
        display_name="GLM-5.2 (Prompt, OpenAI-compatible)",
        url="https://tokenplan.api.greennode.ai",
        org="z-ai",
        license="Proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''
marker = 'api_inference_model_map = {'
idx = src.find(marker)
if idx == -1:
    print("ERROR: could not find api_inference_model_map marker", file=sys.stderr)
    sys.exit(1)
insert_at = src.find('{', idx) + 1
p.write_text(src[:insert_at] + '\n' + block + src[insert_at:])
PY
    fi
    if [[ ! -x "$VENV/bin/bfcl" ]] || ! "$VENV/bin/python" -c "import soundfile" 2>/dev/null; then
        echo "=== Setting up BFCL venv (first time) ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" -e "$BFCL_DIR" "soundfile>=0.12.0"
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
    # Cache-bust: check scicode + inspect_ai import works
    if [[ ! -x "$VENV/bin/inspect" ]] || ! "$VENV/bin/python" -c "import scicode; import inspect_ai" 2>/dev/null; then
        echo "=== Setting up SciCode venv (first time) ==="
        uv venv --clear --seed "$VENV"
        # SciCode pyproject pins unpinned "datasets" → resolver picks 2.14.4,
        # but inspect-ai requires datasets>=2.16.  datasets 2.16.1 has a bug
        # with SciCode1/SciCode dataset (TypeError in generate_from_dict).
        # Pin datasets==5.0.1 + pyarrow==25.0.1 (known good, same as LCB/swebench).
        uv pip install --python "$VENV/bin/python" \
            "datasets==5.0.1" "pyarrow==25.0.1" "openai>=3.1" "anthropic" "config" \
            "litellm" "inspect-ai" "rich" "pytest" "pytest-cov" \
            "matplotlib" "scipy" "sympy" "h5py" "jsonlines" \
            "google-generativeai"
        uv pip install --python "$VENV/bin/python" --no-deps -e "$SCICODE_DIR"
    fi
    export QUALITY_SCICODE_VENV="$VENV"
    export QUALITY_SCICODE_DIR="$SCICODE_DIR"
}

setup_swebench_pro() {
    local SWEBENCH_DIR="$QUALITY_CACHE_DIR/SWE-bench_Pro-os"
    local VENV="$QUALITY_CACHE_DIR/.venv-swebenchpro"
    if [[ ! -d "$SWEBENCH_DIR" ]]; then
        echo "=== Cloning SWE-bench Pro (first time, with submodules) ==="
        git clone --recurse-submodules --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$SWEBENCH_DIR"
    fi
    # Ensure SWE-agent submodule is present (cache may have shallow clone without it)
    if [[ ! -d "$SWEBENCH_DIR/SWE-agent/.git" ]]; then
        echo "=== Initializing SWE-agent submodule ==="
        git -C "$SWEBENCH_DIR" submodule update --init --recursive
    fi
    # Generate instances.yaml if missing (required by run_swebench_pro.py)
    local INSTANCES_YAML="$SWEBENCH_DIR/SWE-agent/data/instances.yaml"
    if [[ ! -f "$INSTANCES_YAML" ]]; then
        echo "=== Generating instances.yaml from HuggingFace dataset ==="
        if [[ ! -x "$VENV/bin/python" ]]; then
            uv venv --clear --seed "$VENV"
            uv pip install --python "$VENV/bin/python" \
                -r "$SWEBENCH_DIR/requirements.txt" \
                "mini-swe-agent" "litellm" "rich" "pyyaml" "datasets" "tqdm"
        fi
        "$VENV/bin/python" "$SWEBENCH_DIR/helper_code/generate_sweagent_instances.py" \
            --dockerhub_username "${DOCKERHUB_USERNAME:-jefzda}" \
            --output_path "$INSTANCES_YAML"
    fi
    if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import yaml" 2>/dev/null; then
        echo "=== Setting up SWE-bench Pro venv (first time) ==="
        uv venv --clear --seed "$VENV"
        uv pip install --python "$VENV/bin/python" \
            -r "$SWEBENCH_DIR/requirements.txt" \
            "mini-swe-agent" "litellm" "rich" "pyyaml"
    fi
    export QUALITY_SWEBENCHPRO_VENV="$VENV"
    export QUALITY_SWEBENCH_DIR="$SWEBENCH_DIR"
}

setup_deepswe() {
    local DEEPSWE_DIR="$QUALITY_CACHE_DIR/deep-swe"
    if [[ ! -d "$DEEPSWE_DIR" ]]; then
        echo "=== Cloning DeepSWE (first time) ==="
        git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git "$DEEPSWE_DIR"
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

    # result.json (singular) — DeepSWE / pier output; copy as results.json
    # so benchmark-tmpl.yml's `ls results*.json` check passes.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/results.json"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -maxdepth 2 -type f -name 'result.json' ! -path '*/ipython-session-bundle-*' -print0 2>/dev/null || true)

    # sample*.jsonl — lm-eval logged samples
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name 'sample*.jsonl' -print0 2>/dev/null || true)

    # eval_results*.json — SWE-bench Pro, SciCode
    # Also copy as results.json so benchmark-tmpl.yml's `ls results*.json` check passes.
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        if [[ ! -f "$DEST/results.json" ]]; then
            cp -f "$f" "$DEST/results.json"
        fi
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
    # LCB writes to output/<model>/<scenario>_<n>_<temp>.json and _eval.json
    # Copy first .json as results.json so benchmark-tmpl.yml's glob matches.
    local LCB_FIRST_JSON=""
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f \( -name '*.jsonl' -o -name 'lcb_results*.json' \) -print0 2>/dev/null || true)
    # Also pick up LCB's output/*.json files
    while IFS= read -r -d '' f; do
        cp -f "$f" "$DEST/"
        if [[ -z "$LCB_FIRST_JSON" ]]; then
            LCB_FIRST_JSON="$f"
        fi
        COPIED=$((COPIED + 1))
    done < <(find "$OUT_BASE" -type f -name '*.json' ! -name 'results*.json' ! -name 'eval_results*.json' -print0 2>/dev/null || true)
    if [[ -n "$LCB_FIRST_JSON" && ! -f "$DEST/results.json" ]]; then
        cp -f "$LCB_FIRST_JSON" "$DEST/results.json"
    fi

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
