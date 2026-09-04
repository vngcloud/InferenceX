#!/usr/bin/env bash
set -euo pipefail

# Quality-eval script for SWE-bench Pro, adapted for InferenceX CI.
# Env vars are set by runners/launch_quality-eval.sh, not sourced from .env.
# Requires Docker on the runner for Phase 2 evaluation containers.
#
# Required env: QUALITY_ENDPOINT, QUALITY_API_KEY, QUALITY_MODEL_NAME
# Optional env: RUN_ID, LIMIT, WORKERS, EVAL_WORKERS, REDO_EXISTING, REDO_EVAL,
#               DOCKERHUB_USERNAME, COST_LIMIT, STEP_LIMIT

WORKSPACE_DIR="${QUALITY_WORKSPACE:-$(pwd)}"

export OPENAI_API_KEY="$QUALITY_API_KEY"
export OPENAI_API_BASE="$QUALITY_ENDPOINT"
export MSWEA_MODEL_API_KEY="$QUALITY_API_KEY"
export MSWEA_SILENT_STARTUP=1

PYTHON="${QUALITY_SWEBENCHPRO_VENV:-$WORKSPACE_DIR/.venv-swebenchpro}/bin/python"

RAW_MODEL="${QUALITY_MODEL_NAME#openai/}"

RUN_ID="${RUN_ID:-$(echo "$RAW_MODEL" | tr -c '[:alnum:]._-' '_')}"

SWEBENCH_DIR="${QUALITY_SWEBENCH_DIR:-$WORKSPACE_DIR/SWE-bench_Pro-os}"
INSTANCES_YAML="$SWEBENCH_DIR/SWE-agent/data/instances.yaml"
RAW_SAMPLE="${QUALITY_SWEBENCH_RAW_SAMPLE:-$SWEBENCH_DIR/helper_code/sweap_eval_full_v2.jsonl}"
CONFIG="${QUALITY_SWEBENCH_CONFIG:-$WORKSPACE_DIR/benchmarks/single_node/quality/tasks/swebench-pro/swebench_pro.yaml}"
RUN_SWEBENCH_PRO="${QUALITY_SWEBENCH_RUN_SCRIPT:-$WORKSPACE_DIR/benchmarks/single_node/quality/tasks/swebench-pro/run_swebench_pro.py}"

OUT_DIR="$WORKSPACE_DIR/jobs/$RUN_ID/swebench_pro"
PRED_DIR="$OUT_DIR/preds"
PATCHES_JSON="$OUT_DIR/patches.json"
EVAL_DIR="$OUT_DIR/eval"

mkdir -p "$PRED_DIR" "$EVAL_DIR"

LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
REDO_EXISTING="${REDO_EXISTING:-0}"
REDO_EVAL="${REDO_EVAL:-0}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-jefzda}"
COST_LIMIT="${COST_LIMIT:-3.0}"
STEP_LIMIT="${STEP_LIMIT:-250}"

SLICE_ARG=()
if [[ -n "$LIMIT" ]]; then
  SLICE_ARG=(--slice "0:${LIMIT}")
fi

REDO_ARG=()
if [[ "$REDO_EXISTING" == "1" ]]; then
  REDO_ARG=(--redo-existing)
fi

REDO_EVAL_ARG=()
if [[ "$REDO_EVAL" == "1" ]]; then
  REDO_EVAL_ARG=(--redo)
fi

echo "=== SWE-bench Pro run ==="
echo "  RUN_ID            : $RUN_ID"
echo "  Model             : $RAW_MODEL"
echo "  Phase 1 (agent)   : mini-swe-agent (litellm -> $OPENAI_API_BASE)"
echo "  Phase 2 (eval)    : swe_bench_pro_eval.py --use_local_docker"
echo "  Instances yaml    : $INSTANCES_YAML"
echo "  Raw sample        : $RAW_SAMPLE"
echo "  Output dir        : $OUT_DIR"
echo "  Workers (agent)   : $WORKERS"
echo "  Workers (eval)    : $EVAL_WORKERS"
echo "  Cost/step limit   : \$${COST_LIMIT} / ${STEP_LIMIT} steps per instance"
if [[ -n "$LIMIT" ]]; then
  echo "  Limit             : first $LIMIT instances"
fi
echo

RUN_CONFIG="$OUT_DIR/run_config.yaml"
sed -e "s/step_limit: [0-9]*/step_limit: ${STEP_LIMIT}/" \
    -e "s/cost_limit: [0-9.]*$/cost_limit: ${COST_LIMIT}/" \
    "$CONFIG" > "$RUN_CONFIG"

echo "--- Phase 1: agent patch generation ---"
"$PYTHON" "$RUN_SWEBENCH_PRO" \
  --instances-path "$INSTANCES_YAML" \
  --output "$PRED_DIR" \
  --config "$RUN_CONFIG" \
  --model "openai/${RAW_MODEL}" \
  --workers "$WORKERS" \
  "${SLICE_ARG[@]}" \
  "${REDO_ARG[@]}"

echo "--- Gathering patches ---"
"$PYTHON" - "$PRED_DIR/preds.json" "$PATCHES_JSON" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
patches = [
    {"instance_id": v["instance_id"], "patch": v.get("model_patch") or "", "prefix": "agent"}
    for v in data.values()
]
json.dump(patches, open(dst, "w"), indent=2)
print(f"Wrote {len(patches)} patches to {dst}")
PY

echo "--- Phase 2: patch evaluation ---"
cd "$SWEBENCH_DIR"
"$PYTHON" swe_bench_pro_eval.py \
  --raw_sample_path "$RAW_SAMPLE" \
  --patch_path "$PATCHES_JSON" \
  --output_dir "$EVAL_DIR" \
  --scripts_dir run_scripts \
  --num_workers "$EVAL_WORKERS" \
  --dockerhub_username "$DOCKERHUB_USERNAME" \
  --use_local_docker \
  "${REDO_EVAL_ARG[@]}"

echo
echo "=== SWE-bench Pro complete ==="
echo "  Predictions : $PRED_DIR/preds.json"
echo "  Patches     : $PATCHES_JSON"
echo "  Eval results: $EVAL_DIR/eval_results.json"
echo
"$PYTHON" - <<PY
import json
r = json.load(open("$EVAL_DIR/eval_results.json"))
n = len(r)
p = sum(1 for v in r.values() if v)
print(f"Pass@1: {p}/{n} ({100*p/n:.1f}%)" if n else "No results")
PY
