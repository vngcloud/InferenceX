#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: serve the target + a trained Gemma-4 DFlash/DFlash2 draft in vLLM or SGLang,
# log the draft's block-embedding norms (test A), load the export (test E) and measure
# greedy acceptance on overfit and held-out prompts (test B). One engine/config per job.
# Plan: __plan__/[01-10-26] gemma4-dflash-engine-verification.md
#
# Usage (KEY=VALUE after the script):
#   ./scripts/slurm.sh submit scripts/job_gemma4_engine_verify.sh ENGINE=vllm TAG=vllm-dflash
#   ... ENGINE=sglang TAG=sglang-dflash-scale1
#   ... ENGINE=vllm TAG=vllm-dflash-x73 DFV_SCALE=73.32            (test C, vLLM: scale up)
#   ... ENGINE=sglang TAG=sglang-dflash-div73 DFV_SGLANG_NOISE_SCALE=0.013639   (test C)
#   ... RUN=gemma4-dflash2-smoke-overfit CONFIG_PATCH='{"dflash_config":{"final_logit_softcapping":30}}'
# Results: /shared/phucnlt2/verify-results/<TAG>/{embed_norms.jsonl,overfit.json,heldout.json,server.log}
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=g4-engine-verify
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --time=02:30:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-engine-verify_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-engine-verify_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"                                   # vllm | sglang
RUN="${RUN:-gemma4-dflash-smoke-overfit}"
TAG="${TAG:-$ENGINE-$RUN}"
TARGET_MODEL="${TARGET_MODEL:-RedHatAI/gemma-4-31B-it-FP8-block}"
NUM_ROWS="${NUM_ROWS:-32}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONFIG_PATCH="${CONFIG_PATCH:-}"                           # JSON deep-merged into the draft config.json
DFV_SCALE="${DFV_SCALE:-1}"                                # vLLM block-embedding factor (test C)
DFV_SGLANG_NOISE_SCALE="${DFV_SGLANG_NOISE_SCALE:-}"       # sglang noise-embedding factor (test C)
EAGER="${EAGER:-1}"                                        # 1 = no CUDA graphs, so the norm hook runs on every call
PORT="${PORT:-$((20000 + SLURM_JOB_ID % 20000))}"   # jobs share the node: never a fixed port
RESULTS="/shared/phucnlt2/verify-results/$TAG"

TEST_NODE="${TEST_NODE:-0}"   # 1 = test-partition node: no hgx local env, vLLM only (see job_gemma4_engine_verify_test.sh)
if [ "$TEST_NODE" = "1" ]; then
    [ "$ENGINE" = "vllm" ] || { echo "✗ test-partition nodes only fit the vLLM venv"; exit 1; }
    LOCAL_ROOT="${LOCAL_ROOT:-/tmp/phucnlt2}"
    export HOME=/shared/phucnlt2 XDG_CACHE_HOME="$LOCAL_ROOT/xdg-cache"
    export HF_HOME=/shared/phucnlt2/hf-cache HF_HUB_CACHE=/shared/phucnlt2/hf-cache/hub HF_HUB_OFFLINE=1
    export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/torchinductor-cache" TRITON_CACHE_DIR="$LOCAL_ROOT/triton-cache"
    export FLASHINFER_WORKSPACE_BASE="$LOCAL_ROOT"
    mkdir -p "$LOCAL_ROOT" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" /shared/phucnlt2/job-logs
else
    source /shared/phucnlt2/speculative-training-pipeline/scripts/hgx_local_env.sh   # sglang venv + caches
fi
mkdir -p "$RESULTS"
echo "=== engine verify $TAG: $ENGINE draft=$RUN on $SLURMD_NODENAME, job $SLURM_JOB_ID ==="
date; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "✗ no GPU allocation visible"; exit 1; }

# Draft export on node-local disk (NFS is slow), optionally with a patched config.json.
DRAFT_DIR="$LOCAL_ROOT/drafts/$TAG"
rm -rf "$DRAFT_DIR"; mkdir -p "$DRAFT_DIR"
cp /shared/phucnlt2/checkpoints/$RUN/hf-draft/* "$DRAFT_DIR"/ || exit 1
python3 - "$DRAFT_DIR/config.json" "$CONFIG_PATCH" <<'PY'
import json, sys
path, patch = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))
def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
if patch:
    merge(cfg, json.loads(patch))
    json.dump(cfg, open(path, "w"), indent=2)
print("draft config:", cfg["architectures"], cfg.get("dflash_config"))
PY

TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--${TARGET_MODEL/\//--}/snapshots/* | head -1)"
[ -f "$TARGET_PATH/config.json" ] || { echo "✗ no local snapshot for $TARGET_MODEL"; exit 1; }

export PYTHONPATH="/shared/phucnlt2/speculative-training-pipeline/scripts/verify_engine:${PYTHONPATH:-}"
export DFV_LOG="$RESULTS/embed_norms.jsonl"; rm -f "$DFV_LOG"
export DFV_SCALE
[ "${DUMP:-0}" = "1" ] && export DFV_DUMP="$RESULTS/aux"
[ "${SGL_CLONE:-0}" = "1" ] && export DFV_SG_CLONE=1   # test fix for the in-place aux aliasing
[ "${SGL_LAYERS:-0}" = "1" ] && export DFV_SG_LAYERS="$RESULTS/layers"   # sglang per-layer outputs (bisect vs HF)
[ "${AUDIT:-0}" = "1" ] && { export DFV_AUDIT="$RESULTS/audit.jsonl" DFV_DRAFT_DIR="$DRAFT_DIR"; rm -f "$DFV_AUDIT"; }
[ -n "$DFV_SGLANG_NOISE_SCALE" ] && export DFV_SGLANG_NOISE_SCALE

EAGER_FLAG_VLLM=""; EAGER_FLAG_SGL=""
[ "$EAGER" = "1" ] && { EAGER_FLAG_VLLM="--enforce-eager"; EAGER_FLAG_SGL="--disable-cuda-graph"; }
if [ "$ENGINE" = "vllm" ]; then
    unset CUDA_HOME
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
    # venv-vllm's python is /usr/bin/python3 (3.12 here) and its 8.4 GB NFS tree is too slow to copy
    # or import cold (jobs 1100, 1153). Install the same pin node-local with uv instead (~20 s).
    PY310="$LOCAL_ROOT/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"
    UV="$LOCAL_ROOT/uv"
    if [ "$TEST_NODE" = "1" ]; then   # fresh node: uv from astral (~30 MB) + uv-managed python 3.10
        # the system python3.10 has no Python.h, which triton needs to build cuda_utils (job 1194)
        PY310=3.10
        export UV_PYTHON_INSTALL_DIR="$LOCAL_ROOT/uv-python" UV_PYTHON_PREFERENCE=only-managed
        [ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$LOCAL_ROOT" sh
    fi
    VLLM_VENV="$LOCAL_ROOT/venv-vllm-fresh"
    [ "$TEST_NODE" = "1" ] && VLLM_VENV="$LOCAL_ROOT/venv-vllm-uvpy2"
    (
        flock 9
        if [ ! -f "$VLLM_VENV/.complete" ]; then
            echo ">>> installing vllm==0.30.0 into $VLLM_VENV"
            rm -rf "$VLLM_VENV"
            UV_CACHE_DIR="$LOCAL_ROOT/uv-cache" "$UV" venv --python "$PY310" "$VLLM_VENV" \
                && UV_CACHE_DIR="$LOCAL_ROOT/uv-cache" "$UV" pip install --python "$VLLM_VENV/bin/python" vllm==0.30.0 \
                && touch "$VLLM_VENV/.complete"
        fi
    ) 9>"$LOCAL_ROOT/venv-vllm-fresh.lock"
    [ -f "$VLLM_VENV/.complete" ] || { echo "✗ vllm install failed"; exit 1; }
    PY310="$VLLM_VENV/bin/python"; CLIENT_PY="$PY310"
    "$PY310" -m vllm.entrypoints.cli.main serve "$TARGET_PATH" --served-model-name target \
        --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --max-model-len 8192 \
        ${EAGER_FLAG_VLLM} --gpu-memory-utilization 0.85 --max-num-seqs 4 --no-enable-prefix-caching --trust-remote-code ${VLLM_EXTRA:-} \
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":15}" \
        > "$RESULTS/server.log" 2>&1 &
else
    python -m sglang.launch_server --model-path "$TARGET_PATH" --host 127.0.0.1 --port "$PORT" \
        --tp-size 1 --context-length 8192 --mem-fraction-static 0.7 --attention-backend triton \
        ${EAGER_FLAG_SGL} --swa-full-tokens-ratio 1.0 --max-running-requests 4 --disable-radix-cache \
        --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_DIR" \
        --speculative-num-draft-tokens 16 --trust-remote-code \
        > "$RESULTS/server.log" 2>&1 &
fi
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; sleep 5; kill -9 $SERVER_PID 2>/dev/null' EXIT

READY=0
for _ in $(seq "${READY_ITERS:-720}"); do   # 5 s each; default ≤ 60 min
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then READY=1; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
    sleep 5
done
if [ $READY != 1 ]; then
    echo "✗ test E: server did not come up; last log lines:"; tail -60 "$RESULTS/server.log"; exit 1
fi
echo "✓ test E: server up with the draft ($(date))"
grep -i -E "missing|unexpected|not found in|mismatch|dflash" "$RESULTS/server.log" | head -20

for split in ${SPLITS:-overfit heldout}; do
    [ "$split" = none ] && continue   # SPLITS=none: bring the server up (audit hooks) and skip the client
    if [ "${FORCED:-0}" = "1" ]; then   # forced-prefix run: drift-free acceptance (forced_client.py)
        ${CLIENT_PY:-python} scripts/verify_engine/forced_client.py --engine "$ENGINE" --base-url "http://127.0.0.1:$PORT" \
            --model target --tokenizer "$TARGET_PATH" --data data/gemma4-dflash-smoke/$split.jsonl --n "$NUM_ROWS" \
            --out "$RESULTS/forced_$split.json" || echo "✗ forced client failed ($split)"
        continue
    fi
    ${CLIENT_PY:-python} scripts/verify_engine/accept_client.py --engine "$ENGINE" --base-url "http://127.0.0.1:$PORT" \
        --model target --tokenizer "$TARGET_PATH" --data data/gemma4-dflash-smoke/$split.jsonl --n "$NUM_ROWS" \
        --max-tokens "$MAX_TOKENS" --out "$RESULTS/$split.json" || echo "✗ client failed ($split)"
done
echo "--- embed norms (test A)"; cat "$DFV_LOG" 2>/dev/null | head -8
echo "✓ done $(date)"
