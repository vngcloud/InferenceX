#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: Gemma-4-31B DFlash smoke — gates 4a (static) + 4b (capture).
#
# Captures DFlash target features (layers 1,15,29,43,57 of 60) for the smoke
# splits built by scripts/build_gemma4_dflash_smoke.py, then checks them.
# DFlash and DFlash2 share these features. Plan: __plan__/[30-09-26] gemma4-dflash-smoke.md
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_gemma4_dflash_capture.sh
#   ./scripts/slurm.sh submit scripts/job_gemma4_dflash_capture.sh "SPLITS=overfit"
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=g4-dflash-capture
# 8xH200 node; one GPU is enough (FP8 target fits with room for KV).
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
# Features, venv and model copy are on this node's local disk.
#SBATCH --nodelist=hgx-h200-01
#SBATCH --time=02:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-dflash-capture_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-dflash-capture_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_MODEL="${TARGET_MODEL:-RedHatAI/gemma-4-31B-it-FP8-block}"
DRAFT_CONFIG="${DRAFT_CONFIG:-configs/specforge/gemma-4-31b-dflash.json}"
SMOKE_DIR="${SMOKE_DIR:-data/gemma4-dflash-smoke}"
SPLITS="${SPLITS:-overfit heldout long}"
# Feature directory (under SMOKE_DIR) and its node-local target. Override both to capture into a
# separate directory (e.g. clean features) without touching the existing ones.
FEAT_NAME="${FEAT_NAME:-hidden_states}"
MEM_FRACTION="${MEM_FRACTION:-0.7}"
# SGLang Gemma 4 accepts only trtllm_mha (Blackwell) or triton on H200.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"

# ── Environment ──────────────────────────────────────────────────────────────
# Node-local venv, model, features and caches: see scripts/hgx_local_env.sh.
source /shared/phucnlt2/speculative-training-pipeline/scripts/hgx_local_env.sh

echo "=== Gemma-4 DFlash capture job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Target: $TARGET_MODEL   Draft config: $DRAFT_CONFIG   Splits: $SPLITS"
echo "Date:   $(date)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
# The 8-GPU node lists every GPU in nvidia-smi; only the allocated one may be used.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "✗ no GPU allocation visible"; exit 1; }
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline
FEAT_LOCAL="${FEAT_LOCAL:-$LOCAL_ROOT/gemma4-dflash-smoke/$FEAT_NAME}"
link_local "$SMOKE_DIR/$FEAT_NAME" "$FEAT_LOCAL"
echo "Features: $SMOKE_DIR/$FEAT_NAME -> $(readlink "$SMOKE_DIR/$FEAT_NAME")"
# prepare_hidden_states.py passes its own --cache-dir (./cache) to HF, which hides
# HF_HUB_CACHE; give it the local snapshot directory instead of the repo id.
TARGET_PATH="$TARGET_MODEL"
if [ ! -d "$TARGET_PATH" ]; then
    TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--${TARGET_MODEL/\//--}/snapshots/* | head -1)"
fi
[ -f "$TARGET_PATH/config.json" ] || { echo "✗ no local snapshot for $TARGET_MODEL"; exit 1; }
echo "Model:    $TARGET_PATH"

echo ">>> Gate 4a: static checks (configs, draft shapes, template loss masks)"
python scripts/check_gemma4_dflash.py static || exit 1

for split in $SPLITS; do
    # KV pool = batch x max_length tokens. Gemma 4's sliding-window pool defaults to
    # 0.8 of that, so --sglang-swa-full-tokens-ratio 1.0 below is required.
    case "$split" in
        long) MAX_LENGTH=10240; BATCH=2 ;;
        *)    MAX_LENGTH=2048;  BATCH=8 ;;
    esac
    OUT="$SMOKE_DIR/$FEAT_NAME/$split"
    echo ">>> Gate 4b: capture $split (max_length $MAX_LENGTH) -> $OUT"
    if [ -f "$OUT/.captured" ]; then
        echo "    already captured; rm $OUT/.captured to redo"
    else
    rm -rf "$OUT"
    torchrun --nproc_per_node=1 \
        third_party/SpecForge/scripts/prepare_hidden_states.py \
        --target-model-path "$TARGET_PATH" \
        --data-path "$SMOKE_DIR/$split.jsonl" \
        --output-path "$OUT" \
        --draft-model-config "$DRAFT_CONFIG" \
        --strategy dflash \
        --chat-template gemma4 \
        --is-preformatted \
        --max-length "$MAX_LENGTH" \
        --tp-size 1 \
        --batch-size "$BATCH" \
        --sglang-context-length $((MAX_LENGTH + 16)) \
        --sglang-attention-backend "$ATTENTION_BACKEND" \
        --sglang-swa-full-tokens-ratio 1.0 \
        --sglang-mem-fraction-static "$MEM_FRACTION" || { echo "✗ capture $split FAILED"; exit 1; }
    touch "$OUT/.captured"
    fi
    du -sh "$OUT"
    python scripts/check_gemma4_dflash.py features "$OUT" || exit 1
done

echo "✓ Capture gates passed: $(date)"
