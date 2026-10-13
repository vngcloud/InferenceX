#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: Gemma-4-31B DFlash smoke — gates 4c (overfit) / 4d (long context).
#
# Trains one smoke config on features from job_gemma4_dflash_capture.sh,
# samples GPU memory every 5 s, and for overfit configs applies the curve gate.
# Plan: __plan__/[30-09-26] gemma4-dflash-smoke.md
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_gemma4_dflash_train.sh
#       [SAVE_ROOT=/shared/phucnlt2/checkpoints keeps the final checkpoint + HF export]
#   ./scripts/slurm.sh submit scripts/job_gemma4_dflash_train.sh \
#       SPECFORGE_CONFIG=configs/specforge/gemma4-dflash2-smoke-overfit.yaml
#   ./scripts/slurm.sh submit scripts/job_gemma4_dflash_train.sh \
#       SPECFORGE_CONFIG=configs/specforge/gemma4-dflash-smoke-long.yaml
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=g4-dflash-train
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
# Features, venv and model copy are on this node's local disk.
#SBATCH --nodelist=hgx-h200-01
#SBATCH --time=04:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-dflash-train_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-dflash-train_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
SPECFORGE_CONFIG="${SPECFORGE_CONFIG:-configs/specforge/gemma4-dflash-smoke-overfit.yaml}"

# ── Environment ──────────────────────────────────────────────────────────────
# Node-local venv, model, features and caches: see scripts/hgx_local_env.sh.
source /shared/phucnlt2/speculative-training-pipeline/scripts/hgx_local_env.sh

echo "=== Gemma-4 DFlash train job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "SpecForge config: $SPECFORGE_CONFIG"
echo "Date:   $(date)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
# The 8-GPU node lists every GPU in nvidia-smi; only the allocated one may be used.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "✗ no GPU allocation visible"; exit 1; }
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline
link_local data/gemma4-dflash-smoke/hidden_states "$LOCAL_ROOT/gemma4-dflash-smoke/hidden_states"
OUTPUT_DIR="$(awk '/^output_dir:/{print $2}' "$SPECFORGE_CONFIG")"
# Checkpoints are ~25 GB each (optimizer state) and a run holds up to ~4 at once.
# The shared node root disk is too contended for that, so smoke runs write them to
# RAM (/dev/shm, counted against --mem) and drop them on exit; metrics live in the log.
CKPT_ROOT="${CKPT_ROOT:-/dev/shm/phucnlt2}"
CKPT_DIR="$CKPT_ROOT/$(basename "$OUTPUT_DIR")"
rm -rf "$CKPT_DIR"
case "$CKPT_ROOT" in /dev/shm/*) trap 'rm -rf "$CKPT_DIR"' EXIT ;; esac
link_local "$OUTPUT_DIR" "$CKPT_DIR"
echo "Outputs: $OUTPUT_DIR -> $(readlink "$OUTPUT_DIR")"

GPU_LOG="/shared/phucnlt2/job-logs/g4-dflash-train_${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
    -l 5 > "$GPU_LOG" &
SMI_PID=$!

specforge train --config "$SPECFORGE_CONFIG"
STATUS=$?
kill "$SMI_PID" 2>/dev/null
echo "---"
echo "Peak GPU memory: $(awk -F', ' 'max<$2{max=$2} END{print max}' "$GPU_LOG") MiB  (samples: $GPU_LOG)"

if [ $STATUS -ne 0 ]; then
    echo "✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi
case "$SPECFORGE_CONFIG" in
    *overfit*)
        echo ">>> Gate 4c: overfit curve"
        python scripts/check_gemma4_dflash.py curve \
            "/shared/phucnlt2/job-logs/g4-dflash-train_${SLURM_JOB_ID}.out" || exit 1 ;;
esac
# Keep the final checkpoint: SAVE_ROOT is set -> copy the newest checkpoint (training state, ~25 GB)
# and an HF-format draft export (weights only) out of /dev/shm before the EXIT trap drops it.
if [ -n "${SAVE_ROOT:-}" ]; then
    RUN_ID="$(awk '/^run_id:/{print $2}' "$SPECFORGE_CONFIG")"
    LAST_CKPT="$(ls -d "$CKPT_DIR"/"$RUN_ID"-step* | sort -V | tail -1)"
    DEST="$SAVE_ROOT/$RUN_ID"
    mkdir -p "$DEST"
    echo ">>> Saving $LAST_CKPT -> $DEST"
    cp -r "$LAST_CKPT" "$DEST/" || { echo "✗ checkpoint copy failed"; exit 1; }
    DRAFT_CFG="$(awk '/draft_model_config:/{print $2}' "$SPECFORGE_CONFIG")"
    python -m specforge.export.to_hf --checkpoint "$LAST_CKPT" \
        --draft-config "$DRAFT_CFG" --output-dir "$DEST/hf-draft" \
        || echo "! HF export failed (checkpoint copy is intact)"
    ls -la "$DEST" "$DEST/hf-draft" 2>&1
fi
echo "✓ Training job passed: $(date)"
