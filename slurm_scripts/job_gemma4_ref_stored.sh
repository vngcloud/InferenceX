#!/bin/bash
#SBATCH --job-name=g4-ref-stored
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-ref-stored_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-ref-stored_%j.err
# Stored SpecForge training features vs HF reference (were the training features the deviated ones?).
set -eo pipefail
cd /shared/phucnlt2/speculative-training-pipeline
source scripts/hgx_local_env.sh
TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/* | head -1)"
R=/shared/phucnlt2/verify-results
D=/tmp/phucnlt2/gemma4-dflash-smoke/hidden_states/overfit/rows_0-2000
python scripts/verify_engine/ref_vs_stored.py --model "$TARGET_PATH" --ckpt $D/data_0.ckpt $D/data_1.ckpt $D/data_2.ckpt
echo "✓ done $(date)"
