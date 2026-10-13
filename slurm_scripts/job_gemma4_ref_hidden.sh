#!/bin/bash
#SBATCH --job-name=g4-ref-hidden
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-ref-hidden_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-ref-hidden_%j.err
# HF reference for the aux hidden states (which engine matches the unfused model?).
set -eo pipefail
cd /shared/phucnlt2/speculative-training-pipeline
source scripts/hgx_local_env.sh
TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/* | head -1)"
R=/shared/phucnlt2/verify-results
python scripts/verify_engine/ref_hidden.py --model "$TARGET_PATH" --data data/gemma4-dflash-smoke/overfit.jsonl \
    --out "$R/dump2-ref/aux.ref.pt" \
    --layers-cmp "$R/layers-sglang-dflash/layers.sglang.pt" --cmp sglang="$R/dump2-sglang-dflash/aux.sglang.2.pt" vllm="$R/dump2-t-vllm-dflash/aux.vllm.3.pt"
echo "✓ done $(date)"
