#!/bin/bash
# hgx-h200-02 variant of job_gemma4_sglang_fix.sh (node-local model copy instead of /mnt/hf_hub_cache).
#SBATCH --job-name=g4-sglfix-02
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-02
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-sglfix-02_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-sglfix-02_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

export HF_HUB_CACHE=/tmp/phucnlt2/hf_hub_cache
source /shared/phucnlt2/speculative-training-pipeline/scripts/job_gemma4_sglang_fix.sh
