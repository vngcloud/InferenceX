#!/bin/bash
# hgx-h200-02 variant of job_gemma4_engine_verify.sh (needs job_setup_hgx02.sh to have run once).
# Same KEY=VALUE params, e.g.:
#   ./scripts/slurm.sh submit scripts/job_gemma4_engine_verify_hgx02.sh ENGINE=sglang TAG=dump-sglang-dflash DUMP=1 FORCED=1 SPLITS=overfit NUM_ROWS=1
#SBATCH --job-name=g4-engine-verify-02
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hgx-h200-02
#SBATCH --time=02:30:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-engine-verify-02_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-engine-verify-02_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

export HF_HUB_CACHE=/tmp/phucnlt2/hf_hub_cache
source /shared/phucnlt2/speculative-training-pipeline/scripts/job_gemma4_engine_verify.sh
