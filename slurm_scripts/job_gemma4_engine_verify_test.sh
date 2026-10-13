#!/bin/bash
# Test-partition variant of job_gemma4_engine_verify.sh (1x H200 nodes, no hgx-local venvs, ~23 GB free
# disk, target read from the /shared NFS cache). vLLM only: the node has no room for the sglang venv.
# Same KEY=VALUE params as the main script:
#   ./scripts/slurm.sh submit scripts/job_gemma4_engine_verify_test.sh TAG=drift-vllm-dflash-t SPLITS=overfit
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=g4-engine-verify-t
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hoanq3-h200-1x-han-3   # han-3-1 has a CUDA 12.4 driver: too old for torch cu130
#SBATCH --time=04:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-engine-verify-t_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-engine-verify-t_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

export TEST_NODE=1 ENGINE="${ENGINE:-vllm}" READY_ITERS="${READY_ITERS:-1440}"   # ≤ 2 h for the cold NFS load
source /shared/phucnlt2/speculative-training-pipeline/scripts/job_gemma4_engine_verify.sh
