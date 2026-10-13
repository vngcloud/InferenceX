#!/bin/bash
# Free node-local space under /tmp/phucnlt2 on the test node (our own leftovers only; keeps the vLLM venv).
#SBATCH --job-name=clean-test-tmp
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hoanq3-h200-1x-han-3
#SBATCH --time=00:15:00
#SBATCH --output=/shared/phucnlt2/job-logs/clean-test-tmp_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/clean-test-tmp_%j.err
R=/tmp/phucnlt2
echo "--- before"; df -h /tmp | tail -1; du -sh $R/* 2>/dev/null
for d in drafts xdg-cache uv-cache torchinductor-cache triton-cache; do rm -rf "${R:?}/$d"; done
echo "--- after"; df -h /tmp | tail -1; du -sh $R/* 2>/dev/null
