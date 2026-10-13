#!/bin/bash
#SBATCH --job-name=probe-cu
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/phucnlt2/job-logs/probe-cu_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/probe-cu_%j.err
export UV_CACHE_DIR=/tmp/phucnlt2/uv-cache
V=/tmp/phucnlt2/venv-sgl0521b; S=$V/lib/python3.10/site-packages
ls -la $V/.complete
/tmp/phucnlt2/uv pip list --python $V/bin/python 2>&1 | grep -iE "^torch|accelerate|nvidia-cuda|^transformers|^sglang |flashinfer|cuda-toolkit"
grep -h "define CUDA_VERSION" $S/nvidia/cu13/include/cuda.h
$V/bin/python -c "import accelerate; print('accelerate', accelerate.__version__)" 2>&1 | tail -2
grep -n "pip install" /shared/phucnlt2/speculative-training-pipeline/scripts/job_gemma4_sglang0521.sh
