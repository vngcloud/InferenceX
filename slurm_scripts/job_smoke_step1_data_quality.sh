#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: Step 1 — Data quality evaluation (model-based entropy via vLLM)
#
# Usage:
#   sbatch scripts/job_smoke_step1_data_quality.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=smoke-s1
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/phucnlt2/job-logs/smoke-s1_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/smoke-s1_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/smoke-qwen3-4b.yaml}"
DATA_PATH="${DATA_PATH:-data/gateway-smoke/raw/train.jsonl}"

# ── Environment ──────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/shared/phucnlt2/venv}"
source "$VENV_DIR/bin/activate"
mkdir -p /shared/phucnlt2/job-logs

export HOME=/shared/phucnlt2
export XDG_CACHE_HOME=/shared/phucnlt2/.cache
export HF_HOME=/shared/phucnlt2/hf-cache
export FLASHINFER_WORKSPACE_BASE=/shared/phucnlt2
export TORCHINDUCTOR_CACHE_DIR=/shared/phucnlt2/torchinductor-cache
export TRITON_CACHE_DIR=/shared/phucnlt2/triton-cache
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if [ -d /shared/phucnlt2/python-headers/usr/include/python3.10 ]; then
    PYINC="/shared/phucnlt2/python-headers/usr/include/python3.10:/shared/phucnlt2/python-headers/usr/include"
    export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${PYINC}:${CPLUS_INCLUDE_PATH:-}"
fi

# No CUDA_HOME here on purpose: spec-data (this venv) only tokenizes, and the vLLM
# server runs from venv-vllm with its own CUDA wheels. Exporting this venv's cu13
# toolkit made FlashInfer JIT-link against it and fail (`cannot find -lcudart`).
unset CUDA_HOME

echo "=== Step 1 (data quality) job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Config: $CONFIG"
echo "Data:   $DATA_PATH"
echo "Date:   $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline

# vLLM runs from its own venv (see entropy_vllm_executable in $CONFIG).
VLLM_BIN="${VLLM_BIN:-/shared/phucnlt2/venv-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
    echo "✗ vLLM not found at $VLLM_BIN — run: ./scripts/slurm.sh setup-vllm"
    exit 1
fi
echo "vLLM:   $VLLM_BIN"

echo ">>> Step 1: Data quality evaluation (model-based entropy via vLLM)..."
spec-data data-quality \
    --data-path "$DATA_PATH" \
    --config "$CONFIG" \
    --method model \
    --engine vllm \
    --output /shared/phucnlt2/job-logs/smoke-s1_${SLURM_JOB_ID}_quality.json
STATUS=$?
echo "---"

if [ $STATUS -eq 0 ]; then
    echo "✓ Step 1 PASSED (exit 0)"
    echo "  Quality report: /shared/phucnlt2/job-logs/smoke-s1_${SLURM_JOB_ID}_quality.json"
else
    echo "✗ Step 1 FAILED (exit $STATUS)"
fi
echo "=== Step 1 complete: $(date) ==="
