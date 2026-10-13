#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: fit model-based entropy thresholds on SPEED-Bench (one-off per model)
#
# The target model answers SPEED-Bench throughput prompts (low/mixed/high tiers),
# the answers are scored exactly like the data-quality check, and the thresholds that
# best separate the tiers are written to data_quality.entropy_calibration_path.
# Run this before step 1 whenever the target model or entropy settings change.
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_calibrate_entropy.sh
#   ./scripts/slurm.sh submit scripts/job_calibrate_entropy.sh PER_TIER=200
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=entropy-cal
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/phucnlt2/job-logs/entropy-cal_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/entropy-cal_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/smoke-qwen3-4b.yaml}"
SPEEDBENCH_PATH="${SPEEDBENCH_PATH:-data/speedbench_inspect/throughput_1k.jsonl}"
PER_TIER="${PER_TIER:-100}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

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

echo "=== Entropy calibration job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Config:      $CONFIG"
echo "SPEED-Bench: $SPEEDBENCH_PATH ($PER_TIER prompts/tier, max_tokens=$MAX_TOKENS)"
echo "Date:        $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline

VLLM_BIN="${VLLM_BIN:-/shared/phucnlt2/venv-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
    echo "✗ vLLM not found at $VLLM_BIN — run: ./scripts/slurm.sh setup-vllm"
    exit 1
fi
if [ ! -f "$SPEEDBENCH_PATH" ]; then
    echo "✗ $SPEEDBENCH_PATH missing — ./scripts/slurm.sh sync copies it from the local repo"
    exit 1
fi

spec-data calibrate-entropy \
    --config "$CONFIG" \
    --speedbench-path "$SPEEDBENCH_PATH" \
    --per-tier "$PER_TIER" \
    --max-tokens "$MAX_TOKENS"
STATUS=$?
echo "---"
if [ $STATUS -eq 0 ]; then
    echo "✓ Calibration PASSED (exit 0)"
else
    echo "✗ Calibration FAILED (exit $STATUS)"
fi
echo "=== Calibration complete: $(date) ==="
exit $STATUS
