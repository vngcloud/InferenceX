#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: quality of the gemma-4-31b-it gateway bench (entropy tiers, traffic category)
#
# 1. Calibrate entropy thresholds for the target on SPEED-Bench (skipped when the
#    calibration JSON already exists; FORCE_CAL=1 to redo it).
# 2. `spec-data data-quality` on the bench rows (conversation + gateway response).
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_gemma4_data_quality.sh
#   ./scripts/slurm.sh submit scripts/job_gemma4_data_quality.sh FORCE_CAL=1
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gemma4-quality
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
# han-3-1 has a CUDA 12.4 driver; venv-vllm torch is cu130 ("driver too old").
#SBATCH --exclude=hoanq3-h200-1x-han-3-1
#SBATCH --time=04:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/gemma4-quality_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/gemma4-quality_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/gemma-4-31b-gateway.yaml}"
DATA_PATH="${DATA_PATH:-data/gemma-4-31b-it/bench/2026-09-18_2026-09-25/rows.jsonl}"
SPEEDBENCH_PATH="${SPEEDBENCH_PATH:-data/speedbench_inspect/throughput_1k.jsonl}"
PER_TIER="${PER_TIER:-100}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
FORCE_CAL="${FORCE_CAL:-0}"

# ── Environment ──────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/shared/phucnlt2/venv}"
source "$VENV_DIR/bin/activate"
mkdir -p /shared/phucnlt2/job-logs /shared/phucnlt2/entropy-calibration

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

# spec-data only tokenizes; the vLLM server runs from venv-vllm with its own CUDA wheels.
unset CUDA_HOME

echo "=== Gemma 4 data quality job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Config: $CONFIG"
echo "Data:   $DATA_PATH"
echo "Date:   $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline

VLLM_BIN="${VLLM_BIN:-/shared/phucnlt2/venv-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
    echo "✗ vLLM not found at $VLLM_BIN — run: ./scripts/slurm.sh setup-vllm"
    exit 1
fi
for f in "$DATA_PATH" "$SPEEDBENCH_PATH"; do
    if [ ! -f "$f" ]; then
        echo "✗ $f missing — ./scripts/slurm.sh sync copies it from the local repo"
        exit 1
    fi
done

CAL_PATH=$(python -c "from spec_data.config import PipelineConfig as P; print(P.from_yaml('$CONFIG').data_quality.entropy_calibration_path)")

# ── Step 1: entropy calibration ──────────────────────────────────────────────
if [ "$FORCE_CAL" = "1" ] || [ ! -f "$CAL_PATH" ]; then
    echo ">>> Step 1: calibrating entropy thresholds → $CAL_PATH"
    spec-data calibrate-entropy \
        --config "$CONFIG" \
        --speedbench-path "$SPEEDBENCH_PATH" \
        --per-tier "$PER_TIER" \
        --max-tokens "$MAX_TOKENS"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "✗ Calibration FAILED (exit $STATUS)"
        exit $STATUS
    fi
else
    echo ">>> Step 1: reusing calibration $CAL_PATH (FORCE_CAL=1 to redo)"
fi
echo "---"

# ── Step 2: data quality ─────────────────────────────────────────────────────
REPORT="/shared/phucnlt2/job-logs/gemma4-quality_${SLURM_JOB_ID}_quality.json"
PER_SAMPLE="/shared/phucnlt2/job-logs/gemma4-quality_${SLURM_JOB_ID}_per_sample.jsonl"
echo ">>> Step 2: data quality on $DATA_PATH"
spec-data data-quality \
    --data-path "$DATA_PATH" \
    --config "$CONFIG" \
    --method model \
    --engine vllm \
    --output "$REPORT" \
    --per-sample-output "$PER_SAMPLE"
STATUS=$?
echo "---"
if [ $STATUS -eq 0 ]; then
    echo "✓ Data quality PASSED (exit 0)"
    echo "  Quality report: $REPORT"
    echo "  Per-sample (category, entropy tier): $PER_SAMPLE"
else
    echo "✗ Data quality FAILED (exit $STATUS)"
fi
echo "=== Complete: $(date) ==="
exit $STATUS
