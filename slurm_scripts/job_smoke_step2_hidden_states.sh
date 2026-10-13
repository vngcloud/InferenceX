#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: Step 2 — Hidden states + vocab mapping (SpecForge prepare_hidden_states)
#
# Usage:
#   sbatch scripts/job_smoke_step2_hidden_states.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=smoke-s2
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/phucnlt2/job-logs/smoke-s2_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/smoke-s2_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH="${DATA_PATH:-data/gateway-smoke/raw/train.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_CONFIG="${DRAFT_CONFIG:-third_party/SpecForge/configs/qwen3-4b-eagle3.json}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-./data/gateway-smoke/hidden_states}"

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

if [ -z "${CUDA_HOME:-}" ]; then
    VENV_CUDA="$VENV_DIR/lib/python3.10/site-packages/nvidia/cu13"
    if [ -x "$VENV_CUDA/bin/nvcc" ]; then
        CUDA_HOME="$VENV_CUDA"
    elif command -v nvcc &>/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [ -d /usr/local/cuda ] && [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_HOME=/usr/local/cuda
    elif [ -d /opt/cuda ] && [ -x /opt/cuda/bin/nvcc ]; then
        CUDA_HOME=/opt/cuda
    else
        CUDA_HOME="$VENV_CUDA"
    fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Step 2 (hidden states) job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Target model: $TARGET_MODEL"
echo "Data:         $DATA_PATH"
echo "Output dir:   $HIDDEN_STATES_DIR"
echo "CUDA_HOME: $CUDA_HOME ($(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc NOT FOUND'))"
echo "Date:   $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline

echo ">>> Step 2: Preparing hidden states + vocab mapping from ${TARGET_MODEL}..."
torchrun --nproc_per_node=1 \
    third_party/SpecForge/scripts/prepare_hidden_states.py \
    --target-model-path "$TARGET_MODEL" \
    --data-path "$DATA_PATH" \
    --output-path "$HIDDEN_STATES_DIR" \
    --draft-model-config "$DRAFT_CONFIG" \
    --strategy eagle3 \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 8 \
    --sglang-mem-fraction-static 0.8
STATUS=$?
echo "---"

if [ $STATUS -eq 0 ]; then
    echo "✓ Step 2 PASSED (exit 0)"
    echo "  Hidden states: $HIDDEN_STATES_DIR"
else
    echo "✗ Step 2 FAILED (exit $STATUS)"
fi
echo "=== Step 2 complete: $(date) ==="
