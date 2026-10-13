#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm smoke test: Qwen3-4B EAGLE-3 pipeline verification
#
# Tests:
#   1. Data quality eval (model-based entropy via vLLM)
#   2. Hidden state + vocab mapping generation (SpecForge prepare_hidden_states.py)
#   3. EAGLE-3 training (50 steps)
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_smoke.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=smoke
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/phucnlt2/job-logs/smoke_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/smoke_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/smoke-qwen3-4b.yaml}"
DATA_PATH="${DATA_PATH:-data/gateway-smoke/raw/train.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_CONFIG="${DRAFT_CONFIG:-third_party/SpecForge/configs/qwen3-4b-eagle3.json}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-./data/gateway-smoke/hidden_states}"
CLEANUP="${CLEANUP:-1}"   # 1 = remove hidden states + cache after training

# ── Environment ──────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/shared/phucnlt2/venv}"
source "$VENV_DIR/bin/activate"
mkdir -p /shared/phucnlt2/job-logs

# Compute nodes don't have writable /home/phucnlt2 — force HOME and all cache
# dirs to /shared so flashinfer, triton, torch inductor, HF, etc. can write.
# Must override HOME unconditionally: Slurm sets HOME=/home/<user> (read-only
# on compute nodes), so ${HOME:-fallback} keeps the bad value.
# flashinfer uses FLASHINFER_WORKSPACE_BASE env var (falls back to pathlib.Path.home()
# which reads HOME).  Note: FLASHINFER_WORKSPACE_DIR is derived, NOT read from env.
export HOME=/shared/phucnlt2
export XDG_CACHE_HOME=/shared/phucnlt2/.cache
export HF_HOME=/shared/phucnlt2/hf-cache
export FLASHINFER_WORKSPACE_BASE=/shared/phucnlt2
export TORCHINDUCTOR_CACHE_DIR=/shared/phucnlt2/torchinductor-cache
export TRITON_CACHE_DIR=/shared/phucnlt2/triton-cache
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Python dev headers (python3.10-dev not installed on compute nodes, no sudo).
# Extracted from libpython3.10-dev .deb to /shared by slurm.sh setup.
# Triton's JIT compiler needs Python.h + pyconfig.h (multiarch) to build cuda_utils.c.
if [ -d /shared/phucnlt2/python-headers/usr/include/python3.10 ]; then
    PYINC="/shared/phucnlt2/python-headers/usr/include/python3.10:/shared/phucnlt2/python-headers/usr/include"
    export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${PYINC}:${CPLUS_INCLUDE_PATH:-}"
fi

# CUDA toolkit path (deep_gemm / sglang / flashinfer JIT need CUDA_HOME + nvcc)
# Auto-detect: check venv's nvidia/cu13 pip package first, then standard locations
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
        CUDA_HOME="$VENV_CUDA"  # fallback to venv package
    fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Smoke test job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Config: $CONFIG"
echo "HOME:    $HOME"
echo "HF_HOME: $HF_HOME"
echo "FLASHINFER_WORKSPACE_DIR: $FLASHINFER_WORKSPACE_DIR"
echo "CUDA_HOME: $CUDA_HOME ($(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc NOT FOUND'))"
echo "Date:   $(date)"
echo "---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

cd /shared/phucnlt2/speculative-training-pipeline

# ── Step 1: Data quality eval (model forward via vLLM) ──────────────────────
echo ">>> Step 1/3: Data quality evaluation (model-based entropy via vLLM)..."
spec-data data-quality \
    --data-path "$DATA_PATH" \
    --config "$CONFIG" \
    --method model \
    --engine vllm \
    --output /shared/phucnlt2/job-logs/smoke_${SLURM_JOB_ID}_quality.json
echo "---"

# ── Step 2: Prepare hidden states + vocab mapping (SpecForge) ────────────────
echo ">>> Step 2/3: Preparing hidden states + vocab mapping from ${TARGET_MODEL}..."
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
echo "---"

# ── Step 3: Train EAGLE-3 draft (50 steps) ───────────────────────────────────
echo ">>> Step 3/3: Training EAGLE-3 draft (50 steps)..."
specforge train --config configs/specforge/qwen3-4b-eagle3-smoke.yaml
echo "---"

echo "=== Smoke test complete: $(date) ==="
echo "=== Outputs:"
echo "    Quality report: /shared/phucnlt2/job-logs/smoke_${SLURM_JOB_ID}_quality.json"
echo "    Training log:   /shared/phucnlt2/job-logs/smoke_${SLURM_JOB_ID}.out"
echo "    Checkpoints:    ./outputs/qwen3-4b-eagle3-smoke/"

# ── Cleanup ──────────────────────────────────────────────────────────────────
if [ "$CLEANUP" = "1" ]; then
    echo ">>> Cleaning up hidden states + cache..."
    du -sh ./data/gateway-smoke/hidden_states ./cache 2>/dev/null
    rm -rf ./data/gateway-smoke/hidden_states ./cache
    echo "✓ Cleaned"
fi
