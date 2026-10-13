#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: SpecForge draft model training
#
# Usage:
#   sbatch scripts/job_train.sh                          # defaults
#   sbatch scripts/job_train.sh --config configs/glm-5.2.yaml
#
# Or via the wrapper:
#   ./scripts/slurm.sh submit scripts/job_train.sh
#
# Override defaults via env vars before sbatch:
#   NUM_GPUS=4 CONFIG=configs/glm-4.7-flash.yaml sbatch scripts/job_train.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=spec-train
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/spec-train_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/spec-train_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config (override via env vars) ───────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG="${CONFIG:-configs/glm-5.2.yaml}"
SPECFORGE_CONFIG="${SPECFORGE_CONFIG:-}"
SKIP_DATA="${SKIP_DATA:-1}"   # 1 = skip data prep, 0 = run collect+regenerate

# ── Environment setup ────────────────────────────────────────────────────────
# venv created once via `./scripts/slurm.sh setup` (on login node).
# /shared is NFS → visible from compute nodes.
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
# Triton's JIT compiler needs Python.h to build cuda_utils.c.
if [ -d /shared/phucnlt2/python-headers/usr/include/python3.10 ]; then
    export C_INCLUDE_PATH="/shared/phucnlt2/python-headers/usr/include/python3.10:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="/shared/phucnlt2/python-headers/usr/include/python3.10:${CPLUS_INCLUDE_PATH:-}"
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

echo "=== Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Config: $CONFIG"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "User:   $(whoami)"
echo "Date:   $(date)"
echo "---"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

# ── Training ─────────────────────────────────────────────────────────────────
cd /shared/phucnlt2/speculative-training-pipeline

if [ "$SKIP_DATA" = "0" ]; then
    echo ">>> Running data prep (collect + regenerate)..."
    spec-data collect --config "$CONFIG"
    spec-data regenerate --config "$CONFIG"
fi

echo ">>> Starting training..."

if [ -n "$SPECFORGE_CONFIG" ]; then
    spec-data train --config "$CONFIG" --specforge-config "$SPECFORGE_CONFIG" --skip-data
else
    spec-data train --config "$CONFIG" --skip-data
fi

echo "=== Training complete: $(date) ==="
echo "=== Checkpoint in: $(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['specforge_run']['output_dir'])" 2>/dev/null || echo 'check config') ==="
