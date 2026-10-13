#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm job: can a category-classify request reuse the entropy request's prefix cache?
# See scripts/probe_prefix_cache.py for the phases.
#
# Usage:
#   ./scripts/slurm.sh submit scripts/job_probe_prefix_cache.sh
#   ./scripts/slurm.sh submit scripts/job_probe_prefix_cache.sh \
#       MODEL=RedHatAI/gemma-4-31B-it-FP8-block MAX_MODEL_LEN=131072 \
#       SERVER_EXTRA_ARGS="--max-num-batched-tokens 4096" SERVER_ENV="VLLM_USE_DEEP_GEMM=0" \
#       DATA_PATH=data/gemma-4-31b-it/bench/2026-09-18_2026-09-25/rows.jsonl
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=probe-pcache
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
# han-3-1 has a CUDA 12.4 driver; venv-vllm torch is cu130 ("driver too old").
#SBATCH --exclude=hoanq3-h200-1x-han-3-1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/probe-pcache_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/probe-pcache_%j.err
#SBATCH --chdir=/shared/phucnlt2/speculative-training-pipeline

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-data/gateway-smoke/raw/train.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"  # space-separated vllm serve flags
SERVER_ENV="${SERVER_ENV:-}"                # space-separated KEY=VALUE for the server

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
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# vLLM runs from venv-vllm with its own CUDA wheels (see job_smoke_step1_data_quality.sh).
unset CUDA_HOME

VLLM_BIN="${VLLM_BIN:-/shared/phucnlt2/venv-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
    echo "✗ vLLM not found at $VLLM_BIN — run: ./scripts/slurm.sh setup-vllm"
    exit 1
fi

echo "=== Prefix-cache probe job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Model: $MODEL | Data: $DATA_PATH | Samples/set: $NUM_SAMPLES | Date: $(date)"
"${VLLM_BIN%/vllm}/python" -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---"

EXTRA=()
for a in $SERVER_EXTRA_ARGS; do EXTRA+=("--server-extra-arg=$a"); done
for kv in $SERVER_ENV; do EXTRA+=("--server-env=$kv"); done

python scripts/probe_prefix_cache.py "${EXTRA[@]}" \
    --model "$MODEL" \
    --data-path "$DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-model-len "$MAX_MODEL_LEN" \
    --vllm-executable "$VLLM_BIN" \
    --server-log /shared/phucnlt2/job-logs/probe-pcache_${SLURM_JOB_ID}_vllm.log \
    --output /shared/phucnlt2/job-logs/probe-pcache_${SLURM_JOB_ID}.json
STATUS=$?
echo "=== Probe finished (exit $STATUS): $(date) ==="
exit $STATUS
