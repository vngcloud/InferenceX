#!/bin/bash
#SBATCH --job-name=g4-sgl0521
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-sgl0521_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-sgl0521_%j.err
# Does the Gemma-4 aux-capture aliasing bug exist in sglang 0.5.21 (latest)?
# Fresh node-local venv (no shared venv touched), stock engine, DUMP=1 captures + per-layer outputs,
# then HF reference compare. Run on the same prompt/draft as the 0.5.18 runs.
set -uo pipefail
R=/tmp/phucnlt2; V=$R/venv-sgl0521d
export UV_PYTHON_INSTALL_DIR="$R/uv-python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$R/uv-cache"
export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5
UV="$R/uv"; [ -x "$UV" ] || UV="$(command -v uv)"
if [ ! -f $V/.complete ]; then
    rm -rf $V
    "$UV" venv --python 3.10 $V && for i in 1 2 3; do
      timeout 20m "$UV" pip install --python $V/bin/python "sglang==0.5.21" pytest accelerate \
        torch==2.13.0 torchvision==0.28.0 \
      && "$UV" pip install --no-deps --python $V/bin/python nvidia-cuda-runtime==13.4.92 \
        nvidia-cuda-cupti==13.4.92 nvidia-cuda-nvrtc==13.4.92 && touch $V/.complete && break
      echo "install attempt $i failed/stalled, retrying"; done
fi
[ -f $V/.complete ] || { echo "✗ venv install failed"; exit 1; }
# 0.5.21 alone resolves cuda-runtime/cupti/nvrtc 13.0.x next to nvcc 13.4.92 (flashinfer JIT "headers incompatible"),
# so those are force-installed (--no-deps; torch's cuda-toolkit pin forbids a joint resolve) at the 0.5.18 venv's 13.4.92.
$V/bin/python -c "import sglang, torch, transformers; print('sglang', sglang.__version__, 'torch', torch.__version__, 'tf', transformers.__version__)"

cd /shared/phucnlt2/speculative-training-pipeline
export VENV_DIR=$V ENGINE=sglang TAG=sgl0521-dflash DUMP=1 SGL_LAYERS=1 FORCED=1 SPLITS=overfit NUM_ROWS=32 PORT=$((20000 + SLURM_JOB_ID % 20000))
bash scripts/job_gemma4_engine_verify.sh || echo "✗ engine verify failed"

source scripts/hgx_local_env.sh
TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/* | head -1)"
V2=/shared/phucnlt2/verify-results; T=$V2/sgl0521-dflash
ls $T
AUXF=$(ls $T/aux.sglang.*.pt | head -3 | tr '\n' ' ')
python scripts/verify_engine/ref_hidden.py --model "$TARGET_PATH" --data data/gemma4-dflash-smoke/overfit.jsonl \
    --out "$T/aux.ref.pt" --layers-cmp "$T/layers.sglang.pt" \
    --cmp $(for f in $AUXF; do echo "sgl0521_$(basename $f .pt)=$f"; done)
echo "✓ done $(date)"
