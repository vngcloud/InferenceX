#!/bin/bash
#SBATCH --job-name=g4-sglfix
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-sglfix_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-sglfix_%j.err
# Set up the patched sglang (Noridom1/sglang branch gemma4-aux-clone = v0.5.21 + PR #39286 backport)
# in a fresh node-local venv, then gate it: the Gemma-4 aux captures must match HF
# (L57 cosine ~0.967; stock 0.5.21 gives 0.715). Same flow as job_gemma4_sglang0521.sh.
# hgx-h200-02 variant: job_gemma4_sglang_fix_hgx02.sh.
set -uo pipefail
R=/tmp/phucnlt2; V=$R/venv-sgl0521fix; FORK=$R/sglang-fork
FORK_URL=https://github.com/Noridom1/sglang.git; FORK_BRANCH=gemma4-aux-clone
mkdir -p $R
export UV_PYTHON_INSTALL_DIR="$R/uv-python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$R/uv-cache"
export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5
UV="$R/uv"; [ -x "$UV" ] || UV="$(command -v uv)"
[ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$R" sh; UV="$R/uv"; }

# Target model: hgx-h200-02 has no /mnt/hf_hub_cache, so it uses a node-local copy.
if [ "${HF_HUB_CACHE:-}" = "$R/hf_hub_cache" ]; then
    SRC=/shared/phucnlt2/hf-cache/hub/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots
    DST=$R/hf_hub_cache/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots
    for s in $(ls $SRC); do
        mkdir -p $DST/$s && cp -rL --update=none $SRC/$s/. $DST/$s/ || { echo "✗ model copy failed"; exit 1; }
    done
fi

# Patched fork source (always re-fetched so a pushed fix is picked up).
rm -rf $FORK
timeout 10m git clone --depth 1 -b $FORK_BRANCH $FORK_URL $FORK || { echo "✗ fork clone failed"; exit 1; }
git -C $FORK log --oneline -1
grep -n "hidden_states.clone()" $FORK/python/sglang/srt/models/gemma4_causal.py || { echo "✗ fix missing in fork"; exit 1; }

# Deps come from the 0.5.21 wheel (same two-step install as job_gemma4_sglang0521.sh), then the fork's
# python/ replaces the wheel's sglang package (--no-deps).
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
# SGLANG_BUILD_RUST_EXTS=none: the build otherwise wants cargo; the Rust extensions are not needed here.
SGLANG_BUILD_RUST_EXTS=none "$UV" pip install --no-deps --reinstall --python $V/bin/python $FORK/python || { echo "✗ fork install failed"; exit 1; }
# HOME is unwritable on the nodes; torch/transformers create cache dirs at import.
export HOME=/shared/phucnlt2 XDG_CACHE_HOME=$R/xdg-cache TORCHINDUCTOR_CACHE_DIR=$R/torchinductor-cache
$V/bin/python - <<'PY' || exit 1
import inspect, sglang, torch
from sglang.srt.models import gemma4_causal
print('sglang', sglang.__version__, 'from', sglang.__file__, 'torch', torch.__version__)
assert 'hidden_states.clone()' in inspect.getsource(gemma4_causal), 'installed sglang lacks the clone fix'
print('✓ installed sglang carries the aux-capture clone fix')
PY

cd /shared/phucnlt2/speculative-training-pipeline
NODE=$(hostname)
TAG=sgl0521fix-$NODE
export VENV_DIR=$V ENGINE=sglang TAG DUMP=1 SGL_LAYERS=1 FORCED=1 SPLITS=overfit NUM_ROWS=32 PORT=$((20000 + SLURM_JOB_ID % 20000))
bash scripts/job_gemma4_engine_verify.sh || echo "✗ engine verify failed"

source scripts/hgx_local_env.sh
TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/* | head -1)"
T=/shared/phucnlt2/verify-results/$TAG
ls $T
AUXF=$(ls $T/aux.sglang.*.pt | head -3 | tr '\n' ' ')
python scripts/verify_engine/ref_hidden.py --model "$TARGET_PATH" --data data/gemma4-dflash-smoke/overfit.jsonl \
    --out "$T/aux.ref.pt" --layers-cmp "$T/layers.sglang.pt" \
    --cmp $(for f in $AUXF; do echo "sglfix_$(basename $f .pt)=$f"; done)
echo "✓ done $NODE $(date)"
