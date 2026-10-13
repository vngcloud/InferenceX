#!/bin/bash
#SBATCH --job-name=g4-capture-fix
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/g4-capture-fix_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/g4-capture-fix_%j.err
# Re-capture the Gemma-4 DFlash smoke features with the patched sglang (gemma4-aux-clone,
# v0.5.21 + #39286) into hidden_states_clean, then gate them against HF and the old features.
# Plan: __plan__/[05-10-26] gemma4-recapture-retrain-and-embed-scale.md (task 2).
# Usage: ./scripts/slurm.sh submit scripts/job_gemma4_dflash_capture_fix.sh ["SPLITS=overfit heldout"]
set -uo pipefail
R=/tmp/phucnlt2; V=$R/venv-sgl0521fix-sf; FORK=$R/sglang-fork
FORK_URL=https://github.com/Noridom1/sglang.git; FORK_BRANCH=gemma4-aux-clone
REPO=/shared/phucnlt2/speculative-training-pipeline
mkdir -p $R
export UV_PYTHON_INSTALL_DIR="$R/uv-python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$R/uv-cache"
export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=5
UV="$R/uv"; [ -x "$UV" ] || UV="$(command -v uv)"
[ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$R" sh; UV="$R/uv"; }

rm -rf $FORK
timeout 10m git clone --depth 1 -b $FORK_BRANCH $FORK_URL $FORK || { echo "✗ fork clone failed"; exit 1; }
git -C $FORK log --oneline -1
grep -n "hidden_states.clone()" $FORK/python/sglang/srt/models/gemma4_causal.py || { echo "✗ fix missing in fork"; exit 1; }

# Route A: the verified patched-sglang recipe plus the SpecForge deps (SpecForge pins transformers 5.12.1).
if [ ! -f $V/.complete ]; then
    rm -rf $V
    "$UV" venv --python 3.10 $V && for i in 1 2 3; do
      timeout 20m "$UV" pip install --python $V/bin/python "sglang==0.5.21" pytest accelerate \
        torch==2.13.0 torchvision==0.28.0 \
      && "$UV" pip install --no-deps --python $V/bin/python nvidia-cuda-runtime==13.4.92 \
        nvidia-cuda-cupti==13.4.92 nvidia-cuda-nvrtc==13.4.92 && touch $V/.partial && break
      echo "install attempt $i failed/stalled, retrying"; done
    [ -f $V/.partial ] || { echo "✗ base venv install failed"; exit 1; }
    "$UV" pip install --python $V/bin/python "transformers==5.12.1" datasets tqdm click huggingface-hub \
        numpy openai-harmony pydantic psutil pyyaml safetensors requests tensorboard wandb yunchang \
        || { echo "✗ SpecForge deps failed"; exit 1; }
    touch $V/.complete
fi
SGLANG_BUILD_RUST_EXTS=none "$UV" pip install --no-deps --reinstall --python $V/bin/python $FORK/python || { echo "✗ fork install failed"; exit 1; }
"$UV" pip install --no-deps --python $V/bin/python -e $REPO/third_party/SpecForge || { echo "✗ specforge install failed"; exit 1; }
"$UV" pip check --python $V/bin/python 2>&1 | head -20 || true

export HOME=/shared/phucnlt2 XDG_CACHE_HOME=$R/xdg-cache TORCHINDUCTOR_CACHE_DIR=$R/torchinductor-cache
$V/bin/python - <<'PY' || exit 1
import inspect, sglang, torch, transformers, specforge
from sglang.srt.models import gemma4_causal
print('sglang', sglang.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)
assert 'hidden_states.clone()' in inspect.getsource(gemma4_causal), 'installed sglang lacks the clone fix'
print('✓ G2a: installed sglang carries the aux-capture clone fix')
PY

cd $REPO
export VENV_DIR=$V FEAT_NAME=hidden_states_clean
SPLITS="${SPLITS:-overfit heldout}" bash scripts/job_gemma4_dflash_capture.sh || { echo "✗ capture failed"; exit 1; }

# G2c/G2d: new features vs HF, and new vs old.
source scripts/hgx_local_env.sh
TARGET_PATH="$(ls -d "$HF_HUB_CACHE"/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/* | head -1)"
N=$R/gemma4-dflash-smoke/hidden_states_clean/overfit; O=$R/gemma4-dflash-smoke/hidden_states/overfit
ND=$(dirname "$(ls $N/*/data_0.ckpt | head -1)"); OD=$(dirname "$(ls $O/*/data_0.ckpt | head -1)")
echo ">>> G2c: new features vs HF"
python scripts/verify_engine/ref_vs_stored.py --model "$TARGET_PATH" --ckpt $ND/data_0.ckpt $ND/data_1.ckpt $ND/data_2.ckpt
echo ">>> G2d: new vs old features (cosine per layer, first 1200 tokens)"
python - "$ND" "$OD" <<'PY'
import sys, torch
F = torch.nn.functional
for i in range(3):
    n = torch.load(f"{sys.argv[1]}/data_{i}.ckpt", map_location="cpu"); o = torch.load(f"{sys.argv[2]}/data_{i}.ckpt", map_location="cpu")
    same = torch.equal(n["input_ids"], o["input_ids"]); T = min(1200, len(n["input_ids"]))
    cs = [F.cosine_similarity(n["hidden_states"][0, :T, j*5376:(j+1)*5376].float(), o["hidden_states"][0, :T, j*5376:(j+1)*5376].float(), dim=-1).mean().item() for j in range(5)]
    print(f"row {i}: same input_ids={same} cos(new,old) per layer [1,15,29,43,57] =", [round(c, 4) for c in cs])
PY
echo "✓ done $(date)"
