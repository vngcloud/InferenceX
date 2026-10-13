#!/bin/bash
# One-time node-local setup on hgx-h200-02 (no /tmp/phucnlt2, no /mnt/hf_hub_cache model there):
#   - sglang venv (same freeze as /shared/phucnlt2/venv-freeze.txt) at /tmp/phucnlt2/venv-py310
#   - target model snapshot copied from the shared HF cache to /tmp/phucnlt2/hf_hub_cache
# Afterwards run engine-verify jobs with HF_HUB_CACHE=/tmp/phucnlt2/hf_hub_cache.
#SBATCH --job-name=setup-hgx02
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hgx-h200-02
#SBATCH --time=03:00:00
#SBATCH --output=/shared/phucnlt2/job-logs/setup-hgx02_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/setup-hgx02_%j.err
set -u
R=/tmp/phucnlt2; mkdir -p $R
export UV_PYTHON_INSTALL_DIR="$R/uv-python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$R/uv-cache"
UV="$R/uv"
[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$R" sh
if [ ! -f $R/venv-py310/.complete ]; then
    rm -rf $R/venv-py310
    "$UV" venv --python 3.10 $R/venv-py310 \
      && "$UV" pip install --no-deps --python $R/venv-py310/bin/python -r /shared/phucnlt2/venv-freeze.txt \
      && touch $R/venv-py310/.complete
fi
[ -f $R/venv-py310/.complete ] || { echo "✗ venv install failed"; exit 1; }
$R/venv-py310/bin/python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
SRC=/shared/phucnlt2/hf-cache/hub/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots
DST=$R/hf_hub_cache/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots
for s in $(ls $SRC); do
    mkdir -p $DST/$s && cp -rL --update=none $SRC/$s/. $DST/$s/ || { echo "✗ model copy failed"; exit 1; }
done
du -sh $R/hf_hub_cache; ls $DST/*/ | head; df -h /tmp | tail -1
echo "✓ setup done $(date)"
