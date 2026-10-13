#!/bin/bash
#SBATCH --job-name=inspect-feat
#SBATCH --partition=hardware-hcm
#SBATCH --nodes=1
#SBATCH --nodelist=hgx-h200-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/shared/phucnlt2/job-logs/inspect-feat_%j.out
#SBATCH --error=/shared/phucnlt2/job-logs/inspect-feat_%j.err
# Print the layout of the stored SpecForge training features (node-local on hgx-h200-01).
D=/tmp/phucnlt2/gemma4-dflash-smoke/hidden_states/overfit
ls "$D"; ls "$D/rows_0-2000" | head -5; ls "$D/rows_0-2000" | wc -l; du -sh "$D"
source /tmp/phucnlt2/venv-py310/bin/activate
python - <<PY
import glob, torch
fs = sorted(glob.glob("$D/rows_0-2000/*"))
print("files:", len(fs), fs[:2])
d = torch.load(fs[0], map_location="cpu")
for k, v in d.items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else type(v), getattr(v, "dtype", ""))
PY
