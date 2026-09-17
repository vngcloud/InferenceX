"""Copy the pinned client out of the read-only checkout and pin its HF revision."""
import hashlib
import json
from pathlib import Path
import shutil
from campaign import DATASET_REV

dest=Path('/tmp/agentx-aiperf')
shutil.copytree('/repo/utils/aiperf',dest)
file=dest/'src/aiperf/dataset/loader/base_hf_dataset.py'
source=file.read_text()
old='hf_revision: str | None = None'
assert source.count(old)==1
new=f'hf_revision: str | None = "{DATASET_REV}"'
file.write_text(source.replace(old,new))
Path('/results/client-pin.json').write_text(json.dumps(dict(
    modification='Set existing HF loader revision default; replay algorithm unchanged',
    old=old,new=new,sha256=hashlib.sha256(file.read_bytes()).hexdigest()),indent=2))
