"""Summarize the original stat recorder's globally reduced logical counts."""
import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1])
files=list(root.glob('expert_distribution_recorder_*.pt'))
assert len(files)==1, f'Expected one global stat dump, found {len(files)}'
data=torch.load(files[0],map_location='cpu',weights_only=True)
counts=data['logical_count'].float()
assert counts.ndim==3 and counts.shape[-1]==256 and 0<counts.shape[0]<=200, counts.shape
# stat output is already globally reduced; never sum it again across TP ranks.
total=counts.sum(-1)
valid=total>0
def quantiles(x):
    x=x[valid]
    return {'p50':x.quantile(.5).item(),'p95':x.quantile(.95).item()} if x.numel() else None
result={'buffer_slots':counts.shape[0],'nonempty_slots':int((total.sum(-1)>0).sum()),'layers':counts.shape[1],
    'hottest_expert_share':quantiles(counts.max(-1).values/total.clamp_min(1)),
    'empty_expert_fraction':quantiles((counts==0).float().mean(-1)),
    'prefill_decode_labels':'unavailable','rank_load':'unavailable until expert placement is verified',
    'counts_source':'single globally reduced stat dump; no TP replica summation',
    'limitations':'circular-buffer chronology and cross-rank forward-pass alignment unavailable; pooled slots are not globally aligned iterations'}
(root/'expert-summary.json').write_text(json.dumps(result,indent=2))
