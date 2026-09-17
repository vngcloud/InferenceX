"""Summarize the original stat recorder's globally reduced logical counts."""
import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1])

def summarize(files):
    assert files, 'No expert-distribution dumps found'
    payloads=[torch.load(f,map_location='cpu',weights_only=True) for f in sorted(files)]
    counts=payloads[0]['logical_count'].float()
    assert counts.ndim==3 and counts.shape[-1]==256 and 0<counts.shape[0]<=200, counts.shape
# EP8 writes one globally reduced dump. EP1 writes the same logical router
# counts once per TP rank. Fail closed unless every additional file is an exact
# replica; replicated logical counts must never be summed.
    for payload in payloads[1:]:
        candidate=payload['logical_count'].float()
        assert torch.equal(candidate,counts), 'Multiple recorder dumps are not identical TP replicas'
    total=counts.sum(-1)
    valid=total>0
    def quantiles(x):
        x=x[valid]
        return {'p50':x.quantile(.5).item(),'p95':x.quantile(.95).item()} if x.numel() else None
    return {'buffer_slots':counts.shape[0],'nonempty_slots':int((total.sum(-1)>0).sum()),'layers':counts.shape[1],
        'dump_files':len(files),'identical_tp_replicas':len(files),
        'selections_per_layer_pass':quantiles(total),
        'hottest_expert_share':quantiles(counts.max(-1).values/total.clamp_min(1)),
        'empty_expert_fraction':quantiles((counts==0).float().mean(-1)),
        'rank_load':'unavailable until expert placement is verified',
        'counts_source':'one logical-count view; exact TP replicas verified and deduplicated'}

phase_dirs=[p for p in (root/'prefill',root/'decode') if p.is_dir()]
if phase_dirs:
    result={'method':'deterministic phase-separated routing microexperiment',
            'phases':{p.name:summarize(list(p.glob('expert_distribution_recorder_*.pt'))) for p in phase_dirs}}
else:
    result=summarize(list(root.glob('expert_distribution_recorder_*.pt')))
    result.update(prefill_decode_labels='unavailable',
        limitations='circular-buffer chronology and cross-rank forward-pass alignment unavailable; pooled slots are not globally aligned iterations')
(root/'expert-summary.json').write_text(json.dumps(result,indent=2))
