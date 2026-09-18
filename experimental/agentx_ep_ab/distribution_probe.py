"""Bounded, eager, routing-accounting experiment. Fails before interpreting bad counts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

import torch
from transformers import AutoTokenizer
from distribution_metrics import metrics

ROOT=Path(os.environ['RESULT_DIR'])
EP=int(os.environ['EP_SIZE'])
BASE='http://localhost:8889'


def call(endpoint: str, body: dict | None = None):
    req=urllib.request.Request(BASE+'/'+endpoint,
        data=json.dumps(body or {}).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=300) as response:
        data=response.read()
        if endpoint=='generate': return json.loads(data)


def analyze_case(directory: Path, expected_tokens: int) -> dict:
    files=sorted(directory.glob('expert_distribution_recorder_*.pt'))
    assert len(files)==8, f'Expected all eight rank dumps, got {len(files)}'
    payloads=[torch.load(f,map_location='cpu',weights_only=True) for f in files]
    rank_records={}
    for p in payloads:
        records=p['records']
        assert records, 'Empty per-pass recorder'
        rank=records[0]['rank']
        assert rank not in rank_records
        rank_records[rank]=records
    assert set(rank_records)==set(range(8))
    totals={rank:sum((r['global_physical_count'].to(torch.int64) for r in records))
            for rank,records in rank_records.items()}
    if EP==1:
        reference=totals[0]
        assert all(torch.equal(reference,v) for v in totals.values()), 'EP1 replica counts differ; no automatic normalization'
        counts=reference
    else:
        counts=sum(totals.values())
    active=counts.sum(-1)>0
    observed=counts.sum(-1)[active]
    assert observed.numel()>0 and torch.all(observed==expected_tokens*8), (
        f'Accounting failed: expected {expected_tokens*8} assignments/layer, observed {observed.tolist()}')
    # Do not infer placement from sorted expert IDs. Require the actual map.
    mapping=payloads[0]['last_physical_to_logical_map'].cpu()
    assert all(torch.equal(mapping,p['last_physical_to_logical_map'].cpu()) for p in payloads)
    assert mapping.ndim==2 and mapping.shape==counts.shape
    assert all(sorted(row.tolist())==list(range(256)) for row in mapping), 'Redundant or invalid expert mapping'
    layer_metrics=[]
    for layer in torch.where(active)[0].tolist():
        owners=[0]*256
        logical=[0]*256
        for physical,expert in enumerate(mapping[layer].tolist()):
            owners[expert]=physical//32
            logical[expert]=int(counts[layer,physical])
        layer_metrics.append({'layer':layer,**metrics(logical,owners)})
    phase_totals={}
    for mode in sorted({r['diagnostic_batch']['forward_mode'] for records in rank_records.values() for r in records}):
        by_rank={rank:sum((r['global_physical_count'].to(torch.int64) for r in records
                          if r['diagnostic_batch']['forward_mode']==mode),torch.zeros_like(counts))
                 for rank,records in rank_records.items()}
        if EP==1:
            assert all(torch.equal(by_rank[0],v) for v in by_rank.values()), 'Phase replica mismatch'
            phase_counts=by_rank[0]
        else: phase_counts=sum(by_rank.values())
        phase_layers=[]
        for layer in torch.where(phase_counts.sum(-1)>0)[0].tolist():
            logical=[0]*256;owners=[0]*256
            for physical,expert in enumerate(mapping[layer].tolist()):
                owners[expert]=physical//32
                logical[expert]=int(phase_counts[layer,physical])
            phase_layers.append({'layer':layer,**metrics(logical,owners)})
        phase_totals[mode]=phase_layers
    return {'accounting':'passed','ep':EP,'expected_logical_tokens':expected_tokens,
            'summary_time_scope':'case and forward-mode totals; not per-GEMM M',
            'count_domain':'router assignments, case totals; not kernel padded rows',
            'rank_load_kind':'actual fixed ownership' if EP==8 else 'projected contiguous EP8 ownership',
            'layers':layer_metrics,
            'by_forward_mode':phase_totals,
            'per_rank_passes':{rank:len(rows) for rank,rows in rank_records.items()},
            'per_pass_metrics':'raw per-pass records retained; global pass alignment not assumed'}


def run_case(name: str, length: int, batch: int, output: int, ids: list[int]):
    case=ROOT/name;case.mkdir()
    prompts=[(ids*((length+len(ids)-1)//len(ids)))[:length] for _ in range(batch)]
    before=set(ROOT.glob('expert_distribution_recorder_*.pt'))
    call('start_expert_distribution_record')
    try:
        response=call('generate',{'input_ids':prompts,'sampling_params':{
            'temperature':0,'max_new_tokens':output,'ignore_eos':True},'stream':False})
    finally:
        call('stop_expert_distribution_record');call('dump_expert_distribution_record')
    if isinstance(response,dict): response=[response]
    assert len(response)==batch
    usage=[r['meta_info'] for r in response]
    assert all(r['prompt_tokens']==length and r['completion_tokens']==output for r in usage)
    for f in set(ROOT.glob('expert_distribution_recorder_*.pt'))-before:shutil.move(f,case/f.name)
    manifest={'name':name,'batch':batch,'input_tokens_per_request':length,'output_tokens_per_request':output,
              'input_ids_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              'response_text_sha256':[hashlib.sha256(r.get('text','').encode()).hexdigest() for r in response],
              'usage':usage,'expected_routed_tokens':batch*(length+output-1)}
    (case/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=analyze_case(case,manifest['expected_routed_tokens'])
    (case/'metrics.json').write_text(json.dumps(result,indent=2))


tokenizer=AutoTokenizer.from_pretrained('/models/PhalaCloud/GLM-5.2-W4AFP8',trust_remote_code=True)
ids=tokenizer.encode('def dispatch(tokens, experts): return sorted(tokens)\nExplain tensor shapes, routing and communication.\n',add_special_tokens=False)
started=time.monotonic()
try:
    for length in (8,64,512):run_case(f'calibration-{length}',length,1,1,ids)
    # Only expand after every small-token accounting check passes.
    for length in (512,8192,32768):
        for batch in (1,8):
            assert time.monotonic()-started<900,'Diagnostic budget exhausted'
            run_case(f'prefill-{length}-b{batch}',length,batch,1,ids)
    for length in (512,8192):
        for batch in (1,8,16):
            assert time.monotonic()-started<900,'Diagnostic budget exhausted'
            run_case(f'decode-{length}-b{batch}',length,batch,33,ids)
except Exception as exc:
    (ROOT/'distribution-failure.json').write_text(json.dumps({'error':str(exc),'interpretation_allowed':False}))
    raise
