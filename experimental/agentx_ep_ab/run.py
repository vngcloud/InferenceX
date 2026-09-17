#!/usr/bin/env python3
"""Fail-closed, host-pinned CI runner for one screening point."""
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from campaign import ROOT, IMAGE, MODEL_REV, DATASET_REV, AIPERF_REV, matrix, recipe

def output(*args):
    return subprocess.check_output(args,text=True).strip()

def run():
    diagnostic=sys.argv[1].startswith('diag_')
    key=sys.argv[1].removeprefix('diag_')
    row = next(r for r in matrix() if r['id']==key).copy()
    if diagnostic:
        assert row['pair']=='B'
        row.update(id=sys.argv[1],diagnostic=True)
    out = ROOT/'campaign-results'/row['id']
    out.mkdir(parents=True,exist_ok=True)
    started = time.time()
    name = f"agentx-ab-{os.environ['GITHUB_RUN_ID']}-{row['id']}"
    lock = open('/tmp/agentx-h200-2-eight-gpus.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert socket.gethostname()=='hoanq3-h200-8x-han-1', 'Wrong physical host'
    assert os.environ.get('RUNNER_NAME')=='h200-greennode_06', 'Wrong CI runner'
    gpu = output('nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader')
    assert len(gpu.splitlines())==8 and all('H200' in line for line in gpu.splitlines())
    assert not output('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'), 'GPUs are occupied'
    assert output('git','-C',str(ROOT/'utils/aiperf'),'rev-parse','HEAD')==AIPERF_REV
    subprocess.run(['docker','image','inspect',IMAGE],stdout=subprocess.DEVNULL,check=True)
    cache = pathlib.Path('/mnt/hf_hub_cache')
    model = cache/f'models--PhalaCloud--GLM-5.2-W4AFP8/snapshots/{MODEL_REV}'
    index = json.loads((model/'model.safetensors.index.json').read_text())
    assert all((model/f).is_file() for f in set(index['weight_map'].values())), 'Incomplete model snapshot'
    model_mounts=['-v',f'{model}:/models/PhalaCloud/GLM-5.2-W4AFP8:ro',
                  '-v',f'{model.parent.parent}/blobs:/models/blobs:ro']
    subprocess.run(['docker','run','--rm','--network','none',*model_mounts,
                    '--entrypoint','python3',IMAGE,'-c',
                    'import json,pathlib; p=pathlib.Path("/models/PhalaCloud/GLM-5.2-W4AFP8"); '
                    'assert json.loads((p/"config.json").read_text())["model_type"]; '
                    'i=json.loads((p/"model.safetensors.index.json").read_text()); '
                    'assert all((p/f).is_file() for f in set(i["weight_map"].values()))'],check=True)
    # Isolated refs prevent cached moving branches from changing corpus/tokenizer.
    isolated = out/'hf-cache'
    for repo,rev in [('datasets--semianalysisai--cc-traces-weka-062126-256k',DATASET_REV),
                     ('models--zai-org--GLM-5.2-FP8','ba978f7d347eaf65d22f1a86833408afdb953541')]:
        assert (cache/repo/'snapshots'/rev).is_dir(), f'Missing pinned snapshot {repo}/{rev}'
        dest = isolated/repo
        (dest/'refs').mkdir(parents=True,exist_ok=True)
        for part in ['blobs','snapshots']:
            link=dest/part
            if not link.is_symlink(): link.symlink_to(cache/repo/part)
        (dest/'refs/main').write_text(rev)
    (out/'recipe.sh').write_text(recipe(row))
    env = dict(PORT='8888',MODEL='zai-org/GLM-5.2-FP8',MODEL_PREFIX='glm5.2deep',
        TP='8',EP_SIZE=str(row['ep']),CONC=str(row['ccu']),DP_ATTENTION='true',
        SPEC_DECODING='mtp' if row['spec'] else 'none',KV_OFFLOADING='dram' if row['hicache'] else 'none',
        KV_OFFLOAD_BACKEND='hicache' if row['hicache'] else '',
        KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}' if row['hicache'] else '',
        TOTAL_CPU_DRAM_GB=str(int(os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')*0.8/1e9)),
        HICACHE_RATIO='2',DURATION='1200',RESULT_DIR='/results',RESULT_FILENAME=row['id'],
        IMAGE=IMAGE,FRAMEWORK='sglang',PRECISION='fp4',RUNNER_TYPE='h200-greennode_06',
        SCENARIO_TYPE='agentic-coding',IS_AGENTIC='1',RUN_EVAL='false',EVAL_ONLY='false',
        INFMAX_CONTAINER_WORKSPACE='/repo',AGENTIC_OUTPUT_DIR='/results',
        HF_HUB_CACHE='/results/hf-cache',AIPERF_EXPERIMENTAL_FAST='0',
        AIPERF_TOKENIZER='/results/hf-cache/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541',
        AIPERF_WARMUP_REQUESTS_PER_LANE='10',AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES='0',
        AIPERF_UV_CACHE_DIR='/mnt/uv-cache',GITHUB_RUN_ID=os.environ['GITHUB_RUN_ID'])
    if diagnostic:
        env.update(DURATION='300',AIPERF_UNSAFE_OVERRIDE='true',
                   SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR='/results',
                   AIPERF_WARMUP_REQUESTS_PER_LANE='1',
                   AGENTX_DIAGNOSTIC_DEEPEP_MODE='normal' if row['ep']==8 else 'not-applicable')
    (out/'provenance.json').write_text(json.dumps(dict(config=row,env=env,gpus=gpu,
        model_revision=MODEL_REV,dataset_revision=DATASET_REV,aiperf_revision=AIPERF_REV,
        commit=output('git','rev-parse','HEAD'),
        historical_difference='max-running-requests fixed at 16; historical recipe used 2*CCU',
        diagnostic_deviation=(
            'EP8 expert-distribution diagnostic forces DeepEP normal because the pinned '
            'SGLang stat recorder does not implement deepep_mode=auto; records router selections '
            'before dispatch, uses one warm-up request per lane, and is excluded from performance comparison'
            if diagnostic and row['ep']==8 else None)),indent=2))
    # Use an existing exporter, or start one owned only by this job.
    exporter = None
    if subprocess.call(['curl','--fail','--silent','--max-time','5','http://localhost:9400/metrics'],stdout=subprocess.DEVNULL):
        exporter=name+'-dcgm'
        subprocess.run(['docker','run','-d','--rm','--name',exporter,'--gpus','all','--network','host','--cap-add','SYS_ADMIN',
                        'nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04'],check=True)
    cmd=['docker','run','--name',name,'--init','--gpus','all','--ipc=host','--network','host','--shm-size=32g',
         '-v',f'{ROOT}:/repo:ro','-v',f'{out}:/results','-v',f'{cache}:{cache}:ro',
         *model_mounts,
         '-v','/mnt/uv-cache:/mnt/uv-cache','-w','/results']
    for k,v in env.items(): cmd+=['-e',f'{k}={v}']
    cmd+=['--entrypoint','bash',IMAGE,'/results/recipe.sh']
    def stop(signum,frame):
        subprocess.run(['docker','stop','--time','20',name],stdout=subprocess.DEVNULL)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    code=1
    try:
        with (out/'container.log').open('w') as log:
            # Preserve the complete container log as an artifact while also
            # streaming SGLang startup and benchmark progress to Actions.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code=process.wait()
    finally:
        # Container has exited: release its owned resources, preserve artifacts.
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL)
        if exporter: subprocess.run(['docker','rm','-f',exporter],stdout=subprocess.DEVNULL)
        subprocess.run(['docker','run','--rm','--network','none','-v',f'{out}:/results',
                        '--entrypoint','chown',IMAGE,'-R',f'{os.getuid()}:{os.getgid()}','/results'],check=True)
        (out/'allocation.json').write_text(json.dumps(dict(seconds=time.time()-started,exit_code=code)))
    raise SystemExit(code)

if __name__=='__main__': run()
