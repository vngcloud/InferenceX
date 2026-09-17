#!/usr/bin/env python3
"""Pinned research campaign. No changes to the default benchmark matrix."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGE = 'lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1'
MODEL_REV = 'e42e1aee344f812b6fb73f503bfbe6a227727396'
DATASET_REV = '8fecd2fc56694469f758f0afbbb6335ad3043740'
AIPERF_REV = '754356e9a39acc6cc6afb242d123bb57c3fb6f75'
PAIRS = [('A',24,True,True,'lpm'), ('B',24,False,True,'lpm'),
         ('E',8,True,True,'lpm'), ('F',8,False,True,'lpm'),
         ('C',24,False,False,'lpm'), ('D',24,False,False,'fcfs')]

def matrix():
    rows = []
    for i,(pair,ccu,spec,cache,schedule) in enumerate(PAIRS):
        for ep in ([1,8] if i % 2 == 0 else [8,1]):
            rows.append(dict(id=f'{pair.lower()}_ep{ep}',pair=pair,ccu=ccu,ep=ep,
                             spec=spec,hicache=cache,schedule=schedule))
    return rows

def recipe(row):
    source = (ROOT/'benchmarks/single_node/agentic/glm5.2deep_fp4_h200_sglang.sh').read_text()
    # Transform only the original recipe's explicit experimental controls.
    replacements = {
        'MAX_RUNNING_REQUESTS=$((2 * CONC))':'MAX_RUNNING_REQUESTS=16',
        '  --moe-a2a-backend deepep\n': '  --moe-a2a-backend deepep\n' if row['ep']==8 else '',
        '  --schedule-policy lpm\n':f"  --schedule-policy {row['schedule']}\n",
        'resolve_trace_source\ninstall_agentic_deps': 'python3 /repo/experimental/agentx_ep_ab/prepare_client.py\nexport AIPERF_DIR=/tmp/agentx-aiperf\ninstall_agentic_deps\n"$AIPERF_UV_BIN" pip freeze --python "$AIPERF_PYTHON" > "$RESULT_DIR/client-packages.txt"\nTRACE_SOURCE_FLAG="--public-dataset semianalysis_cc_traces_weka_062126_256k"',
        'source "$(dirname "$0")/../../benchmark_lib.sh"':'source /repo/benchmarks/benchmark_lib.sh',
    }
    for old,new in replacements.items():
        assert source.count(old)==1, old
        source = source.replace(old,new)
    if row.get('diagnostic'):
        source=source.replace('  --enable-metrics\n','  --enable-metrics\n  --expert-distribution-recorder-mode stat\n  --expert-distribution-recorder-buffer-size 200\n')
        source=source.replace('run_agentic_replay_and_write_outputs "$RESULT_DIR"',
            'python3 /repo/experimental/agentx_ep_ab/record.py &\nRECORDER_PID=$!\n'
            'set +e\nrun_agentic_replay_and_write_outputs "$RESULT_DIR"\nREPLAY_RC=$?\n'
            'wait "$RECORDER_PID"\nRECORD_RC=$?\nset -e\n'
            'python3 /repo/experimental/agentx_ep_ab/summarize_experts.py "$RESULT_DIR"\n'
            'test "$REPLAY_RC" = 0 && test "$RECORD_RC" = 0')
    return source

def main():
    if sys.argv[1]=='matrix':
        print(json.dumps(matrix(),indent=2)); return
    row = next(r for r in matrix() if r['id']==sys.argv[2])
    if sys.argv[1]=='recipe':
        print(recipe(row)); return
    raise SystemExit('Expected matrix or recipe ID')

if __name__=='__main__': main()
