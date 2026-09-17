"""Record the final <=200 consecutive passes of a five-minute diagnostic."""
import json
import os
import re
from pathlib import Path
import time
import urllib.request

out=Path(os.environ['RESULT_DIR'])
log=out/'benchmark.log'
deadline=time.monotonic()+5400
while time.monotonic()<deadline:
    text=log.read_text(errors='replace') if log.exists() else ''
    if re.search(r'Phase profiling(?: \(profiling\))? started', text):
        break
    time.sleep(1)
else:
    raise RuntimeError('No profiling start marker; recorder not started')

def call(action):
    with urllib.request.urlopen(urllib.request.Request(
        f'http://localhost:8889/{action}_expert_distribution_record',data=b'',method='POST'),timeout=60) as r:
        assert r.status==200

call('start')
start=time.time()
try:
    time.sleep(295)
finally:
    call('stop')
    call('dump')
(out/'recorder.json').write_text(json.dumps(dict(mode='stat',buffer_passes=200,
    elapsed_seconds=time.time()-start,phase='separate diagnostic after normal warmup',
    deepep_mode=os.environ.get('AGENTX_DIAGNOSTIC_DEEPEP_MODE','not-applicable'),
    timed_screening=False,phase_labels='unavailable in original stat output',
    window='last up to 200 consecutive recorded passes; not first 200'),indent=2))
