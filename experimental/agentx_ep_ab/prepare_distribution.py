"""Diagnostic-only metadata instrumentation of the immutable image's recorder."""
import hashlib
import inspect
import json
from pathlib import Path
import sglang

path = Path(sglang.__file__).parent/'srt/eplb/expert_distribution.py'
source = path.read_text()
old_start = '        outputs = {}\n        with self._current_forward_pass_id.with_value(forward_pass_id):'
new_start = '''        outputs = {}
        self._diagnostic_batch = dict(
            forward_mode=forward_batch.forward_mode.name,
            local_input_tokens=int(forward_batch.input_ids.numel()),
        )
        with self._current_forward_pass_id.with_value(forward_pass_id):'''
old_end = '            single_pass_data = gatherer.collect()\n            self._accumulator.append('
new_end = '            single_pass_data = gatherer.collect()\n            single_pass_data["diagnostic_batch"] = self._diagnostic_batch\n            self._accumulator.append('
assert source.count(old_start) == source.count(old_end) == 1, 'Unsupported recorder source'
patched = source.replace(old_start,new_start).replace(old_end,new_end)
path.write_text(patched)
Path('/results/recorder-instrumentation.json').write_text(json.dumps({
    'original_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'instrumented_sha256':hashlib.sha256(patched.encode()).hexdigest(),
    'change':'Attach forward mode and local input tensor length; count algorithm unchanged',
    'performance_comparable':False},indent=2))
