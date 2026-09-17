"""Run a deterministic, phase-separated expert-routing microexperiment."""
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(os.environ["RESULT_DIR"])
BACKEND = "http://localhost:8889"
FRONTEND = "http://localhost:8888/v1/chat/completions"


def control(action):
    req = urllib.request.Request(
        f"{BACKEND}/{action}_expert_distribution_record", data=b"", method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        assert response.status == 200


def request(prompt, max_tokens):
    body = json.dumps({
        "model": "zai-org/GLM-5.2-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        FRONTEND, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=14400) as response:
        result = json.load(response)
    usage = result.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": time.time() - start,
    }


def run_phase(name, requests):
    before = set(ROOT.glob("expert_distribution_recorder_*.pt"))
    control("start")
    results = []
    try:
        for prompt, max_tokens in requests:
            results.append(request(prompt, max_tokens))
    finally:
        control("stop")
        control("dump")
    created = set(ROOT.glob("expert_distribution_recorder_*.pt")) - before
    assert created, f"No recorder dumps created for {name}"
    phase_dir = ROOT / name
    phase_dir.mkdir()
    for path in created:
        shutil.move(path, phase_dir / path.name)
    (phase_dir / "requests.json").write_text(json.dumps(results, indent=2))


base = (
    "You are analyzing a distributed inference engine. Explain the following code and preserve "
    "all identifiers, invariants, tensor shapes, and execution-order constraints.\n"
    "def route(tokens, experts, capacity): return stable_dispatch(tokens, experts, capacity)\n"
)
# Exact prompt bytes and request order are identical for EP1 and EP8. Prefill
# uses long inputs with one output token; decode uses fixed inputs and 256 output
# tokens. This is a routing microexperiment, not an AgentX performance score.
prefill = [(base * repeats, 1) for repeats in (64, 128, 256, 512)]
decode = [(base * 64, 256) for _ in range(4)]
manifest = {
    "kind": "deterministic phase-separated routing microexperiment",
    "prefill_requests": len(prefill),
    "decode_requests": len(decode),
    "temperature": 0,
    "sequential": True,
}
(ROOT / "matched-routing-manifest.json").write_text(json.dumps(manifest, indent=2))
run_phase("prefill", prefill)
run_phase("decode", decode)
