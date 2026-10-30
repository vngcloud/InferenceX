import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLAN = textwrap.dedent("""\
    session_id,i,parent,shared,isl,rendered_isl,lcp_parent,osl,delay_ms,timeout
    room-1,0,,0,12000,12000,0,10,,5.0
    room-1,1,0,11990,12100,12100,11990,10,800,5.0
    room-1,2,1,12090,12150,12150,12090,10,800,10.0
    room-2,0,,0,3000,3000,0,10,,5.0
""")
FAKE_HUB = textwrap.dedent('''\
    import os
    from . import errors
    def hf_hub_download(repo_id, filename, revision=None, repo_type=None, token=None):
        with open(os.environ["FAKE_HF_LOG"], "a") as f:
            f.write(filename + "\\n")
        return os.environ["FAKE_HF_FILE"]
''')
FAKE_ERRORS = textwrap.dedent('''\
    class HfHubHTTPError(Exception):
        response = None
    class RepositoryNotFoundError(HfHubHTTPError): pass
''')


def record(room, turn, ttft_ms, phase="profiling"):
    return {"metadata": {"conversation_id": room, "turn_index": turn, "benchmark_phase": phase},
            "metrics": {"time_to_first_token": {"value": ttft_ms, "unit": "ms"}}}


@pytest.fixture()
def world(tmp_path):
    """A mini repo with the script and a lock, a fake huggingface_hub serving the plan, and one run artifact."""
    repo = tmp_path / "repo"
    (repo / "utils/serving").mkdir(parents=True)
    (repo / "scenarios/datasets").mkdir(parents=True)
    for name in ("livekit_ttft.py", "fetch_custom_dataset.py"):
        shutil.copy(REPO / "utils/serving" / name, repo / "utils/serving/")
    plan = tmp_path / "plan.csv"
    plan.write_text(PLAN)
    sha = hashlib.sha256(PLAN.encode()).hexdigest()
    (repo / "scenarios/datasets/lock.yaml").write_text(
        f"lk-v1-plan: {{repo: Green-Lab/x, revision: {'a' * 40}, file: plan.v1.csv, sha256: {sha}}}\n")
    fake = tmp_path / "fake/huggingface_hub"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_HUB)
    (fake / "errors.py").write_text(FAKE_ERRORS)

    run = tmp_path / "artifacts/agentic_x_c96/aiperf_artifacts"
    run.mkdir(parents=True)
    (run.parent / "dataset.txt").write_text("lk-v1 Green-Lab/x@aaa file=trace.v1.jsonl sha256=0 path=/x\n")
    rows = [record("room-1", 0, 900), record("room-1", 1, 100), record("room-1", 2, 300),
            record("room-2", 0, 1500), record("room-2", 0, 50, phase="warmup")]
    (run / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "FAKE_HF_LOG": str(tmp_path / "calls"),
           "FAKE_HF_FILE": str(plan)}
    env.pop("GREENLAB_HF_TOKEN", None)
    env.pop("RESULT_DIR", None)
    return {"repo": repo, "env": env, "plan": plan, "artifacts": tmp_path / "artifacts",
            "calls": tmp_path / "calls"}


def run(world, *args):
    return subprocess.run([sys.executable, str(world["repo"] / "utils/serving/livekit_ttft.py"), *map(str, args)],
                          env=world["env"], capture_output=True, text=True, check=False)


def test_artifact_dir_uses_the_locked_plan_of_its_dataset(world):
    out = run(world, world["artifacts"])
    assert out.returncode == 0, out.stderr
    assert world["calls"].read_text() == "plan.v1.csv\n"
    lines = out.stdout.splitlines()
    # 4 profiling requests (the warmup one is dropped); first turns 0.9 s and 1.5 s, later 0.1 s and 0.3 s.
    assert lines[1] == "all: n=4 p95=1.50s  first-turn share=50.0%  first-turn share of >p95: 0%"
    assert lines[2].startswith("  first: n=    2 p50=1.50 ") and ">1s=50%" in lines[2]
    assert "median uncached=12000" in lines[2]
    assert lines[3].startswith("  later: n=    2 p50=0.30 ") and "median uncached=110" in lines[3]
    # The 10 s-timeout request leaves the "5s turns" view.
    assert lines[4].startswith("5s turns: n=3 ")
    assert "  first turns uncached 0-5000: n=1 p50=1.50" in out.stdout
    assert "  first turns uncached 11000-20000: n=1 p50=0.90" in out.stdout


def test_local_plan_skips_the_fetch(world):
    export = world["artifacts"] / "agentic_x_c96/aiperf_artifacts/profile_export.jsonl"
    out = run(world, export, "--plan", world["plan"])
    assert out.returncode == 0, out.stderr
    assert not world["calls"].exists()
    assert "all: n=4 p95=1.50s" in out.stdout


def test_dataset_without_a_plan_entry_fails_with_a_hint(world):
    (world["artifacts"] / "agentic_x_c96/dataset.txt").write_text("tau2-cs Green-Lab/y@b file=t sha256=0 path=/y\n")
    out = run(world, world["artifacts"])
    assert out.returncode == 1
    assert "no tau2-cs-plan in scenarios/datasets/lock.yaml; pass --plan" in out.stderr


def test_warm_up_rows_are_reported_apart(world):
    # A -warm plan: room-1's dial warm-up is turn 0, so its first turn is turn 1.
    world["plan"].write_text(textwrap.dedent("""\
        session_id,i,parent,shared,isl,rendered_isl,lcp_parent,osl,delay_ms,timeout,warm
        room-1,0,,0,11936,11936,,1,,5.0,1
        room-1,1,,0,12000,12000,0,10,3000,5.0,
        room-1,2,1,11990,12100,12100,11990,10,800,5.0,
        room-2,0,,0,3000,3000,0,10,,5.0,
    """))
    export = world["artifacts"] / "agentic_x_c96/aiperf_artifacts/profile_export.jsonl"
    rows = [record("room-1", 0, 700), record("room-1", 1, 120), record("room-1", 2, 100), record("room-2", 0, 1500)]
    export.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = run(world, export, "--plan", world["plan"])
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[1].startswith("all: n=3 ")
    assert lines[2].startswith("  first: n=    2 p50=1.50 ")
    assert lines[-1] == ("warm-ups (excluded above): n=1 p50=0.70 p90=0.70 p95=0.70 p99=0.70 max=0.70  "
                         "done <=0.5s=0.0% <=1s=100.0% <=1.5s=100.0% <=2s=100.0%")


def test_open_loop_window_replayed_calls_and_ttfs(world):
    # openloop.json next to aiperf_artifacts: calls opened in [60, 600) s after the first request are scored.
    run_dir = world["artifacts"] / "agentic_x_c96"
    (run_dir / "openloop.json").write_text(json.dumps({"score_window_s": [60.0, 600.0]}))

    def timed(room, turn, ttft_ms, start_s, gaps_ms=()):
        r = record(room, turn, ttft_ms)
        r["metadata"]["request_start_ns"] = int(start_s * 1e9)
        r["metrics"]["inter_chunk_latency"] = {"value": list(gaps_ms), "unit": "ms"}
        return r

    rows = [timed("room-2", 0, 2000, 0),                                    # ramp: left out
            timed("room-1~2", 0, 900, 100), timed("room-1~2", 1, 200, 700, [10] * 11),   # opened at 100 s: kept
            timed("room-2~3", 0, 400, 650)]                                 # tail: left out
    (run_dir / "aiperf_artifacts/profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = run(world, world["artifacts"], "--plan", world["plan"])
    assert out.returncode == 0, out.stderr
    assert "open loop: calls opened in [60, 600) s" in out.stdout
    assert "all: n=2 " in out.stdout
    # the replayed room-1~2 reads room-1's plan rows; its turn 1 TTFS = 0.2 s + 11 x 10 ms
    assert "  TTFS (12 tokens): n=1 p50=0.31 " in out.stdout
