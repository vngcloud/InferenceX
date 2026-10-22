import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_dummySECRET"
FAKE_HUB = textwrap.dedent('''\
    import json, os
    from . import errors
    def hf_hub_download(repo_id, filename, revision=None, repo_type=None, token=None):
        with open(os.environ["FAKE_HF_LOG"], "a") as f:
            f.write(json.dumps({"repo": repo_id, "file": filename, "revision": revision,
                                "repo_type": repo_type, "token": token}) + "\\n")
        if os.environ.get("FAKE_HF_RAISE") == "notfound":
            raise errors.RepositoryNotFoundError("404 for " + repo_id)
        return os.environ["FAKE_HF_FILE"]
''')
FAKE_ERRORS = textwrap.dedent('''\
    class HfHubHTTPError(Exception):
        response = None
    class RepositoryNotFoundError(HfHubHTTPError): pass
    class GatedRepoError(RepositoryNotFoundError): pass
''')
CONTENT = b'{"session_id": "s1", "messages": []}\n'


@pytest.fixture()
def world(tmp_path):
    """A mini repo (script + lock) and a fake huggingface_hub on PYTHONPATH."""
    import hashlib
    sha = hashlib.sha256(CONTENT).hexdigest()
    repo = tmp_path / "repo"
    (repo / "utils/serving").mkdir(parents=True)
    (repo / "scenarios/datasets").mkdir(parents=True)
    shutil.copy(REPO / "utils/serving/fetch_custom_dataset.py", repo / "utils/serving/")
    (repo / "scenarios/datasets/lock.yaml").write_text(
        f"tau2-cs: {{repo: Green-Lab/x, revision: {'a' * 40}, file: trace.jsonl, sha256: {sha}}}\n")
    fake = tmp_path / "fake/huggingface_hub"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_HUB)
    (fake / "errors.py").write_text(FAKE_ERRORS)
    data = tmp_path / "trace.jsonl"
    data.write_bytes(CONTENT)
    results = tmp_path / "results"
    results.mkdir()
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "FAKE_HF_LOG": str(tmp_path / "calls"),
           "FAKE_HF_FILE": str(data), "RESULT_DIR": str(results), "RUNNER_NAME": "r1"}
    env.pop("GREENLAB_HF_TOKEN", None)
    return {"repo": repo, "env": env, "data": data, "results": results, "calls": tmp_path / "calls", "sha": sha}


def fetch(world, *args, **env):
    return subprocess.run([sys.executable, str(world["repo"] / "utils/serving/fetch_custom_dataset.py"), *args],
                          env={**world["env"], **env}, capture_output=True, text=True)


def test_token_passed_path_printed_provenance_written(world):
    out = fetch(world, "tau2-cs", GREENLAB_HF_TOKEN=SECRET)
    assert out.returncode == 0, out.stderr
    assert out.stdout == f"{world['data']}\n"
    call = world["calls"].read_text()
    assert f'"token": "{SECRET}"' in call and f'"revision": "{"a" * 40}"' in call and '"repo_type": "dataset"' in call
    assert "GREENLAB_HF_TOKEN: set" in out.stderr
    assert SECRET not in out.stdout + out.stderr
    provenance = (world["results"] / "dataset.txt").read_text()
    assert provenance.startswith(f"tau2-cs Green-Lab/x@{'a' * 40} ") and world["sha"] in provenance


def test_no_token_means_anonymous(world):
    out = fetch(world, "tau2-cs")
    assert out.returncode == 0, out.stderr
    assert '"token": false' in world["calls"].read_text()
    assert "GREENLAB_HF_TOKEN: unset" in out.stderr


def test_sha_mismatch_fails_with_both_hashes(world):
    world["data"].write_bytes(CONTENT + b"x")
    out = fetch(world, "tau2-cs")
    assert out.returncode == 1
    assert out.stderr.splitlines()[-1].startswith("ERROR dataset tau2-cs: sha256 mismatch")
    assert world["sha"] in out.stderr and "got " in out.stderr
    assert "Traceback" not in out.stderr


def test_unreadable_repo_names_token_state_not_value(world):
    out = fetch(world, "tau2-cs", GREENLAB_HF_TOKEN=SECRET, FAKE_HF_RAISE="notfound")
    assert out.returncode == 1
    assert out.stderr.splitlines()[-1] == (
        f"ERROR dataset tau2-cs: Green-Lab/x@{'a' * 40} not readable; GREENLAB_HF_TOKEN is set on runner r1")
    assert SECRET not in out.stdout + out.stderr


def test_unknown_name(world):
    out = fetch(world, "nope")
    assert out.returncode == 1
    assert "ERROR dataset nope: not in scenarios/datasets/lock.yaml" in out.stderr


def test_check_flag(world):
    assert fetch(world, "--check", "tau2-cs").returncode == 0
    assert fetch(world, "--check", "semianalysis_cc_traces_weka_062126").returncode == 1
    assert not world["calls"].exists()
    (world["repo"] / "scenarios/datasets/lock.yaml").write_text("{{not yaml\n")
    bad = fetch(world, "--check", "tau2-cs")
    assert bad.returncode == 2
    assert bad.stderr.startswith("ERROR scenarios/datasets/lock.yaml unreadable")


def test_bash_wrapper_never_traces_token(world):
    script = textwrap.dedent(f'''\
        set -e
        export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=180 MODEL=m FRAMEWORK=vllm
        export INFMAX_CONTAINER_WORKSPACE={world["repo"]}
        source {REPO}/benchmarks/benchmark_lib.sh
        AIPERF_PYTHON={sys.executable}
        set -x
        is_custom_dataset tau2-cs
        fetch_custom_dataset tau2-cs
        [[ $- == *x* ]] && echo XTRACE_RESTORED
        echo "FILE=$IX_CUSTOM_DATASET_FILE"
    ''')
    out = subprocess.run(["bash", "-c", script], env={**world["env"], "GREENLAB_HF_TOKEN": SECRET},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert f"FILE={world['data']}" in out.stdout
    assert "XTRACE_RESTORED" in out.stdout
    assert SECRET not in out.stdout + out.stderr
