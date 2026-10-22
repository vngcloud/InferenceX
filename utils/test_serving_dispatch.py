import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from infx.serving.schema import NAME_PATTERN

REPO = Path(__file__).resolve().parents[1]
DISPATCH = REPO / "utils/serving/dispatch.sh"
SWEEP_DISPATCH = REPO / "utils/serving/sweep_dispatch.sh"


def test_dispatch_is_executable_bash():
    assert os.access(DISPATCH, os.X_OK)
    subprocess.run(["bash", "-n", str(DISPATCH)], check=True)


def test_dispatch_stops_on_invalid_yaml_before_calling_gh(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nrouter: {}\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "gh").write_text("#!/bin/sh\necho GH_CALLED >&2\nexit 99\n")
    (fake_bin / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    out = subprocess.run([str(DISPATCH), str(bad), "--ref", "some-branch"], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 2
    assert "ERROR router.image: Field required" in out.stderr
    assert "GH_CALLED" not in out.stderr


def stub(bin_dir: Path, name: str, body: str) -> None:
    (bin_dir / name).write_text("#!/bin/bash\n" + body)
    (bin_dir / name).chmod(0o755)


def test_ref_without_value_is_a_usage_error(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stub(fake_bin, "gh", "echo GH_CALLED >&2\nexit 99\n")
    out = subprocess.run([str(DISPATCH), "serving/qwen38-bf16-tp1.yaml", "--ref"], cwd=REPO,
                         capture_output=True, text=True, env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"})
    assert out.returncode == 2
    assert out.stderr == "usage: utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]\n"


def test_run_lookup_tolerates_local_clock_ahead(tmp_path):
    # GitHub stamps the run 30 s before the local clock's dispatch time; an
    # older run with the same title must not be picked.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stub(fake_bin, "sleep", "exit 0\n")
    stub(fake_bin, "gh", f"""\
if [[ "$1 $2" == "workflow run" ]]; then
    for arg in "$@"; do [[ "$arg" == test-name=* ]] && printf '%s' "${{arg#test-name=}}" > {tmp_path}/title; done
    exit 0
fi
while [[ "$1" != --jq ]]; do shift; done
now=$(date +%s)
jq -n -r --arg t "e2e Test - $(cat {tmp_path}/title)" --argjson now "$now" \
    '[{{databaseId: 99, displayTitle: $t, createdAt: (($now - 600) | todate)}},
      {{databaseId: 111, displayTitle: $t, createdAt: (($now - 30) | todate)}}]' | jq -r "$2"
""")
    out = subprocess.run([str(DISPATCH), "serving/qwen38-bf16-tp1.yaml", "--ref", "some-branch"], cwd=REPO,
                         capture_output=True, text=True, env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"})
    assert out.returncode == 0, out.stderr
    assert out.stdout == "https://github.com/vngcloud/InferenceX/actions/runs/111\n"


def test_example_serving_yaml_is_valid():
    out = subprocess.run(
        ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "pydantic", "--with", "pyyaml",
         "python", "-P", "-m", "infx.serving.to_matrix", "--check", "serving/qwen38-bf16-tp1.yaml"],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("serving qwen38-bf16-tp1 agentx-cc-062126-smoke (180s) c8 ")


def test_sweep_dispatch_is_executable_bash():
    assert os.access(SWEEP_DISPATCH, os.X_OK)
    subprocess.run(["bash", "-n", str(SWEEP_DISPATCH)], check=True)


def stub_sweep_repo(tmp_path: Path) -> Path:
    """A fake repo tree: sweep_dispatch.sh, three scenarios, a dispatch.sh that
    prints the dispatch arguments to stderr and the YAML it was given."""
    repo = tmp_path / "repo"
    (repo / "utils/serving").mkdir(parents=True)
    (repo / "scenarios").mkdir()
    (repo / "serving").mkdir()
    shutil.copy(SWEEP_DISPATCH, repo / "utils/serving/sweep_dispatch.sh")
    (repo / "utils/serving/sweep_dispatch.sh").chmod(0o755)
    for name in ("sweep-isl128k-osl1k-cache90", "sweep-isl8k-osl512-cache0", "sweep-isl8k-osl512-cache50"):
        (repo / f"scenarios/{name}.yaml").write_text("# generated\n")
    (repo / "utils/serving/dispatch.sh").write_text('#!/usr/bin/env bash\necho "ARGS:$*" >&2\ncat "$1"\n')
    (repo / "utils/serving/dispatch.sh").chmod(0o755)
    (repo / "serving/gemma4-sweep-smoke-1xh200.yaml").write_text(
        "name: gemma4-sweep-smoke-1gpu\nscenario: agentx-cc-062126-smoke\nconc: [1, 4]\n")
    return repo


def run_sweep_dispatch(repo: Path, *args: str, yaml_path: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(repo / "utils/serving/sweep_dispatch.sh"),
                           str(yaml_path or repo / "serving/gemma4-sweep-smoke-1xh200.yaml"), *args],
                          check=False, capture_output=True, text=True)


def dispatched_docs(out: subprocess.CompletedProcess) -> list[dict]:
    """Parse the YAML each stub dispatch.sh received, in dispatch order.

    Each "== <scenario>" header line is the name only; the YAML follows.
    """
    return [yaml.safe_load(block.split("\n", 1)[1]) for block in out.stdout.split("== ")[1:]]


def test_sweep_dispatch_rewrites_and_passes_extra_args(tmp_path):
    repo = stub_sweep_repo(tmp_path)
    out = run_sweep_dispatch(repo, "--ref", "some-branch")
    assert out.returncode == 0, out.stderr
    docs = dispatched_docs(out)
    assert [d["scenario"] for d in docs] == ["sweep-isl128k-osl1k-cache90", "sweep-isl8k-osl512-cache0",
                                             "sweep-isl8k-osl512-cache50"]
    for doc in docs:
        assert doc["conc"] == 1
        assert doc["name"] == f"gemma4-sweep-smoke-1gpu-{doc['scenario'].removeprefix('sweep-')}"
    assert out.stderr.count("--ref some-branch") == 3


def test_sweep_dispatch_only_filter(tmp_path):
    repo = stub_sweep_repo(tmp_path)
    out = run_sweep_dispatch(repo, "--only", "sweep-isl8k-*")
    assert out.returncode == 0, out.stderr
    docs = dispatched_docs(out)
    assert [d["scenario"] for d in docs] == ["sweep-isl8k-osl512-cache0", "sweep-isl8k-osl512-cache50"]
    assert all(d["name"].startswith("gemma4-sweep-smoke-1gpu-isl8k-") for d in docs)


def test_sweep_dispatch_refuses_empty_match_and_missing_yaml(tmp_path):
    repo = stub_sweep_repo(tmp_path)
    out = run_sweep_dispatch(repo, "--only", "sweep-isl999k-*")
    assert out.returncode == 2
    assert "no scenarios/sweep-*.yaml match" in out.stderr
    assert out.stdout == ""
    out = run_sweep_dispatch(repo, yaml_path=repo.parent / "nope.yaml")
    assert out.returncode == 2
    assert "ERROR no such serving YAML" in out.stderr


def test_sweep_dispatch_name_suffix_fits_schema_name_pattern():
    # The longest suffixed name must stay a valid serving name (charset and,
    # once NAME_PATTERN grows a length cap, length).
    scenario = "sweep-isl128k-osl1k-cache90"
    name = f"gemma4-sweep-smoke-1gpu-{scenario.removeprefix('sweep-')}"
    assert re.fullmatch(NAME_PATTERN, name)
