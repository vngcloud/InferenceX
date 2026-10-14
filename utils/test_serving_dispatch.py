import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DISPATCH = REPO / "utils/serving/dispatch.sh"


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
