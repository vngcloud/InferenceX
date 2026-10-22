import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "runners/launch_h200-greennode-slurm-1x.sh"
STUBS = {
    "hostname": "echo hoanq3-h200-1x-han-3-1",
    "id": "echo 4242",
    "squeue": "echo 4711",
    "salloc": "",
    "scancel": "",
    "srun": "cat > /dev/null",
}


def run_launcher(tmp_path: Path, extra_env: dict[str, str], stubs: dict[str, str] = STUBS) -> list[list[str]]:
    """Run the launcher with Slurm/host binaries stubbed; return each stub call's argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    for name, body in stubs.items():
        # One write per call (fields \x1f, records \x1e): the dcgm srun runs in the background.
        (bin_dir / name).write_text(
            f"#!/bin/bash\nrec=$(printf '%s\\x1f' {name} \"$@\")\nprintf '%s\\x1e' \"$rec\" >> {log}\n{body}\n"
        )
        (bin_dir / name).chmod(0o755)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "README").write_text("x\n")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "USER": "runner",
        "GPU_COUNT": "1", "IMAGE": "lmsysorg/sglang:v0.5.9", "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_NAME": "h200-greennode-slurm-1x_01", "FRAMEWORK": "sglang", "SPEC_DECODING": "none",
        "SCENARIO_SUBDIR": "agentic/", "EXP_NAME": "qwen38_tp1_conc8", "PRECISION": "bf16",
        **extra_env,
    }
    out = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return [record.split("\x1f")[:-1] for record in log.read_text().split("\x1e")[:-1]]


@pytest.mark.parametrize("extra_env, nodelist, mounts, bench_script", [
    ({}, "hoanq3-h200-1x-han-3-1",
     "-m /data/inferencex-4242/ws-4711:/workspace -m /data/inferencex-4242/hf:/mnt/hf_hub_cache",
     "benchmarks/single_node/agentic/qwen38_bf16_h200_sglang.sh"),
    ({"IX_SLURM_NODELIST": "node-x", "IX_SCRATCH_ROOT": "/scratch/ix", "IX_HF_CACHE_HOST": "/data/hf-cache"},
     "node-x",
     "-m /scratch/ix-4242/ws-4711:/workspace -m /data/hf-cache:/mnt/hf_hub_cache",
     "benchmarks/single_node/agentic/qwen38_bf16_h200_sglang.sh"),
    ({"SERVE_CMD": "python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B"},
     "hoanq3-h200-1x-han-3-1",
     "-m /data/inferencex-4242/ws-4711:/workspace -m /data/inferencex-4242/hf:/mnt/hf_hub_cache",
     "benchmarks/single_node/agentic/serve_generic.sh"),
], ids=["default", "ix-knobs", "serve-cmd"])
def test_launcher_allocates_mounts_and_runs_bench_script(tmp_path, extra_env, nodelist, mounts, bench_script):
    calls = run_launcher(tmp_path, extra_env)
    sallocs = [call for call in calls if call[0] == "salloc"]
    assert len(sallocs) == 1
    assert f"--nodelist={nodelist}" in sallocs[0]
    bench = [call for call in calls if call[0] == "srun" and "--cpus-per-task=13" in call]
    assert len(bench) == 1
    command = bench[0][-1]
    assert f" {mounts} " in command
    assert command.endswith(f"&& bash {bench_script}'")


def test_greenlab_token_from_runner_env_reaches_container_env(tmp_path):
    """Runner .env -> job env -> launcher `export -p` -> .ix_env (sourced in the container)."""
    stdin_log = tmp_path / "srun_stdin"
    run_launcher(tmp_path, {"SERVE_CMD": "vllm serve m", "GREENLAB_HF_TOKEN": "hf_dummy"},
                 stubs={**STUBS, "srun": f"cat >> {stdin_log}"})
    assert b'declare -x GREENLAB_HF_TOKEN="hf_dummy"' in stdin_log.read_bytes()


def test_dcgm_fieldset_is_readable_by_aiperf_in_the_bench_container(tmp_path):
    """aiperf reads the fieldset in the bench container, which mounts the workspace, not the DCGM path."""
    csv = "benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv"
    (tmp_path / csv).parent.mkdir(parents=True)
    (tmp_path / csv).write_text("DCGM_FI_DEV_GPU_UTIL, gauge, util\n")
    stdin_log = tmp_path / "srun_stdin"
    calls = run_launcher(tmp_path, {"SERVE_CMD": "vllm serve m"}, stubs={**STUBS, "srun": f"cat >> {stdin_log}"})
    assert f'declare -x AIPERF_GPU_TELEMETRY_METRICS_CSV="/workspace/{csv}"'.encode() in stdin_log.read_bytes()
    dcgm = [call for call in calls if call[0] == "srun" and "--mem=4G" in call]
    assert len(dcgm) == 1
    assert f"/ws-4711/{csv}:/etc/dcgm-exporter/custom.csv " in dcgm[0][-1]
    assert dcgm[0][-1].endswith("-f /etc/dcgm-exporter/custom.csv")
