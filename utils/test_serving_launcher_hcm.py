import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "runners/launch_hardware-hcm.sh"
ROUTER_IMAGE = "vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1"


def run_launcher(tmp_path: Path, extra_env: dict[str, str]) -> tuple[list[list[str]], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log, envlog = tmp_path / "calls", tmp_path / "srun_env"
    stubs = {
        "hostname": "echo hgx-h200-02", "squeue": "echo 4711", "salloc": "", "scancel": "",
        "srun": f'echo "${{DYN_DISCOVERY_BACKEND:-none}} ${{DYN_FILE_KV:-none}} ${{DYN_SYSTEM_PORT:-none}} ${{AIPERF_GPU_TELEMETRY_METRICS_CSV:-none}} $(case "$*" in *serve_router.sh*) echo router;; *) echo other;; esac) ${{ENGINE_PORT:-none}}" >> {envlog}\necho "${{ENGINE_PORTS:-none}}" >> {tmp_path}/ports',
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(
            f"#!/bin/bash\nrec=$(printf '%s\\x1f' {name} \"$@\")\nprintf '%s\\x1e' \"$rec\" >> {log}\n{body}\n"
        )
        (bin_dir / name).chmod(0o755)
    workspace = tmp_path / "ws"
    (workspace / "results/.dynkv").mkdir(parents=True)
    (workspace / "results/.dynkv/stale").write_text("ghost worker\n")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "USER": "runner",
        "GPU_COUNT": "8", "IMAGE": "vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0",
        "GITHUB_WORKSPACE": str(workspace), "RUNNER_NAME": "hardware-hcm-8x_02", "FRAMEWORK": "sglang",
        "SPEC_DECODING": "none", "SCENARIO_SUBDIR": "agentic/", "EXP_NAME": "glm53-dyn_c8", "PRECISION": "w4afp8",
        "HF_HUB_CACHE": "/mnt/hf_hub_cache/", "IX_SQUASH_DIR": str(tmp_path / "containers"),
        **extra_env,
    }
    out = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    calls = [record.split("\x1f")[:-1] for record in log.read_text().split("\x1e")[:-1]]
    return calls, envlog.read_text()


def container_steps(calls):
    return [call for call in calls if call[0] == "srun" and any(a.startswith("--container-image=") for a in call)]


@pytest.mark.parametrize("router", [True, False], ids=["dynamo", "engine-only"])
def test_serving_path(tmp_path, router):
    extra = {"SERVE_CMD": "python3 -m dynamo.sglang --model-path=PhalaCloud/GLM-5.3-W4AFP8"}
    if router:
        extra |= {"ROUTER_IMAGE": ROUTER_IMAGE, "ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo"}
    calls, envlog = run_launcher(tmp_path, extra)
    steps = container_steps(calls)
    # The router step runs in a background subshell, so call order is not fixed: match by script.
    bench = [s for s in steps if s[-1] == "benchmarks/single_node/agentic/serve_generic.sh"]
    assert len(bench) == 1 and bench[0][-2] == "bash"
    router_steps = [s for s in steps if s[-1] == "benchmarks/single_node/agentic/serve_router.sh"]
    imports = [c for c in calls if c[0] == "srun" and "enroot import" in c[-1]]
    if router:
        assert len(router_steps) == 1
        assert "--overlap" in router_steps[0]
        squash = str(tmp_path / "containers" / (ROUTER_IMAGE.replace("/", "_").replace(":", "_") + ".sqsh"))
        assert f"--container-image={squash}" in router_steps[0]
        assert any("/mnt/hf_hub_cache/:/mnt/hf_hub_cache/" in a for a in router_steps[0])
        assert len(imports) == 3
        assert "file /workspace/results/.dynkv" in envlog
        assert not (tmp_path / "ws/results/.dynkv").exists()
        assert (tmp_path / "ws/results/.router_exited").read_text().startswith("exit ")
    else:
        assert router_steps == []
        assert len(imports) == 2
        assert "file" not in envlog.split()


def test_dyn_system_port_fits_i16(tmp_path):
    """Dynamo parses DYN_SYSTEM_PORT as i16: an ephemeral port like 50997 crashes it."""
    extra = {"SERVE_CMD": "python3 -m dynamo.sglang --model-path=m", "ROUTER_IMAGE": ROUTER_IMAGE,
             "ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo"}
    _, envlog = run_launcher(tmp_path, extra)
    ports = {int(line.split()[2]) for line in envlog.splitlines() if line.split()[0] == "file" and line.split()[2] != "none"}
    assert ports and all(0 < p <= 32767 for p in ports)


def test_aiperf_telemetry_csv_is_a_workspace_path(tmp_path):
    """aiperf runs in the bench container, which only sees the workspace mount, not the DCGM sidecar's /etc."""
    csv = tmp_path / "benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("DCGM_FI_DEV_GPU_UTIL, gauge, util\n")
    _, envlog = run_launcher(tmp_path, {"SERVE_CMD": "python3 -m dynamo.sglang --model-path=m"})
    paths = {line.split()[3] for line in envlog.splitlines()} - {"none"}
    assert paths, envlog
    assert all(p.startswith("/workspace/") and p.endswith("serve_generic.gpu_metrics.csv") for p in paths), paths


def test_dyn_system_port_not_passed_to_router_step(tmp_path):
    """The router container must not inherit the engine's DYN_SYSTEM_PORT; the bench step still gets it."""
    extra = {"SERVE_CMD": "python3 -m dynamo.sglang --model-path=m", "ROUTER_IMAGE": ROUTER_IMAGE,
             "ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo"}
    _, envlog = run_launcher(tmp_path, extra)
    rows = [line.split() for line in envlog.splitlines() if line.split()[0] == "file"]
    router = [r for r in rows if r[4] == "router"]
    bench = [r for r in rows if r[4] == "other" and r[2] != "none"]
    assert router and all(r[2] == "none" for r in router), rows
    assert bench, rows
    assert all(r[5] == "none" for r in rows), rows   # Dynamo profile never gets ENGINE_PORT


def test_generic_router_replicas_get_distinct_engine_ports(tmp_path):
    extra = {"SERVE_CMD": "vllm serve m", "ROUTER_IMAGE": "vllm/vllm-openai:v0.25.1",
             "ROUTER_CMD": "vllm-router --worker-urls {{engine_urls}}", "ROUTER_PROFILE": "generic",
             "REPLICAS": "2"}
    _, envlog = run_launcher(tmp_path, extra)
    seen = {line for line in (tmp_path / "ports").read_text().splitlines() if line != "none"}
    assert len(seen) == 1
    ports = seen.pop().split()
    assert len(ports) == 2 and len(set(ports)) == 2
    assert {line.split()[-1] for line in envlog.splitlines() if line.split()[-1] != "none"} == {ports[0]}


def test_generic_router_gets_engine_port_and_no_dynamo_env(tmp_path):
    extra = {"SERVE_CMD": "python3 -m sglang.launch_server --model-path=m", "ROUTER_IMAGE": ROUTER_IMAGE,
             "ROUTER_CMD": "python3 -m sglang_router.launch_router", "ROUTER_PROFILE": "generic"}
    calls, envlog = run_launcher(tmp_path, extra)
    rows = [line.split() for line in envlog.splitlines()]
    assert rows and all(r[0] == "none" and r[1] == "none" and r[2] == "none" for r in rows), rows
    router = [r for r in rows if r[4] == "router"]
    other = [r for r in rows if r[4] == "other"]
    assert len(router) == 1 and router[0][5] != "none", rows
    assert other[-1][5] == router[0][5], rows       # the bench step sees the same ENGINE_PORT
    assert 0 < int(router[0][5]) < 65536
    assert (tmp_path / "ws/results/.router_exited").read_text().startswith("exit ")
