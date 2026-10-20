import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROUTER = REPO / "benchmarks/single_node/agentic/serve_router.sh"


def test_serve_router_exports_env_appends_port_and_records_command(tmp_path):
    cmd = "python3 -c 'import os,sys; print(os.environ[\"FOO\"], *sys.argv[1:])' --router-mode=kv"
    out = subprocess.run(["bash", str(ROUTER)], cwd=REPO, capture_output=True, text=True, env={
        "PATH": "/usr/bin:/bin", "ROUTER_CMD": cmd, "ROUTER_PROFILE": "dynamo", "ROUTER_ENV": "FOO=bar\n", "PORT": "9999",
        "RESULT_DIR": str(tmp_path), "DYN_DISCOVERY_BACKEND": "file", "DYN_FILE_KV": str(tmp_path / ".dynkv"),
        # benchmark_lib.sh's agentic guard fires on the script's */agentic/* path; the
        # launcher's job env (benchmark-tmpl.yml) always carries KV_OFFLOADING.
        "KV_OFFLOADING": "none",
    })
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "bar --router-mode=kv --http-host 0.0.0.0 --http-port 9999"
    assert "--http-port 9999" in (tmp_path / "router_command.txt").read_text()


@pytest.fixture
def frontend():
    models = {"data": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(models).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], models
    server.shutdown()


def wait(port: int, tmp_path: Path, pid_expr: str) -> subprocess.CompletedProcess:
    body = f"""
        source benchmarks/benchmark_lib.sh
        {pid_expr}
        wait_for_frontend_model --port {port} --model zai-org/GLM-5.3 --server-log {tmp_path}/server.log \\
            --server-pid "$pid" --router-exited {tmp_path}/router.exited --sleep-interval 0.2
    """
    (tmp_path / "server.log").write_text("engine log line\n")
    return subprocess.run(["bash", "-c", body], cwd=REPO, capture_output=True, text=True, timeout=60)


def test_ready_when_model_listed(frontend, tmp_path):
    port, models = frontend
    models["data"] = [{"id": "zai-org/GLM-5.3"}]
    out = wait(port, tmp_path, "sleep 30 & pid=$!")
    assert out.returncode == 0, out.stderr


def test_fails_fast_when_engine_exits(frontend, tmp_path):
    port, _ = frontend
    out = wait(port, tmp_path, "true & pid=$!; wait $pid")
    assert out.returncode == 1
    assert "engine exited before the frontend listed zai-org/GLM-5.3" in out.stderr
    assert "engine log line" in out.stderr


def test_fails_fast_when_router_step_exited(frontend, tmp_path):
    port, _ = frontend
    (tmp_path / "router.exited").write_text("exit 1\n")
    out = wait(port, tmp_path, "sleep 30 & pid=$!")
    assert out.returncode == 1
    assert "router step exited (exit 1)" in out.stderr


GENERIC = REPO / "benchmarks/single_node/agentic/serve_generic.sh"
STUB_LIB = r'''
check_env_vars() { :; }
install_agentic_deps() { :; }
is_custom_dataset() { return 1; }
resolve_trace_source() { :; }
write_command() { local f="$1"; shift; printf '%q ' "$@" > "$f"; }
wait_for_server_ready() { echo "WAIT health $*" >> "$RESULT_DIR/calls"; }
wait_for_frontend_model() { echo "WAIT frontend $*" >> "$RESULT_DIR/calls"; }
build_replay_cmd_yaml() { echo "METRICS $AIPERF_SERVER_METRICS_URLS" >> "$RESULT_DIR/calls"; }
run_agentic_replay_and_write_outputs() { :; }
expand_storage_placeholders() {
    local t="$1"
    [[ -n "${STORAGE_MASTER_PORT:-}" ]] && t="${t//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    [[ -n "${STORAGE_META_PORT:-}" ]] && t="${t//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    printf '%s' "$t"
}
'''


def run_generic_router(tmp_path, **extra):
    cmd = ("python3 -c 'import os,sys; print(os.environ[\"FOO\"], *sys.argv[1:])' "
           "--port={{router_port}} --worker-urls={{engine_url}} --tmpl '{{ keep }}'")
    env = {
        "PATH": "/usr/bin:/bin", "ROUTER_CMD": cmd, "ROUTER_ENV": "FOO=w{{engine_port}}\n",
        "ROUTER_PROFILE": "generic", "PORT": "9999", "ENGINE_PORT": "9200",
        "RESULT_DIR": str(tmp_path), "KV_OFFLOADING": "none",
    } | extra
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(["bash", str(ROUTER)], cwd=REPO, capture_output=True, text=True, env=env)


def test_serve_router_generic_expands_placeholders_and_appends_nothing(tmp_path):
    out = run_generic_router(tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "w9200 --port=9999 --worker-urls=http://127.0.0.1:9200 --tmpl {{ keep }}"
    assert "--http-port" not in (tmp_path / "router_command.txt").read_text()
    assert "--port=9999" in (tmp_path / "router_command.txt").read_text()


def test_serve_router_generic_needs_engine_port(tmp_path):
    out = run_generic_router(tmp_path, ENGINE_PORT=None)
    assert out.returncode != 0
    assert "ENGINE_PORT" in out.stdout + out.stderr


@pytest.mark.parametrize("mode", ["dynamo", "generic", "generic-default-metrics", "engine-only"])
def test_serve_generic_branches(tmp_path, mode):
    lib = tmp_path / "benchmarks"
    (lib / "single_node/agentic").mkdir(parents=True)
    (lib / "benchmark_lib.sh").write_text(STUB_LIB)
    script = lib / "single_node/agentic/serve_generic.sh"
    script.write_text(GENERIC.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "PhalaCloud/GLM-5.3-W4AFP8", "CONC": "8", "PORT": "8888",
        "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "sglang", "SERVING_SCENARIO": "s",
        "SERVING_LOADER": "l", "AIPERF_HF_CLI": "true", "SERVED_MODEL_NAME": "zai-org/GLM-5.3",
        "SERVE_CMD": "sleep 30",
    }
    if mode == "dynamo":
        env |= {"ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo", "DYN_SYSTEM_PORT": "9100"}
    elif mode.startswith("generic"):
        env |= {"ROUTER_CMD": "python3 -m sglang_router.launch_router", "ROUTER_PROFILE": "generic", "ENGINE_PORT": "9200"}
        if mode == "generic":
            env["ROUTER_METRICS_PORT"] = "18888"
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    calls = (tmp_path / "results/calls").read_text()
    command = (tmp_path / "results/server_command.txt").read_text()
    if mode == "dynamo":
        assert "WAIT frontend --port 8888 --model zai-org/GLM-5.3" in calls
        assert f"--router-exited {tmp_path}/results/.router_exited" in calls
        assert "METRICS http://localhost:8888/metrics,http://localhost:9100/metrics" in calls
        assert "--port" not in command
    elif mode.startswith("generic"):
        assert "WAIT frontend --port 8888 --model zai-org/GLM-5.3" in calls
        metrics = "18888" if mode == "generic" else "8888"
        assert f"METRICS http://localhost:{metrics}/metrics,http://localhost:9200/metrics" in calls
        assert command.strip().endswith("--host 0.0.0.0 --port 9200")
    else:
        assert "WAIT health --port 8888" in calls
        assert "METRICS http://localhost:8888/metrics" in calls
        assert command.strip().endswith("--host 0.0.0.0 --port 8888")


def test_serve_generic_unsets_hf_offline_for_bench_client(tmp_path):
    lib = tmp_path / "benchmarks"
    (lib / "single_node/agentic").mkdir(parents=True)
    # "ready" only once the engine stub has run, so the engine env is recorded before replay
    (lib / "benchmark_lib.sh").write_text(STUB_LIB.replace(
        'wait_for_server_ready() { echo "WAIT health $*" >> "$RESULT_DIR/calls"; }',
        'wait_for_server_ready() { while [ ! -f "$RESULT_DIR/engine_env" ]; do sleep 0.05; done; }').replace(
        'build_replay_cmd_yaml() { echo "METRICS $AIPERF_SERVER_METRICS_URLS" >> "$RESULT_DIR/calls"; }',
        'build_replay_cmd_yaml() { echo "REPLAY HF_HUB_OFFLINE=${HF_HUB_OFFLINE-unset}" >> "$RESULT_DIR/calls"; }'))
    (lib / "single_node/agentic/serve_generic.sh").write_text(GENERIC.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # engine stub: records the env the server launch saw, then stays up like a real server
    (bin_dir / "engine").write_text(f"#!/bin/sh\necho ENGINE HF_HUB_OFFLINE=${{HF_HUB_OFFLINE-unset}} > {tmp_path}/results/engine_env\nsleep 30\n")
    (bin_dir / "engine").chmod(0o755)
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "zai-org/GLM-5.3", "CONC": "8", "PORT": "8888",
        "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "sglang", "SERVING_SCENARIO": "s",
        "SERVING_LOADER": "l", "AIPERF_HF_CLI": "true", "SERVE_CMD": "engine",
        "HF_HUB_OFFLINE": "1",  # from the serving YAML's env: for the engine
    }
    out = subprocess.run(["bash", str(lib / "single_node/agentic/serve_generic.sh")], capture_output=True,
                         text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "results/engine_env").read_text().strip() == "ENGINE HF_HUB_OFFLINE=1"
    assert "REPLAY HF_HUB_OFFLINE=unset" in (tmp_path / "results/calls").read_text()


OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def test_serve_generic_offline_env_scoped_to_engine(tmp_path):
    """YAML env offline flags are for the engine only: dataset fetch and model pre-download see them blank."""
    lib = tmp_path / "benchmarks"
    (lib / "single_node/agentic").mkdir(parents=True)
    probe = " ".join(f"{v}=${{{v}-unset}}" for v in OFFLINE_VARS)
    (lib / "benchmark_lib.sh").write_text(STUB_LIB.replace(
        "is_custom_dataset() { return 1; }", "is_custom_dataset() { return 0; }"
    ).replace(
        'wait_for_server_ready() { echo "WAIT health $*" >> "$RESULT_DIR/calls"; }',
        'wait_for_server_ready() { while [ ! -f "$RESULT_DIR/engine_env" ]; do sleep 0.05; done; }',
    ).replace(
        'build_replay_cmd_yaml() { echo "METRICS $AIPERF_SERVER_METRICS_URLS" >> "$RESULT_DIR/calls"; }',
        f'build_replay_cmd_yaml() {{ echo "REPLAY {probe}" >> "$RESULT_DIR/calls"; }}',
    ) + f'fetch_custom_dataset() {{ echo "FETCH {probe}" >> "$RESULT_DIR/calls"; export IX_CUSTOM_DATASET_FILE=/x; }}\n')
    (lib / "single_node/agentic/serve_generic.sh").write_text(GENERIC.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "nvidia-smi": "",
        "engine": f"echo ENGINE {probe} > {tmp_path}/results/engine_env\nsleep 30\n",
        "hfcli": f'echo "CLI {probe}" >> {tmp_path}/results/calls\n',
    }.items():
        (bin_dir / name).write_text("#!/bin/sh\n" + body)
        (bin_dir / name).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "zai-org/GLM-5.3", "CONC": "8", "PORT": "8888",
        "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "sglang", "SERVING_SCENARIO": "s",
        "SERVING_LOADER": "l", "AIPERF_HF_CLI": str(bin_dir / "hfcli"), "SERVE_CMD": "engine",
        "SERVE_ENV": "\n".join(f"{v}=1" for v in OFFLINE_VARS),
    }
    (tmp_path / "results").mkdir()
    out = subprocess.run(["bash", str(lib / "single_node/agentic/serve_generic.sh")], capture_output=True,
                         text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    engine = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
    blank = "HF_HUB_OFFLINE= TRANSFORMERS_OFFLINE= HF_DATASETS_OFFLINE="
    assert (tmp_path / "results/engine_env").read_text().strip() == f"ENGINE {engine}"
    calls = (tmp_path / "results/calls").read_text().splitlines()
    assert f"FETCH {blank}" in calls
    assert f"CLI {blank}" in calls
    assert "REPLAY HF_HUB_OFFLINE=unset TRANSFORMERS_OFFLINE=unset HF_DATASETS_OFFLINE=unset" in calls


@pytest.mark.parametrize("ports, late, urls", [
    ("9200 9201", "", "http://127.0.0.1:9200 http://127.0.0.1:9201"),
    (None, None, "http://127.0.0.1:9200"),
    ("9200 9201 9202", "2", "http://127.0.0.1:9200"),
], ids=["replicas", "single-engine", "scale-up-leaves-late-out"])
def test_serve_router_expands_engine_urls(tmp_path, ports, late, urls):
    cmd = "python3 -c 'import sys; print(*sys.argv[1:])' --worker-urls {{engine_urls}}"
    out = run_generic_router(tmp_path, ROUTER_CMD=cmd, ENGINE_PORTS=ports, SCALE_UP_REPLICAS=late)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == f"--worker-urls {urls}"


def run_generic_replicas(tmp_path, stub_lib=STUB_LIB, **extra):
    lib = tmp_path / "benchmarks"
    (lib / "single_node/agentic").mkdir(parents=True)
    (lib / "benchmark_lib.sh").write_text(stub_lib)
    script = lib / "single_node/agentic/serve_generic.sh"
    script.write_text(GENERIC.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    engine = (f'echo "$CUDA_VISIBLE_DEVICES $*" >> {tmp_path}/engines\n'
              f'echo "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH" > {tmp_path}/engine_env\nsleep 30\n')
    for name, body in {"nvidia-smi": "", "engine": engine}.items():
        (bin_dir / name).write_text("#!/bin/sh\n" + body)
        (bin_dir / name).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "RedHatAI/gemma-4-31B-it-FP8-block", "CONC": "8",
        "PORT": "8888", "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "vllm", "SERVING_SCENARIO": "s",
        "SERVING_LOADER": "l", "AIPERF_HF_CLI": "true", "SERVED_MODEL_NAME": "RedHatAI/gemma-4-31B-it-FP8-block",
        "SERVE_CMD": "engine --tensor-parallel-size 2", "ROUTER_CMD": "vllm-router", "ROUTER_PROFILE": "generic",
        "ENGINE_PORT": "9200", "ENGINE_PORTS": "9200 9201", "REPLICAS": "2", "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    } | extra
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)


def test_serve_generic_replicas_split_gpus_ports_and_logs(tmp_path):
    out = run_generic_replicas(tmp_path)
    assert out.returncode == 0, out.stderr
    engines = tmp_path / "engines"
    for _ in range(100):
        if engines.exists() and len(engines.read_text().splitlines()) == 2:
            break
        time.sleep(0.05)
    assert sorted(engines.read_text().splitlines()) == [
        "4,5 --tensor-parallel-size 2 --host 0.0.0.0 --port 9200",
        "6,7 --tensor-parallel-size 2 --host 0.0.0.0 --port 9201",
    ]
    calls = (tmp_path / "results/calls").read_text()
    assert f"WAIT health --port 9200 --server-log {tmp_path}/results/server.log" in calls
    assert f"WAIT health --port 9201 --server-log {tmp_path}/results/server.1.log" in calls
    assert "WAIT frontend --port 8888 --model RedHatAI/gemma-4-31B-it-FP8-block" in calls
    assert "METRICS http://localhost:8888/metrics,http://localhost:9200/metrics,http://localhost:9201/metrics" in calls
    assert (tmp_path / "results/server_command.1.txt").read_text().strip().endswith("--port 9201")


@pytest.fixture
def add_worker_router():
    posts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Successfully added worker")

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"urls": []}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], posts
    server.shutdown()


def test_serve_generic_scale_up_adds_late_engine_after_profiling_starts(tmp_path, add_worker_router):
    port, posts = add_worker_router
    # the stub replay logs aiperf's profiling start, then outlives SCALE_UP_AFTER
    replay = ('run_agentic_replay_and_write_outputs() { mkdir -p "$1/aiperf_artifacts/logs"; '
              'echo "PhaseRunner - NOTICE - Phase profiling (profiling) started" > "$1/aiperf_artifacts/logs/aiperf.log"; sleep 3; }')
    out = run_generic_replicas(
        tmp_path, stub_lib=STUB_LIB.replace("run_agentic_replay_and_write_outputs() { :; }", replay),
        PORT=str(port), SCALE_UP_REPLICAS="1", SCALE_UP_AFTER="0.5", STORAGE_MASTER_PORT="50051",
        SERVE_ENV='IX_MOONCAKE_STORE_CONFIG={"master_server_address": "127.0.0.1:{{storage_master_port}}"}\n',
    )
    assert out.returncode == 0, out.stderr
    assert posts == ["/add_worker?url=http://127.0.0.1:9201"]
    joined = [json.loads(line) for line in (tmp_path / "results/scale_up.jsonl").read_text().splitlines()]
    assert [j["url"] for j in joined] == ["http://127.0.0.1:9201"] and joined[0]["added_ns"] > 0
    config = tmp_path / "results/mooncake_store_config.json"
    assert json.loads(config.read_text()) == {"master_server_address": "127.0.0.1:50051"}
    assert f"MOONCAKE_CONFIG_PATH={config}" in (tmp_path / "engine_env").read_text()


def test_serve_generic_replicas_need_one_port_each(tmp_path):
    out = run_generic_replicas(tmp_path, ENGINE_PORTS="9200")
    assert out.returncode == 1
    assert "replicas 2 needs 2 ports in ENGINE_PORTS" in out.stderr


def test_serve_router_never_prints_router_env_values(tmp_path):
    cmd = "python3 -c 'print(1)'"
    out = subprocess.run(["bash", str(ROUTER)], cwd=REPO, capture_output=True, text=True, env={
        "PATH": "/usr/bin:/bin", "ROUTER_CMD": cmd, "ROUTER_PROFILE": "dynamo", "ROUTER_ENV": "FOO=s3cr3t-value\n", "PORT": "9999",
        "RESULT_DIR": str(tmp_path), "DYN_DISCOVERY_BACKEND": "file", "DYN_FILE_KV": str(tmp_path / ".dynkv"),
        "KV_OFFLOADING": "none",
    })
    assert out.returncode == 0, out.stderr
    assert "router env: FOO" in out.stdout
    assert "s3cr3t-value" not in out.stdout + out.stderr
