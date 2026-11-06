import json
import os
import shlex
import socket
import subprocess
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[3] / "benchmarks/single_node/serving/bench.sh"
REAL_BENCH_PKG = Path(__file__).resolve().parents[2] / "bench"
# Stands in for python3 -m dynamo.vllm and dynamo.sglang: records what it was started with, then
# serves /health on its Dynamo system port until bench.sh kills it.
FAKE_ENGINE = """\
import http.server, json, os, sys
port = int(os.environ["DYN_SYSTEM_PORT"])
record = {"port": port, "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "argv": sys.argv[1:]}
with open(os.path.join(os.environ["RESULT_DIR"], f"engine-{port}.json"), "w") as out:
    json.dump(record, out)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""
# Stands in for vllm serve: records its env, then serves /health on --port.
FAKE_VLLM = """\
#!/usr/bin/env python3
import http.server, json, os, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
record = {"gpus": os.environ["CUDA_VISIBLE_DEVICES"], "dyn_port": os.environ.get("DYN_SYSTEM_PORT")}
with open(os.path.join(os.environ["RESULT_DIR"], f"engine-{port}.json"), "w") as out:
    json.dump(record, out)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""
FAKE_CLIENT = """\
import json, os, sys
with open(os.path.join(os.environ["RESULT_DIR"], f"client-{sys.argv[1]}.json"), "w") as out:
    json.dump({"metrics": os.environ.get("AIPERF_SERVER_METRICS_URLS")}, out)
"""


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    return ports


def frontend(port: int, model: str) -> ThreadingHTTPServer:
    class Models(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": model}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """bench.sh's cwd with a fake client and Dynamo engines, and a bin with a fake nvidia-smi."""
    workspace = tmp_path / "ws"
    (workspace / "infx/serving").mkdir(parents=True)
    (workspace / "dynamo").mkdir()
    (workspace / "infx/__init__.py").write_text("")
    (workspace / "infx/serving/__init__.py").write_text("")
    (workspace / "infx/serving/client.py").write_text(FAKE_CLIENT)
    (workspace / "infx/bench").symlink_to(REAL_BENCH_PKG)
    (workspace / "dynamo/__init__.py").write_text("")
    (workspace / "dynamo/vllm.py").write_text(FAKE_ENGINE)
    (workspace / "dynamo/sglang.py").write_text(FAKE_ENGINE)
    fakes = tmp_path / "bin"
    fakes.mkdir()
    (fakes / "nvidia-smi").write_text(textwrap.dedent("""\
        #!/bin/bash
        [[ "$*" == *--query-gpu* ]] && printf '%s\\n' 0 1 2 3 4 5 6 7
        exit 0
    """))
    (fakes / "nvidia-smi").chmod(0o755)
    return workspace, fakes


@pytest.mark.parametrize(
    ("module", "model_flag", "extra"),
    [
        ("dynamo.vllm", "--model", {"enable_kv_cache_events": True}),
        ("dynamo.sglang", "--model-path", {}),
    ],
)
def test_dynamo_replicas_get_their_own_gpus_and_ports(tmp_path, module, model_flag, extra):
    workspace_dir, fakes = workspace(tmp_path)
    results = tmp_path / "results"
    port, *system_and_kv = free_ports(5)
    system_ports, kv_ports = system_and_kv[:2], system_and_kv[2:]
    server = frontend(port, "org/model")
    env = {
        "PATH": f"{fakes}:{os.environ['PATH']}", "RESULT_DIR": str(results), "MODEL": "org/model",
        "PORT": str(port), "SERVE_CMD": shlex.join(["python3", "-m", module, model_flag, "org/model"]),
        "ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo",
        "DYN_SYSTEM_PORTS": " ".join(map(str, system_ports)),
        "DYN_KV_EVENT_PORTS": " ".join(map(str, kv_ports)),
    }  # fmt: skip
    try:
        out = subprocess.run(["bash", str(BENCH)], cwd=workspace_dir, env=env, capture_output=True,
                             text=True, check=False, timeout=120)
    finally:
        server.shutdown()
    assert out.returncode == 0, out.stdout + out.stderr
    engines = [json.loads((results / f"engine-{p}.json").read_text()) for p in system_ports]
    assert [engine["gpus"] for engine in engines] == ["0,1,2,3", "4,5,6,7"]
    for engine, kv_port in zip(engines, kv_ports, strict=True):
        flag = engine["argv"].index("--kv-events-config")
        config = {"publisher": "zmq", "endpoint": f"tcp://*:{kv_port}", **extra}
        assert json.loads(engine["argv"][flag + 1]) == config
    metrics = json.loads((results / "client-run.json").read_text())["metrics"].split(",")
    assert metrics == [f"http://localhost:{p}/metrics" for p in (port, *system_ports)]


def test_engines_without_dynamo_get_no_dynamo_system_port(tmp_path):
    workspace_dir, fakes = workspace(tmp_path)
    (fakes / "vllm").write_text(FAKE_VLLM)
    (fakes / "vllm").chmod(0o755)
    results = tmp_path / "results"
    (port,) = free_ports(1)
    env = {
        "PATH": f"{fakes}:{os.environ['PATH']}", "RESULT_DIR": str(results), "MODEL": "org/model",
        "PORT": str(port), "SERVE_CMD": shlex.join(["vllm", "serve", "org/model"]),
    }  # fmt: skip
    out = subprocess.run(["bash", str(BENCH)], cwd=workspace_dir, env=env, capture_output=True,
                         text=True, check=False, timeout=120)  # fmt: skip
    assert out.returncode == 0, out.stdout + out.stderr
    engine = json.loads((results / f"engine-{port}.json").read_text())
    assert engine == {"gpus": "0,1,2,3,4,5,6,7", "dyn_port": None}
