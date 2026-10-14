import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = r'''
    export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=180
    export INFMAX_CONTAINER_WORKSPACE="$PWD"
    export AIPERF_CLI=aiperf MODEL=Qwen/Qwen3.8-27B FRAMEWORK=vllm
    export SERVING_SCENARIO=agentx-cc-062126-smoke
'''


def run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", "set -e\n" + ENV + "source benchmarks/benchmark_lib.sh\n" + body],
                          cwd=REPO, capture_output=True, text=True)


def test_replay_cmd_exact():
    out = run(r'''
        export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
        export AIPERF_SERVER_METRICS_URLS=http://localhost:8888/metrics
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert out.stdout == (
        f"aiperf profile --config {REPO}/scenarios/agentx-cc-062126-smoke.yaml"
        " --url http://localhost:8888 --model Qwen/Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B"
        " --concurrency 8 --gpu-telemetry http://localhost:9400/metrics"
        " --server-metrics http://localhost:8888/metrics --output-artifact-dir /tmp/r/aiperf_artifacts"
    )


def test_served_model_name_and_no_telemetry():
    out = run(r'''
        export SERVED_MODEL_NAME=q38
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert " --model q38 --tokenizer Qwen/Qwen3.8-27B " in out.stdout
    assert " --no-gpu-telemetry" in out.stdout
    assert "--server-metrics" not in out.stdout


def test_multiple_metrics_urls():
    out = run(r'''
        export AIPERF_SERVER_METRICS_URLS=http://a/metrics,http://b/metrics
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert " --server-metrics http://a/metrics http://b/metrics " in out.stdout


def test_missing_scenario_fails():
    out = run(r'''
        export SERVING_SCENARIO=nope
        build_replay_cmd_yaml /tmp/r
    ''')
    assert out.returncode != 0
    assert "scenarios/nope.yaml" in out.stderr


def test_serve_generic_exports_env_without_tracing_values(tmp_path):
    # nvidia-smi is the first command after the env export: the stub records
    # what the recipe exported, then stops it (set -e) before any real work.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(f'#!/bin/sh\necho "$HF_TOKEN $FOO" > {tmp_path}/seen\nexit 7\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "Qwen/Qwen3.8-27B", "CONC": "8", "PORT": "8888",
        "RESULT_DIR": str(tmp_path / "r"), "FRAMEWORK": "vllm", "SERVE_CMD": "vllm serve Qwen/Qwen3.8-27B",
        "SERVING_SCENARIO": "agentx-cc-062126-smoke", "SERVING_LOADER": "x", "KV_OFFLOADING": "none",
        "SERVING_YAML": "env:\n  HF_TOKEN: hf_secret123", "SERVE_ENV": "HF_TOKEN=hf_secret123\nFOO=a b",
    }
    out = subprocess.run(["bash", str(REPO / "benchmarks/single_node/agentic/serve_generic.sh")],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 7
    assert (tmp_path / "seen").read_text() == "hf_secret123 a b\n"
    assert (tmp_path / "r/serving.yaml").read_text() == "env:\n  HF_TOKEN: hf_secret123"
    assert "hf_secret123" not in out.stdout + out.stderr
    assert out.stdout.splitlines()[:2] == ["serve env: HF_TOKEN", "serve env: FOO"]
