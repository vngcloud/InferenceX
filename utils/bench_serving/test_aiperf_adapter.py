from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aiperf_adapter import build_result, detect_mode, extract_max_concurrency


ADAPTER = Path(__file__).resolve().parent / "aiperf_adapter.py"
PROCESS_RESULT = Path(__file__).resolve().parents[1] / "process_result.py"


def _artifact(concurrency: int = 16) -> dict:
    return {
        "input_config": {
            "models": {"items": [{"name": "meta-llama/Llama-3.1-8B-Instruct"}]},
            "phases": [
                {"name": "warmup", "concurrency": 2},
                {"name": "profiling", "concurrency": concurrency},
            ],
        },
        "total_token_throughput": {"avg": 1234.5},
        "output_token_throughput": {"avg": 987.6},
        "time_to_first_token": {"avg": 101.0, "p99": 202.0},
        "inter_token_latency": {"avg": 11.0, "p99": 22.0},
        "request_latency": {"avg": 1111.0, "p99": 2222.0},
    }


def test_build_result_maps_aiperf_profile_export():
    result = build_result(_artifact(concurrency=32), max_concurrency=32)

    assert result == {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_concurrency": 32,
        "total_token_throughput": 1234.5,
        "output_throughput": 987.6,
        "mean_ttft_ms": 101.0,
        "p99_ttft_ms": 202.0,
        "mean_tpot_ms": 11.0,
        "p99_tpot_ms": 22.0,
        "mean_itl_ms": 11.0,
        "p99_itl_ms": 22.0,
        "mean_e2el_ms": 1111.0,
        "p99_e2el_ms": 2222.0,
    }


def test_detect_mode_fixed_without_search_history(tmp_path: Path):
    assert detect_mode(tmp_path) == "fixed"


def test_detect_mode_search_with_search_history(tmp_path: Path):
    (tmp_path / "search_history.json").write_text("{}")
    assert detect_mode(tmp_path) == "search"


def test_extract_max_concurrency_fixed_reads_profiling_phase():
    assert extract_max_concurrency(_artifact(concurrency=64), None, "fixed") == 64


def test_extract_max_concurrency_search_reads_best_trial():
    search_history = {
        "best_trials": [
            {"variation_values": {"phases.profiling.concurrency": 128}}
        ]
    }
    assert extract_max_concurrency(_artifact(), search_history, "search") == 128


@pytest.mark.integration
def test_main_against_live_server(tmp_path: Path):
    """End-to-end against a real aiperf + serving stack.

    Skipped unless AIPERF_LIVE_URL points at an OpenAI-compatible endpoint
    (e.g. a port-forwarded vLLM server). Drives the real `aiperf` binary, so
    it also pins the adapter's parsing to the installed AIPerf schema rather
    than the hand-built fixtures above. Optional: AIPERF_LIVE_MODEL,
    AIPERF_LIVE_GPU_TELEMETRY_URL.
    """
    url = os.environ.get("AIPERF_LIVE_URL")
    if not url:
        pytest.skip("set AIPERF_LIVE_URL to run the live integration test")
    if not shutil.which("aiperf"):
        pytest.skip("aiperf binary not on PATH")

    model = os.environ.get("AIPERF_LIVE_MODEL", "google/gemma-4-31B-it")
    result_dir = tmp_path / "results"
    cmd = [
        sys.executable, str(ADAPTER),
        "--model", model,
        "--url", url,
        "--endpoint-type", "chat",
        "--concurrency", "4",
        "--request-count", "20",
        "--result-filename", "bmk",
        "--result-dir", str(result_dir),
        "--isl", "128", "--osl", "32", "--random-seed", "1",
    ]
    gpu_url = os.environ.get("AIPERF_LIVE_GPU_TELEMETRY_URL")
    if gpu_url:
        cmd += ["--gpu-telemetry-url", gpu_url]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    result = json.loads((result_dir / "bmk.json").read_text())
    assert result["model_id"] == model
    assert result["max_concurrency"] == 4
    assert result["total_token_throughput"] > 0
    assert result["output_throughput"] > 0
    assert result["mean_ttft_ms"] > 0
    assert result["mean_e2el_ms"] >= result["mean_ttft_ms"]

    process_env = os.environ.copy()
    process_env.update(
        {
            "RUNNER_TYPE": "h100", "FRAMEWORK": "vllm", "PRECISION": "bf16",
            "SPEC_DECODING": "none", "RESULT_FILENAME": "bmk", "ISL": "128",
            "OSL": "32", "DISAGG": "false", "MODEL_PREFIX": "gemma",
            "IMAGE": "vllm/vllm-openai:latest", "TP": "2", "EP_SIZE": "1",
            "DP_ATTENTION": "false", "BENCHMARK_CLIENT": "aiperf",
        }
    )
    processed = subprocess.run(
        [sys.executable, str(PROCESS_RESULT)],
        cwd=result_dir, env=process_env, capture_output=True, text=True,
    )
    assert processed.returncode == 0, processed.stderr
    agg = json.loads((result_dir / "agg_bmk.json").read_text())
    assert agg["benchmark_client"] == "aiperf"
    assert agg["conc"] == 4
    assert agg["tput_per_gpu"] == pytest.approx(result["total_token_throughput"] / 2)
    assert agg["mean_ttft"] == pytest.approx(result["mean_ttft_ms"] / 1000.0)


def test_main_writes_result_consumed_by_process_result(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_aiperf = bin_dir / "aiperf"
    fake_aiperf.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "artifact_dir = Path(sys.argv[sys.argv.index('--artifact-dir') + 1])\n"
        "artifact_dir.mkdir(parents=True, exist_ok=True)\n"
        f"artifact = {json.dumps(_artifact(concurrency=16))!r}\n"
        "(artifact_dir / 'profile_export_aiperf.json').write_text(artifact)\n"
    )
    fake_aiperf.chmod(0o755)

    result_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    proc = subprocess.run(
        [
            sys.executable,
            str(ADAPTER),
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--url",
            "http://0.0.0.0:8888",
            "--concurrency",
            "16",
            "--request-count",
            "160",
            "--result-filename",
            "bmk",
            "--result-dir",
            str(result_dir),
            "--isl",
            "1024",
            "--osl",
            "1024",
            "--random-seed",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads((result_dir / "bmk.json").read_text())["output_throughput"] == 987.6

    process_env = env.copy()
    process_env.update(
        {
            "RUNNER_TYPE": "h100",
            "FRAMEWORK": "vllm",
            "PRECISION": "fp8",
            "SPEC_DECODING": "none",
            "RESULT_FILENAME": "bmk",
            "ISL": "1024",
            "OSL": "1024",
            "DISAGG": "false",
            "MODEL_PREFIX": "llama",
            "IMAGE": "test-image",
            "TP": "8",
            "EP_SIZE": "1",
            "DP_ATTENTION": "false",
            "BENCHMARK_CLIENT": "aiperf",
        }
    )
    processed = subprocess.run(
        [sys.executable, str(PROCESS_RESULT)],
        cwd=result_dir,
        env=process_env,
        capture_output=True,
        text=True,
    )
    assert processed.returncode == 0, processed.stderr
    agg = json.loads((result_dir / "agg_bmk.json").read_text())
    assert agg["benchmark_client"] == "aiperf"
    assert agg["output_tput_per_gpu"] == pytest.approx(987.6 / 8)
    assert agg["mean_ttft"] == pytest.approx(0.101)
