from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aiperf_adapter
from aiperf_adapter import build_result, detect_mode, extract_max_concurrency, run_aiperf


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
        "time_to_first_token": {
            "avg": 101.0, "p50": 150.0, "p75": 160.0, "p90": 180.0,
            "p95": 190.0, "p99": 202.0,
        },
        "inter_token_latency": {
            "avg": 11.0, "p50": 12.0, "p75": 14.0, "p90": 16.0,
            "p95": 18.0, "p99": 22.0,
        },
        "request_latency": {
            "avg": 1111.0, "p50": 1500.0, "p75": 1600.0, "p90": 1800.0,
            "p95": 1900.0, "p99": 2222.0,
        },
    }


def _run_aiperf_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        model="Qwen/Qwen3-4B-Instruct-2507",
        url="http://localhost:8888",
        endpoint_type="chat",
        scenario="inferencex-agentx-mvp",
        endpoint="/v1/chat/completions",
        concurrency=8,
        benchmark_duration=90.0,
        result_dir=tmp_path,
        result_filename="bmk",
        server_metrics_url=None,
        gpu_telemetry_url=None,
        public_dataset="semianalysis_cc_traces_weka_with_subagents_060826",
        input_file=None,
        custom_dataset_type=None,
        tokenizer=None,
        isl=None,
        osl=None,
        random_seed=42,
        failed_request_threshold=0.05,
        trajectory_start_min_ratio=0.25,
        trajectory_start_max_ratio=0.75,
        use_server_token_count=True,
        tokenizer_trust_remote_code=True,
        num_dataset_entries=949,
        slice_duration=1.0,
        unsafe_override=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _capture_aiperf_cmd(monkeypatch, args) -> list[str]:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, check):  # noqa: ARG001
        captured["cmd"] = cmd

    monkeypatch.setattr(aiperf_adapter.subprocess, "run", fake_run)
    run_aiperf(args)
    return captured["cmd"]


def test_run_aiperf_forwards_public_weka(tmp_path: Path, monkeypatch):
    cmd = _capture_aiperf_cmd(monkeypatch, _run_aiperf_args(tmp_path))

    assert cmd[cmd.index("--scenario") + 1] == "inferencex-agentx-mvp"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"
    assert cmd[cmd.index("--benchmark-duration") + 1] == "90.0"
    assert cmd[cmd.index("--public-dataset") + 1] == "semianalysis_cc_traces_weka_with_subagents_060826"
    assert "--output-artifact-dir" in cmd
    assert "--request-count" not in cmd
    assert "--warmup-request-count" not in cmd
    assert "--no-fixed-schedule" not in cmd


def test_run_aiperf_forwards_internal_weka_file(tmp_path: Path, monkeypatch):
    args = _run_aiperf_args(
        tmp_path,
        public_dataset=None,
        input_file="benchmarks/single_node/agentic/datasets/internal.jsonl",
        custom_dataset_type="weka_trace",
        tokenizer="MiniMaxAI/MiniMax-M2.5",
    )
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert cmd[cmd.index("--input-file") + 1] == "benchmarks/single_node/agentic/datasets/internal.jsonl"
    assert cmd[cmd.index("--custom-dataset-type") + 1] == "weka_trace"
    assert cmd[cmd.index("--tokenizer") + 1] == "MiniMaxAI/MiniMax-M2.5"


def test_build_result_maps_aiperf_profile_export():
    result = build_result(_artifact(concurrency=32), max_concurrency=32)

    assert result["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert result["max_concurrency"] == 32
    assert result["total_token_throughput"] == 1234.5
    assert result["output_throughput"] == 987.6
    assert result["mean_ttft_ms"] == 101.0
    assert result["p99_tpot_ms"] == 22.0
    assert result["mean_e2el_ms"] == 1111.0


def test_detect_mode_and_extract_concurrency(tmp_path: Path):
    assert detect_mode(tmp_path) == "fixed"
    assert extract_max_concurrency(_artifact(concurrency=64), None, "fixed") == 64

    (tmp_path / "search_history.json").write_text("{}")
    search_history = {
        "best_trials": [
            {"variation_values": {"phases.profiling.concurrency": 128}}
        ]
    }
    assert detect_mode(tmp_path) == "search"
    assert extract_max_concurrency(_artifact(), search_history, "search") == 128


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
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--url", "http://localhost:8888",
            "--concurrency", "16",
            "--benchmark-duration", "90",
            "--result-filename", "bmk",
            "--result-dir", str(result_dir),
            "--isl", "1024",
            "--osl", "1024",
            "--random-seed", "1",
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
    assert agg["output_tput_per_gpu"] == pytest.approx(987.6 / 8)
    assert agg["mean_ttft"] == pytest.approx(0.101)
