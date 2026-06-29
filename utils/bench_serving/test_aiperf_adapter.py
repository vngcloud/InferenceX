from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import argparse

import aiperf_adapter
from aiperf_adapter import (
    build_result,
    detect_mode,
    extract_max_concurrency,
    run_aiperf,
    validate_request_counts,
)


ADAPTER = Path(__file__).resolve().parent / "aiperf_adapter.py"
PROCESS_RESULT = Path(__file__).resolve().parents[1] / "process_result.py"


def _artifact(concurrency: int = 16, request_count: int = 160) -> dict:
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
            "avg": 101.0, "p50": 150.0, "p75": 160.0, "p90": 180.0, "p95": 190.0, "p99": 202.0,
        },
        "inter_token_latency": {
            "avg": 11.0, "p50": 12.0, "p75": 14.0, "p90": 16.0, "p95": 18.0, "p99": 22.0,
        },
        "request_latency": {
            "avg": 1111.0, "p50": 1500.0, "p75": 1600.0, "p90": 1800.0, "p95": 1900.0, "p99": 2222.0,
        },
        "request_count": {"avg": float(request_count)},
    }


def _run_aiperf_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Build a Namespace covering every attribute run_aiperf reads."""
    defaults = dict(
        model="Qwen/Qwen3-4B-Instruct-2507",
        url="http://0.0.0.0:8888",
        endpoint_type="chat",
        scenario=None,
        endpoint=None,
        concurrency=8,
        request_count=50,
        benchmark_duration=None,
        result_dir=tmp_path,
        result_filename="bmk",
        warmup_request_count=None,
        num_warmup_sessions=None,
        no_fixed_schedule=False,
        server_metrics_url=None,
        gpu_telemetry_url=None,
        public_dataset=None,
        input_file=None,
        custom_dataset_type=None,
        tokenizer=None,
        isl=None,
        osl=None,
        random_seed=None,
        extra_inputs=[],
        goodput=None,
        temperature=None,
        inter_turn_delay_cap_seconds=None,
        use_think_time_only=False,
        dataset_sampling_strategy=None,
        benchmark_grace_period=None,
        workers_max=None,
        failed_request_threshold=None,
        trajectory_start_min_ratio=None,
        trajectory_start_max_ratio=None,
        use_server_token_count=False,
        tokenizer_trust_remote_code=False,
        num_dataset_entries=None,
        slice_duration=None,
        unsafe_override=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _capture_aiperf_cmd(monkeypatch, args) -> list[str]:
    captured: dict = {}

    def fake_run(cmd, check):  # noqa: ARG001 - mirror subprocess.run signature used
        captured["cmd"] = cmd

    monkeypatch.setattr(aiperf_adapter.subprocess, "run", fake_run)
    run_aiperf(args)
    return captured["cmd"]


def test_run_aiperf_mode1_flags_present(tmp_path: Path, monkeypatch):
    """Mode 1 capacity-sweep flags are forwarded to the aiperf CLI."""
    args = _run_aiperf_args(
        tmp_path,
        no_fixed_schedule=True,
        num_warmup_sessions=1,
        input_file="trace.jsonl",
        custom_dataset_type="mooncake_trace",
    )
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert "--no-fixed-schedule" in cmd
    assert ["--num-warmup-sessions", "1"] == cmd[cmd.index("--num-warmup-sessions"):cmd.index("--num-warmup-sessions") + 2]
    assert cmd[cmd.index("--request-count") + 1] == "50"
    assert cmd[cmd.index("--input-file") + 1] == "trace.jsonl"


def test_run_aiperf_omits_mode1_flags_by_default(tmp_path: Path, monkeypatch):
    """Without Mode 1 opt-in the flags are absent (single-replay behavior)."""
    args = _run_aiperf_args(tmp_path, input_file="trace.jsonl")
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert "--no-fixed-schedule" not in cmd
    assert "--num-warmup-sessions" not in cmd


def test_run_aiperf_forwards_use_think_time_only(tmp_path: Path, monkeypatch):
    args = _run_aiperf_args(tmp_path, input_file="trace_dir", use_think_time_only=True)
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert "--use-think-time-only" in cmd

def test_run_aiperf_forwards_tokenizer_when_set(tmp_path: Path, monkeypatch):
    """An explicit tokenizer is forwarded; the model name is left untouched."""
    args = _run_aiperf_args(tmp_path, tokenizer="google/gemma-3-27b-it")
    cmd = _capture_aiperf_cmd(monkeypatch, args)
    assert cmd[cmd.index("--tokenizer") + 1] == "google/gemma-3-27b-it"


def test_run_aiperf_omits_tokenizer_by_default(tmp_path: Path, monkeypatch):
    """Unset tokenizer => no flag => aiperf defaults to the served model."""
    args = _run_aiperf_args(tmp_path)
    cmd = _capture_aiperf_cmd(monkeypatch, args)
    assert "--tokenizer" not in cmd


def test_run_aiperf_forwards_extra_inputs(tmp_path: Path, monkeypatch):
    args = _run_aiperf_args(
        tmp_path,
        extra_inputs=["ignore_eos:true", "min_tokens:512"],
    )
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    # aiperf expects one key:value per --extra-inputs, so the flag repeats.
    values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--extra-inputs"]
    assert values == ["ignore_eos:true", "min_tokens:512"]


def test_run_aiperf_duration_mode_omits_request_count(tmp_path: Path, monkeypatch):
    """Duration-based smoke: --benchmark-duration is passed and --request-count is
    omitted when no request_count is set."""
    args = _run_aiperf_args(
        tmp_path,
        request_count=None,
        benchmark_duration=90.0,
        input_file="trace.jsonl",
        custom_dataset_type="mooncake_trace",
    )
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert "--request-count" not in cmd
    assert cmd[cmd.index("--benchmark-duration") + 1] == "90.0"


def test_run_aiperf_forwards_agentx_weka_flags(tmp_path: Path, monkeypatch):
    args = _run_aiperf_args(
        tmp_path,
        request_count=None,
        benchmark_duration=90.0,
        scenario="inferencex-agentx-mvp",
        endpoint="/v1/chat/completions",
        input_file="trace_dir",
        custom_dataset_type="weka_trace",
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
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert cmd[cmd.index("--scenario") + 1] == "inferencex-agentx-mvp"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"
    assert "--output-artifact-dir" in cmd
    assert "--artifact-dir" not in cmd
    assert cmd[cmd.index("--failed-request-threshold") + 1] == "0.05"
    assert cmd[cmd.index("--trajectory-start-min-ratio") + 1] == "0.25"
    assert cmd[cmd.index("--trajectory-start-max-ratio") + 1] == "0.75"
    assert "--use-server-token-count" in cmd
    assert "--tokenizer-trust-remote-code" in cmd
    assert cmd[cmd.index("--num-dataset-entries") + 1] == "949"
    assert cmd[cmd.index("--slice-duration") + 1] == "1.0"
    assert "--unsafe-override" in cmd


def test_run_aiperf_agentx_weka_drops_legacy_flags(tmp_path: Path, monkeypatch):
    args = _run_aiperf_args(
        tmp_path,
        scenario="inferencex-agentx-mvp",
        input_file="trace_dir",
        custom_dataset_type="weka_trace",
        warmup_request_count=2,
        num_warmup_sessions=1,
        no_fixed_schedule=True,
        inter_turn_delay_cap_seconds=60,
        use_think_time_only=True,
    )
    cmd = _capture_aiperf_cmd(monkeypatch, args)

    assert "--warmup-request-count" not in cmd
    assert "--num-warmup-sessions" not in cmd
    assert "--no-fixed-schedule" not in cmd
    assert "--inter-turn-delay-cap-seconds" not in cmd
    assert "--use-think-time-only" not in cmd


def test_main_skips_request_count_validation_in_duration_mode(tmp_path: Path, monkeypatch):
    """In duration mode the completed count is unknown and overflow/errored turns
    are expected, so main() must not call validate_request_counts."""
    artifact_dir = tmp_path / "bmk_aiperf"
    artifact_dir.mkdir(parents=True)
    # An artifact that would FAIL validate_request_counts (errors > 0).
    artifact = _artifact(request_count=10)
    artifact["error_request_count"] = {"avg": 3.0}
    (artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(artifact))

    monkeypatch.setattr(aiperf_adapter, "run_aiperf", lambda args: artifact_dir)
    called = {"validated": False}
    monkeypatch.setattr(
        aiperf_adapter, "validate_request_counts",
        lambda *a, **k: called.__setitem__("validated", True),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["aiperf_adapter.py", "--model", "m", "--url", "u", "--concurrency", "4",
         "--benchmark-duration", "90", "--result-filename", "bmk",
         "--result-dir", str(tmp_path)],
    )

    aiperf_adapter.main()

    assert called["validated"] is False
    assert (tmp_path / "bmk.json").exists()


def test_build_result_maps_aiperf_profile_export():
    result = build_result(_artifact(concurrency=32), max_concurrency=32)

    assert result == {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_concurrency": 32,
        "total_token_throughput": 1234.5,
        "output_throughput": 987.6,
        "mean_ttft_ms": 101.0,
        "p50_ttft_ms": 150.0, "p75_ttft_ms": 160.0, "p90_ttft_ms": 180.0,
        "p95_ttft_ms": 190.0, "p99_ttft_ms": 202.0,
        "mean_tpot_ms": 11.0,
        "p50_tpot_ms": 12.0, "p75_tpot_ms": 14.0, "p90_tpot_ms": 16.0,
        "p95_tpot_ms": 18.0, "p99_tpot_ms": 22.0,
        "mean_itl_ms": 11.0,
        "p50_itl_ms": 12.0, "p75_itl_ms": 14.0, "p90_itl_ms": 16.0,
        "p95_itl_ms": 18.0, "p99_itl_ms": 22.0,
        "mean_e2el_ms": 1111.0,
        "p50_e2el_ms": 1500.0, "p75_e2el_ms": 1600.0, "p90_e2el_ms": 1800.0,
        "p95_e2el_ms": 1900.0, "p99_e2el_ms": 2222.0,
    }


def test_build_result_maps_benchmark_duration_when_present():
    artifact = _artifact(concurrency=32)
    artifact["benchmark_duration"] = {"avg": 42.5}

    result = build_result(artifact, max_concurrency=32)

    assert result["duration"] == 42.5


def test_build_result_reads_model_from_endpoint_config():
    artifact = _artifact(concurrency=32)
    artifact["input_config"].pop("models")
    artifact["input_config"]["endpoint"] = {
        "model_names": ["Qwen/Qwen3-4B-Instruct-2507"]
    }

    result = build_result(artifact, max_concurrency=32)

    assert result["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_build_result_omits_duration_when_absent():
    result = build_result(_artifact(concurrency=32), max_concurrency=32)

    assert "duration" not in result


def test_detect_mode_fixed_without_search_history(tmp_path: Path):
    assert detect_mode(tmp_path) == "fixed"


def test_detect_mode_search_with_search_history(tmp_path: Path):
    (tmp_path / "search_history.json").write_text("{}")
    assert detect_mode(tmp_path) == "search"


def test_extract_max_concurrency_fixed_reads_profiling_phase():
    assert extract_max_concurrency(_artifact(concurrency=64), None, "fixed") == 64


def test_extract_max_concurrency_fixed_reads_loadgen_concurrency():
    artifact = _artifact(concurrency=64)
    artifact["input_config"].pop("phases")
    artifact["input_config"]["loadgen"] = {"concurrency": 2}

    assert extract_max_concurrency(artifact, None, "fixed") == 2


def test_extract_max_concurrency_search_reads_best_trial():
    search_history = {
        "best_trials": [
            {"variation_values": {"phases.profiling.concurrency": 128}}
        ]
    }
    assert extract_max_concurrency(_artifact(), search_history, "search") == 128

def test_validate_request_counts_accepts_complete_run():
    validate_request_counts(_artifact(request_count=20), expected_request_count=20)

def test_validate_request_counts_rejects_failed_requests():
    artifact = _artifact(request_count=19)
    artifact["error_request_count"] = {"avg": 1.0}

    with pytest.raises(ValueError, match="failed requests"):
        validate_request_counts(artifact, expected_request_count=20)

def test_validate_request_counts_rejects_short_success_count():
    with pytest.raises(ValueError, match="19/20"):
        validate_request_counts(_artifact(request_count=19), expected_request_count=20)

def test_validate_request_counts_rejects_missing_metric():
    artifact = _artifact(request_count=20)
    del artifact["request_count"]

    with pytest.raises(ValueError, match="missing request_count"):
        validate_request_counts(artifact, expected_request_count=20)


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
        "if '--failed-request-threshold' in sys.argv:\n"
        "    print('Unknown option: --failed-request-threshold', file=sys.stderr)\n"
        "    sys.exit(2)\n"
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
