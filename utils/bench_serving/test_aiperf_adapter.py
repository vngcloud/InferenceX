from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aiperf_adapter import (
    SEARCH_RECIPES,
    build_result,
    extract_max_concurrency,
    winner_from_history,
    winner_profile_export,
)


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
        "time_to_first_token": {"avg": 101.0, "p95": 150.0, "p99": 202.0},
        "inter_token_latency": {"avg": 11.0, "p95": 18.0, "p99": 22.0},
        "request_latency": {"avg": 1111.0, "p95": 1800.0, "p99": 2222.0},
    }


def test_build_result_maps_aiperf_profile_export():
    result = build_result(_artifact(concurrency=32), max_concurrency=32)

    assert result == {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_concurrency": 32,
        "total_token_throughput": 1234.5,
        "output_throughput": 987.6,
        "mean_ttft_ms": 101.0,
        "p95_ttft_ms": 150.0,
        "p99_ttft_ms": 202.0,
        "mean_tpot_ms": 11.0,
        "p95_tpot_ms": 18.0,
        "p99_tpot_ms": 22.0,
        "mean_itl_ms": 11.0,
        "p95_itl_ms": 18.0,
        "p99_itl_ms": 22.0,
        "mean_e2el_ms": 1111.0,
        "p95_e2el_ms": 1800.0,
        "p99_e2el_ms": 2222.0,
    }


def test_extract_max_concurrency_reads_profiling_phase():
    assert extract_max_concurrency(_artifact(concurrency=64)) == 64


def _history(concurrency: int, *, feasible: bool, feasible_count: int) -> dict:
    """A minimal search_history.json payload (AIPerf 0.9.0 schema subset)."""
    return {
        "config": {"objectives": [], "sla_filters": []},
        "iterations": [],
        "best_trials": [
            {
                "iteration_idx": 3,
                "objective_values": [float(concurrency) * 100.0],
                # AIPerf keys variation_values by dotted parameter path.
                "variation_values": {"phases.profiling.concurrency": concurrency},
                "feasible": feasible,
                "feasible_count": feasible_count,
                "pareto_rank": 0,
            }
        ],
        "boundary_summary": None,
        "recipe": "max-throughput-itl-sla",
        "convergence_reason": "max_iterations",
    }


def test_winner_from_history_reads_dotted_concurrency_and_feasibility():
    conc, sla_met = winner_from_history(
        _history(24, feasible=True, feasible_count=5)
    )
    assert conc == 24
    assert sla_met is True


def test_winner_from_history_leaf_concurrency_key():
    history = _history(16, feasible=True, feasible_count=2)
    history["best_trials"][0]["variation_values"] = {"concurrency": 16}
    conc, sla_met = winner_from_history(history)
    assert conc == 16
    assert sla_met is True


def test_winner_from_history_infeasible_when_no_feasible_count():
    # AIPerf falls back to the full pool with feasible_count == 0 when no probed
    # point met the SLA; the adapter surfaces that as sla_met=False.
    conc, sla_met = winner_from_history(
        _history(32, feasible=False, feasible_count=0)
    )
    assert conc == 32
    assert sla_met is False


def test_winner_profile_export_prefers_direct_then_nested(tmp_path: Path):
    base = tmp_path / "bmk_aiperf"
    # Direct per-variation file.
    direct = base / "concurrency_16"
    direct.mkdir(parents=True)
    (direct / "profile_export_aiperf.json").write_text(
        json.dumps(_artifact(concurrency=16))
    )
    assert winner_profile_export(base, 16)["input_config"]["phases"][1][
        "concurrency"
    ] == 16

    # Nested under an aggregate/ subdir (multi-trial cell layout).
    nested = base / "concurrency_32" / "aggregate"
    nested.mkdir(parents=True)
    (nested / "profile_export_aiperf.json").write_text(
        json.dumps(_artifact(concurrency=32))
    )
    assert winner_profile_export(base, 32)["input_config"]["phases"][1][
        "concurrency"
    ] == 32


def test_winner_profile_export_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        winner_profile_export(tmp_path / "bmk_aiperf", 99)


def test_search_recipes_are_native_names():
    # These must match AIPerf's own recipe names (aiperf.search_recipes.builtins)
    # since the adapter forwards them verbatim to `aiperf profile`.
    assert "max-throughput-itl-sla" in SEARCH_RECIPES
    assert "max-concurrency-under-sla" in SEARCH_RECIPES


# Fake `aiperf` that mimics the native BO search: it reads the concurrency
# range, "optimises" to --concurrency-max as the winner, writes
# search_history.json next to the artifact dir root, and writes the winner's
# per-variation profile_export_aiperf.json under concurrency_<winner>/.
_FAKE_AIPERF_BO = (
    "#!/usr/bin/env python3\n"
    "import json, sys\n"
    "from pathlib import Path\n"
    "argv = sys.argv\n"
    "def opt(name):\n"
    "    return argv[argv.index(name) + 1] if name in argv else None\n"
    "art = Path(opt('--artifact-dir'))\n"
    "lo = int(opt('--concurrency-min')); hi = int(opt('--concurrency-max'))\n"
    "winner = hi  # pretend BO converged on the top of the range\n"
    "art.mkdir(parents=True, exist_ok=True)\n"
    "history = {\n"
    "    'best_trials': [{'iteration_idx': 1,\n"
    "        'objective_values': [winner * 100.0],\n"
    "        'variation_values': {'phases.profiling.concurrency': winner},\n"
    "        'feasible': True, 'feasible_count': 3, 'pareto_rank': 0}],\n"
    "    'recipe': opt('--search-recipe'), 'convergence_reason': 'max_iterations'}\n"
    "(art / 'search_history.json').write_text(json.dumps(history))\n"
    "cell = art / ('concurrency_%d' % winner)\n"
    "cell.mkdir(parents=True, exist_ok=True)\n"
    "artifact = {\n"
    "    'input_config': {'models': {'items': [{'name': 'm'}]},\n"
    "        'phases': [{'name': 'warmup', 'concurrency': 2},\n"
    "                   {'name': 'profiling', 'concurrency': winner}]},\n"
    "    'total_token_throughput': {'avg': winner * 100.0},\n"
    "    'output_token_throughput': {'avg': winner * 80.0},\n"
    "    'time_to_first_token': {'avg': 100.0, 'p95': 150.0, 'p99': 200.0},\n"
    "    'inter_token_latency': {'avg': winner * 1.0, 'p95': winner * 2.0, 'p99': winner * 2.5},\n"
    "    'request_latency': {'avg': 1000.0, 'p95': 1500.0, 'p99': 2000.0},\n"
    "}\n"
    "(cell / 'profile_export_aiperf.json').write_text(json.dumps(artifact))\n"
)


def test_search_delegates_to_native_bo_and_feeds_process_result(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_aiperf = bin_dir / "aiperf"
    fake_aiperf.write_text(_FAKE_AIPERF_BO)
    fake_aiperf.chmod(0o755)

    result_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    proc = subprocess.run(
        [
            sys.executable, str(ADAPTER),
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--url", "http://0.0.0.0:8888",
            "--request-count", "320",
            "--result-filename", "bmk",
            "--result-dir", str(result_dir),
            "--isl", "1024", "--osl", "1024", "--random-seed", "1",
            "--search-recipe", "max-throughput-itl-sla",
            "--concurrency-min", "8", "--concurrency-max", "32",
            "--sla-ms", "50",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BO winner concurrency=32" in proc.stderr

    result = json.loads((result_dir / "bmk.json").read_text())
    assert result["max_concurrency"] == 32
    assert result["total_token_throughput"] == 3200.0
    assert result["p95_itl_ms"] == 64.0
    assert result["sla_met"] is True
    assert result["search_recipe"] == "max-throughput-itl-sla"

    # The intermediate result must still flow through process_result unchanged.
    process_env = env.copy()
    process_env.update(
        {
            "RUNNER_TYPE": "h100", "FRAMEWORK": "vllm", "PRECISION": "fp8",
            "SPEC_DECODING": "none", "RESULT_FILENAME": "bmk", "ISL": "1024",
            "OSL": "1024", "DISAGG": "false", "MODEL_PREFIX": "llama",
            "IMAGE": "test-image", "TP": "8", "EP_SIZE": "1",
            "DP_ATTENTION": "false", "BENCHMARK_CLIENT": "aiperf",
        }
    )
    processed = subprocess.run(
        [sys.executable, str(PROCESS_RESULT)],
        cwd=result_dir, env=process_env, capture_output=True, text=True,
    )
    assert processed.returncode == 0, processed.stderr
    agg = json.loads((result_dir / "agg_bmk.json").read_text())
    assert agg["conc"] == 32
    assert agg["tput_per_gpu"] == pytest.approx(3200.0 / 8)


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
