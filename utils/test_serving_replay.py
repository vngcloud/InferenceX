import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ENV = r'''
    export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=180
    export INFMAX_CONTAINER_WORKSPACE="$PWD"
    export AIPERF_CLI=aiperf MODEL=Qwen/Qwen3.8-27B FRAMEWORK=vllm
    export SERVING_SCENARIO=agentx-cc-062126-smoke SERVING_SEARCH=0
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
        "SERVING_SEARCH": "0",
        "SERVING_YAML": "env:\n  HF_TOKEN: hf_secret123", "SERVE_ENV": "HF_TOKEN=hf_secret123\nFOO=a b",
    }
    out = subprocess.run(["bash", str(REPO / "benchmarks/single_node/agentic/serve_generic.sh")],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 7
    assert (tmp_path / "seen").read_text() == "hf_secret123 a b\n"
    assert (tmp_path / "r/serving.yaml").read_text() == "env:\n  HF_TOKEN: hf_secret123"
    assert not (tmp_path / "engine_args.yaml").exists()
    assert "hf_secret123" not in out.stdout + out.stderr
    assert out.stdout.splitlines()[:2] == ["serve env: HF_TOKEN", "serve env: FOO"]


def test_serve_generic_writes_engine_args_file_in_cwd(tmp_path):
    # The trtllm serve command carries --config engine_args.yaml; the
    # recipe must write that file in its own cwd (the engine's cwd) before starting.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 7\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    engine_args = "attn_backend: FLASHINFER\nkv_cache_config:\n  enable_block_reuse: true\n"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "RedHatAI/gemma-4-31B-it-FP8-block", "CONC": "150",
        "PORT": "8888", "RESULT_DIR": str(tmp_path / "r"), "FRAMEWORK": "trtllm",
        "SERVE_CMD": "trtllm-serve serve RedHatAI/gemma-4-31B-it-FP8-block",
        "SERVING_SCENARIO": "agentx-cc-062126-smoke", "SERVING_LOADER": "x", "KV_OFFLOADING": "none",
        "SERVING_SEARCH": "0", "SERVING_YAML": "name: x",
        "SERVING_ENGINE_ARGS": engine_args,
    }
    out = subprocess.run(["bash", str(REPO / "benchmarks/single_node/agentic/serve_generic.sh")],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 7
    assert (tmp_path / "engine_args.yaml").read_text() == engine_args


def _run_serve_generic_with_env_dump(tmp_path: Path, framework: str) -> str:
    # Stubbed nvidia-smi dumps the environment it runs with: everything the
    # recipe exports, strips or keeps before the first command runs.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\nenv > {tmp_path}/seen\nexit 7\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "RedHatAI/gemma-4-31B-it-FP8-block", "CONC": "150",
        "PORT": "8888", "RESULT_DIR": str(tmp_path / "r"), "FRAMEWORK": framework,
        "SERVE_CMD": "trtllm-serve serve RedHatAI/gemma-4-31B-it-FP8-block",
        "SERVING_SCENARIO": "agentx-cc-062126-smoke", "SERVING_LOADER": "x", "KV_OFFLOADING": "none",
        "SERVING_SEARCH": "0", "SERVING_YAML": "name: x",
        "SLURM_JOB_ID": "6809", "SLURM_JOB_NODELIST": "hgx-h200-01", "SLURM_PROCID": "0",
    }
    out = subprocess.run(["bash", str(REPO / "benchmarks/single_node/agentic/serve_generic.sh")],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 7, out.stderr
    return (tmp_path / "seen").read_text()


def test_serve_generic_strips_slurm_env_for_trtllm_engine(tmp_path):
    # trtllm-serve spawns TP workers with MPI_Spawn; PRRTE must not see the
    # Slurm allocation env or it launches daemons via srun/ssh and the engine
    # dies with MPI_ERR_UNKNOWN. UCX must stay off the RoCE transport (the
    # runner's 8 MB memlock cannot hold its completion queue): shared memory
    # covers the same-node ranks.
    seen = _run_serve_generic_with_env_dump(tmp_path, "trtllm")
    assert "SLURM_" not in seen
    assert "UCX_TLS=sm,self" in seen


def test_serve_generic_keeps_slurm_env_for_other_frameworks(tmp_path):
    seen = _run_serve_generic_with_env_dump(tmp_path, "vllm")
    assert "SLURM_JOB_ID=6809" in seen
    assert "SLURM_JOB_NODELIST=hgx-h200-01" in seen
    assert "UCX_TLS" not in seen


def test_search_mode_drops_concurrency_and_writes_slo_override_copy(tmp_path):
    # AIPerf silently ignores CLI --search-sla when the config YAML carries a
    # sweep: block (spike notes SLO_OVERRIDE), so the IX_SLO override must be
    # written into a YAML copy in the run dir that --config points at.
    run_dir = tmp_path / "r"
    out = run(f'''
        export SERVING_SEARCH=1 AIPERF_PYTHON={sys.executable}
        export SERVING_SCENARIO=sweep-isl8k-osl512-cache50
        export IX_SLO="time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"
        mkdir -p {run_dir}
        build_replay_cmd_yaml {run_dir}
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    config_copy = run_dir / "scenario_slo_override.yaml"
    assert f" --config {config_copy} " in out.stdout
    assert " --concurrency " not in out.stdout
    assert "--search-sla" not in out.stdout
    doc = yaml.safe_load(config_copy.read_text())
    assert doc["sweep"]["sla_filters"] == [
        {"metric_tag": "time_to_first_token", "stat": "avg", "op": "lt", "threshold": 8000.0},
        {"metric_tag": "inter_token_latency", "stat": "p90", "op": "lt", "threshold": 50.0},
    ]


def test_search_mode_without_slo_override_keeps_scenario_config():
    out = run(r'''
        export SERVING_SEARCH=1
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert f" --config {REPO}/scenarios/agentx-cc-062126-smoke.yaml " in out.stdout
    assert "--search-sla" not in out.stdout
    assert " --concurrency " not in out.stdout


def test_search_mode_rejects_malformed_slo_before_any_run():
    out = run(f'''
        export SERVING_SEARCH=1 AIPERF_PYTHON={sys.executable}
        export IX_SLO="time_to_first_token:avg:lt"
        build_replay_cmd_yaml /tmp/r
    ''')
    assert out.returncode != 0
    assert "metric:stat:op:threshold" in out.stderr


def test_search_mode_unsets_conc_and_still_succeeds():
    out = run(r'''
        export SERVING_SEARCH=1
        unset CONC
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert " --concurrency " not in out.stdout


def test_non_search_mode_unchanged():
    out = run(r'''
        export SERVING_SEARCH=0
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert " --concurrency 8" in out.stdout


def test_empty_serving_search_is_rejected_like_unset():
    # benchmark-tmpl.yml must export SERVING_SEARCH as '0', never '': check_env_vars
    # rejects empty values, which would abort every non-sweep serving run.
    out = run(r'''
        export SERVING_SEARCH=
        build_replay_cmd_yaml /tmp/r
    ''')
    assert out.returncode != 0
    assert "SERVING_SEARCH" in out.stdout + out.stderr


# no_pass_in_range-shaped history: the SLA is physically unreachable, so the
# search stops after one probe; aiperf still exits nonzero on this state.
NO_PASS_HISTORY = json.dumps(
    {
        "boundary_summary": {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": None,
            "infeasible_min": {"value": 1, "iteration_idx": 0},
        },
        "convergence_reason": "smooth_isotonic_no_pass_in_range",
        "iterations": [{}],
    }
)


def _write_stub_aiperf(bin_dir: Path, rc: int = 0, history: str | None = "fixture") -> None:
    # Stub aiperf fakes the adaptive-search artifact layout (search_history.json
    # + the winning probe's export per the spike's ARTIFACT_LAYOUT) and exits
    # ``rc``. ``history``: "fixture" copies the committed fixture, other text is
    # written as-is, None writes no history at all (crash simulation).
    fix = REPO / "utils" / "fixtures" / "sweep"
    if history == "fixture":
        write_history = f'cp {fix}/search_history.json "$out/search_history.json"\n'
    elif history is not None:
        write_history = f'printf %s {shlex.quote(history)} > "$out/search_history.json"\n'
    else:
        write_history = ""
    stub = bin_dir / "aiperf"
    stub.write_text(
        "#!/bin/sh\n"
        'out=""\nprev=""\n'
        'for a in "$@"; do\n'
        '  [ "$prev" = "--output-artifact-dir" ] && out="$a"\n'
        '  prev="$a"\n'
        "done\n"
        '[ -n "$out" ] || exit 2\n'
        f'mkdir -p "$out/search_iter_0008/profile_runs/run_0001"\n'
        + write_history
        + (
            f'cp {fix}/profile_export_aiperf_probe_concurrency256.json '
            '"$out/search_iter_0008/profile_runs/run_0001/profile_export_aiperf.json"\n'
            if history == "fixture"
            else ""
        )
        + f"exit {rc}\n"
    )
    stub.chmod(0o755)


def _run_search_replay(tmp_path: Path, extra_env: str = "", rc: int = 0, history: str | None = "fixture"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_stub_aiperf(bin_dir, rc=rc, history=history)
    run_dir = tmp_path / "r"
    out_dir = tmp_path / "agg"
    out = run(f'''
        export SERVING_SEARCH=1 AIPERF_PYTHON={sys.executable}
        export ENABLE_AGENTX_POWER=0 IS_MULTINODE=false REQUIRE_POWER=0
        export PATH={bin_dir}:$PATH
        {extra_env}
        mkdir -p {run_dir}
        build_replay_cmd_yaml {run_dir}
        RESULT_FILENAME=sweep_test AGENTIC_OUTPUT_DIR={out_dir} \
            run_agentic_replay_and_write_outputs {run_dir}
    ''')
    return out, out_dir


def test_search_replay_writes_sweep_result_json(tmp_path):
    # End-to-end hook check: the replay runner must route through
    # infx.results.sweep instead of the per-request validators.
    out, out_dir = _run_search_replay(tmp_path)
    assert out.returncode == 0, out.stderr
    result = json.loads((out_dir / "sweep_test.json").read_text())
    assert result["state"] == "ok"
    assert result["max_conc"] == 256


def test_search_replay_skips_required_server_metrics_gate(tmp_path):
    # serve_generic.sh exports AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang: by
    # framework, but an adaptive_search artifact layout has no top-level
    # server_metrics_export.{json,csv} — the opt-in gate must not fire.
    out, out_dir = _run_search_replay(
        tmp_path, 'export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"\n'
    )
    assert out.returncode == 0, out.stderr
    assert "required AIPerf server metrics artifacts" not in out.stderr
    assert (out_dir / "sweep_test.json").exists()


def test_server_metrics_gate_still_fires_without_search():
    # Pin the gate itself: non-search mode, prefix set, no server metrics
    # artifacts -> nonzero with the gate's error (not neutered by the skip).
    out = run(r'''
        export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
        mkdir -p /tmp/ix-gate-check/aiperf_artifacts
        validate_required_agentic_server_metrics /tmp/ix-gate-check
    ''')
    assert out.returncode != 0
    assert "required AIPerf server metrics artifacts" in out.stderr


def test_search_replay_stays_green_on_terminal_no_pass_state(tmp_path):
    # AIPerf exits nonzero when no probe passes the SLA, but that terminal
    # state (no_pass_in_range) is a result field, never a failed job: the run
    # must stay green because a parseable search_history.json exists.
    out, out_dir = _run_search_replay(tmp_path, rc=1, history=NO_PASS_HISTORY)
    assert out.returncode == 0, out.stderr
    assert "[sweep] aiperf exited nonzero on a terminal search state" in out.stdout
    result = json.loads((out_dir / "sweep_test.json").read_text())
    assert result["state"] == "no_pass_in_range"
    assert result["max_conc"] is None


def test_search_replay_crash_without_history_still_fails(tmp_path):
    # Genuine crash: nonzero aiperf exit AND no search_history.json -> the run
    # fails with the existing replay error.
    out, out_dir = _run_search_replay(tmp_path, rc=1, history=None)
    assert out.returncode != 0
    assert "exited with code 1" in out.stderr
    assert not (out_dir / "sweep_test.json").exists()
