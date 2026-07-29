import subprocess
from pathlib import Path


def test_greennode_launcher_forwards_kv_backend_metadata() -> None:
    script = Path("runners/launch_h200-greennode.sh").read_text()
    run_env = script.split("RUN_ENV=(", 1)[1].split(")", 1)[0]

    assert "KV_OFFLOAD_BACKEND_METADATA" in run_env.split()
    assert "HICACHE_RATIO" in run_env.split()


def test_agentic_gpu_telemetry_opt_in() -> None:
    script = r'''
        set -e
        export IS_AGENTIC=1 KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache
        export TOTAL_CPU_DRAM_GB=128
        export AIPERF_CLI=aiperf MODEL=model CONC=2 DURATION=90
        export FRAMEWORK=sglang TRACE_SOURCE_FLAG='--public-dataset dataset'
        export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" == *' --gpu-telemetry http://localhost:9400/metrics'* ]]
        [[ "$REPLAY_CMD" != *' --no-gpu-telemetry'* ]]
    '''
    subprocess.run(["bash", "-c", script], check=True)


_AGENTIC_ENV = r"""
    export IS_AGENTIC=1 KV_OFFLOADING=none
    export AIPERF_CLI=aiperf MODEL=model CONC=2 DURATION=90
    export FRAMEWORK=sglang TRACE_SOURCE_FLAG='--public-dataset dataset'
"""


def _run_bash(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", "set -e\n" + _AGENTIC_ENV + body],
                          capture_output=True, text=True)


def test_remote_api_key_is_passed_to_aiperf() -> None:
    result = _run_bash(r'''
        export REMOTE_API_KEY=sk-sentinel-do-not-leak
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" == *' --api-key sk-sentinel-do-not-leak'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_replay_cmd_omits_api_key_when_unset() -> None:
    result = _run_bash(r'''
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" != *' --api-key'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_remote_api_key_with_whitespace_is_rejected() -> None:
    result = _run_bash(r'''
        export REMOTE_API_KEY='sk-has a-space'
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
    ''')
    assert result.returncode != 0
    assert "must not contain whitespace" in result.stderr


def test_remote_api_key_is_redacted_from_benchmark_command() -> None:
    result = _run_bash(r'''
        export REMOTE_API_KEY=sk-sentinel-do-not-leak
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        redacted=$(redact_replay_cmd "$REPLAY_CMD")
        [[ "$redacted" != *sk-sentinel-do-not-leak* ]]
        [[ "$redacted" == *'--api-key $REMOTE_API_KEY'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_redact_replay_cmd_is_a_noop_without_a_key() -> None:
    result = _run_bash(r'''
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$(redact_replay_cmd "$REPLAY_CMD")" == "$REPLAY_CMD" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_run_agentic_replay_and_write_outputs_redacts_key_end_to_end(tmp_path) -> None:
    # Exercises the real write site and xtrace guard inside
    # run_agentic_replay_and_write_outputs (not redact_replay_cmd in
    # isolation), so this fails if benchmark_command.txt's write reverts to a
    # bare `echo "$REPLAY_CMD" >` or the `set -x` guard is removed. AIPERF_CLI
    # and AIPERF_PYTHON are shadowed to the `true` builtin, *after* sourcing
    # (benchmark_lib.sh unconditionally sets both to an AIPERF_VENV path at
    # source time, so pre-source exports get clobbered), so the replay,
    # analysis, and validation subprocess calls are all no-ops requiring no
    # network or real aiperf/python install; write_agentic_result_json does
    # real cd + module-path work in a subshell before invoking $AIPERF_PYTHON,
    # so it is shadowed directly as a no-op function.
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    script = _AGENTIC_ENV + r'''
        export INFMAX_CONTAINER_WORKSPACE=/tmp
        export REMOTE_API_KEY=sk-sentinel-do-not-leak
        source benchmarks/benchmark_lib.sh
        AIPERF_CLI=true
        AIPERF_PYTHON=true
        write_agentic_result_json() { :; }
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
    '''
    result = subprocess.run(["bash", "-c", "set -e\n" + script],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    cmd_text = (result_dir / "benchmark_command.txt").read_text()
    log_text = (result_dir / "benchmark.log").read_text()

    assert "sk-sentinel-do-not-leak" not in cmd_text
    assert "$REMOTE_API_KEY" in cmd_text
    assert "sk-sentinel-do-not-leak" not in log_text
    # On this shell, a bare (unguarded) `set -x` traces the pipeline to the
    # *script's own* stderr rather than into benchmark.log (verified by
    # temporarily removing the guard and observing where the trace landed) --
    # so the guard is checked here too, against the full captured output, to
    # catch that regression regardless of which stream a given bash version
    # routes the xtrace line to.
    assert "sk-sentinel-do-not-leak" not in result.stdout + result.stderr
