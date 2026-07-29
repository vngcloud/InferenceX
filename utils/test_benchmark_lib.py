import glob
import subprocess
from pathlib import Path

import pytest


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
    # bare `echo "$REPLAY_CMD" >` or the `set -x` guard is removed. `set -x`
    # is enabled just before the call (and restored after) to mirror what
    # every real *-remote-bench.sh recipe does -- xtrace is already ON by the
    # time this function runs in production, so a test that never turns it on
    # cannot exercise the guard at all and would pass even if the guard were
    # a no-op. AIPERF_CLI and AIPERF_PYTHON are shadowed to the `true` builtin, *after* sourcing
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
        set -x
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
        set +x
    '''
    result = subprocess.run(["bash", "-c", "set -e\n" + script],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    cmd_text = (result_dir / "benchmark_command.txt").read_text()
    log_text = (result_dir / "benchmark.log").read_text()

    assert "sk-sentinel-do-not-leak" not in cmd_text
    assert "$REMOTE_API_KEY" in cmd_text
    assert "sk-sentinel-do-not-leak" not in log_text
    # A bare (unguarded) `set -x` doesn't land its trace in benchmark.log on
    # this shell (bash 3.2) -- the trace goes to the shell's own stderr, i.e.
    # the CI job log, not through the pipeline's own 2>&1 into tee. GitHub
    # Actions masking only covers stderr, and only for values registered as
    # secrets, so this guard is defense in depth, not the primary control
    # (benchmark_command.txt's redaction is). Check the combined stdout+stderr
    # here, not just the two files, so this regresses regardless of which
    # stream a given bash version actually routes the xtrace line to.
    assert "sk-sentinel-do-not-leak" not in result.stdout + result.stderr


def test_aiperf_tokenizer_is_passed_when_set() -> None:
    result = _run_bash(r'''
        export AIPERF_TOKENIZER=zai-org/GLM-5.2-FP8
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" == *' --tokenizer zai-org/GLM-5.2-FP8'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_replay_cmd_omits_tokenizer_when_unset() -> None:
    result = _run_bash(r'''
        source benchmarks/benchmark_lib.sh
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" != *' --tokenizer '* ]]
        [[ "$REPLAY_CMD" == *' --tokenizer-trust-remote-code'* ]]
    ''')
    assert result.returncode == 0, result.stderr


_CURL_STUB_OK = r"""
    curl() {
        for a in "$@"; do
            if [[ "$a" == "--write-out" ]]; then printf '200'; return 0; fi
        done
        return 0
    }
"""

_REMOTE_ENV = r"""
    export REMOTE_BASE_URL=https://gw.example.test/v1
    export REMOTE_ENGINE_METRICS_URL=https://m.example.test/worker-metrics
    export REMOTE_RUNNER_TYPE=h200-nv
"""


def test_preflight_strips_trailing_v1_from_base_url() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        [[ "$REMOTE_BASE_URL" == "https://gw.example.test" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_runs_without_gpu_telemetry() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        [[ -z "${AIPERF_GPU_TELEMETRY_URL:-}" ]]
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" == *' --no-gpu-telemetry'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_omits_max_context_length_when_unset() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" != *' --max-context-length'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_sets_max_context_length_when_supplied() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_MAX_CONTEXT_LENGTH=131072
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        build_replay_cmd /tmp/aiperf-test
        [[ "$REPLAY_CMD" == *' --max-context-length 131072'* ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_exports_runner_type_and_tokenizer() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_TOKENIZER=zai-org/GLM-5.2-FP8
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        [[ "$RUNNER_TYPE" == "h200-nv" ]]
        [[ "$AIPERF_TOKENIZER" == "zai-org/GLM-5.2-FP8" ]]
        [[ "$AIPERF_SERVER_METRICS_URLS" == "https://m.example.test/worker-metrics" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_fails_on_rejected_api_key() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_API_KEY=sk-wrong
        source benchmarks/benchmark_lib.sh
        curl() {
            for a in "$@"; do
                if [[ "$a" == "--write-out" ]]; then printf '401'; return 0; fi
            done
            return 0
        }
        remote_bench_preflight
    ''')
    assert result.returncode != 0
    assert "HTTP 401" in result.stderr


def test_preflight_fails_when_supplied_gpu_telemetry_is_unreachable() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_GPU_TELEMETRY_URL=https://dcgm.example.test/metrics
        source benchmarks/benchmark_lib.sh
        curl() {
            for a in "$@"; do
                if [[ "$a" == "https://dcgm.example.test/metrics" ]]; then return 7; fi
                if [[ "$a" == "--write-out" ]]; then printf '200'; return 0; fi
            done
            return 0
        }
        remote_bench_preflight
    ''')
    assert result.returncode != 0
    assert "REMOTE_GPU_TELEMETRY_URL" in result.stderr


@pytest.mark.parametrize(
    "input_url,expected_url",
    [
        ("https://host/v1", "https://host"),
        ("https://host/v1/", "https://host"),
        ("https://host", "https://host"),
        ("https://host/", "https://host"),
        ("https://host/glm/sglang", "https://host/glm/sglang"),
    ],
)
def test_preflight_normalizes_base_url(input_url: str, expected_url: str) -> None:
    result = _run_bash(r'''
        export REMOTE_BASE_URL=''' + input_url + r'''
        export REMOTE_ENGINE_METRICS_URL=https://m.example.test/worker-metrics
        export REMOTE_RUNNER_TYPE=h200-nv
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        [[ "$REMOTE_BASE_URL" == "''' + expected_url + r'''" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_authenticated_probe_suppresses_xtrace() -> None:
    # Mirrors the Task 1 end-to-end redaction test: check the combined
    # stdout+stderr of a run with the recipe's own `set -x` active, not just
    # one stream, since GitHub Actions masking only covers stderr and only
    # for registered secrets -- this guard (and this test) is defense in
    # depth, not the primary control.
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_API_KEY=sk-sentinel-do-not-leak
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        set -x
        remote_bench_preflight
        set +x
    ''')
    assert result.returncode == 0, result.stderr
    assert "sk-sentinel-do-not-leak" not in result.stdout + result.stderr


REMOTE_BENCH_RECIPES = sorted(
    glob.glob("benchmarks/single_node/agentic/*-remote-bench.sh")
)


def test_glm52_fp8_remote_bench_recipe_exists() -> None:
    assert (
        "benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh"
        in REMOTE_BENCH_RECIPES
    )


def test_remote_bench_recipes_use_shared_preflight() -> None:
    assert REMOTE_BENCH_RECIPES, "no remote-bench recipes found"
    for path in REMOTE_BENCH_RECIPES:
        script = Path(path).read_text()
        assert "remote_bench_preflight" in script, path
        # Optional vars must not be in check_env_vars, or a target without
        # DCGM or without a context cap fails before preflight can run.
        required = script.split("check_env_vars", 1)[1].split("mkdir", 1)[0]
        assert "REMOTE_GPU_TELEMETRY_URL" not in required, path
        assert "REMOTE_MAX_CONTEXT_LENGTH" not in required, path
        assert "REMOTE_ENGINE_METRICS_URL" in required, path


# Logic that now lives exclusively in remote_bench_preflight(). A recipe that
# calls remote_bench_preflight but ALSO keeps one of these would double-probe
# endpoints and clobber exports the shared function already set correctly --
# the exact regression this consolidation exists to make impossible. Header
# comments legitimately mention some of these names by way of explanation, so
# comment-only lines are stripped before matching.
_FORBIDDEN_INLINE_PREFLIGHT_LOGIC = (
    "curl",
    "/health",
    "export AIPERF_",
    "export RUNNER_TYPE",
    "export MAX_MODEL_LEN",
)


def test_remote_bench_recipes_do_not_reimplement_preflight_logic() -> None:
    assert REMOTE_BENCH_RECIPES, "no remote-bench recipes found"
    for path in REMOTE_BENCH_RECIPES:
        lines = Path(path).read_text().splitlines()
        code = "\n".join(
            line for line in lines if not line.lstrip().startswith("#")
        )
        for needle in _FORBIDDEN_INLINE_PREFLIGHT_LOGIC:
            assert needle not in code, (
                f"{path} re-implements preflight logic: found {needle!r} "
                "outside a comment; this belongs exclusively in "
                "remote_bench_preflight()"
            )


def test_benchmark_tmpl_wires_the_api_key_secret() -> None:
    tmpl = Path(".github/workflows/benchmark-tmpl.yml").read_text()
    assert "REMOTE_API_KEY: ${{ secrets.GREENNODE_API_KEY }}" in tmpl
    assert "REMOTE_TOKENIZER: ${{ inputs.remote-tokenizer }}" in tmpl
    assert "remote-tokenizer:" in tmpl


def test_remote_bench_e2e_does_not_hardcode_kv_offloading() -> None:
    wf = Path(".github/workflows/remote-bench-e2e.yml").read_text()
    assert "kv-offloading: none" not in wf
    assert "kv-offloading: ${{ matrix.config.kv-offloading || 'none' }}" in wf
    assert "kv-offload-backend: ${{ matrix.config.kv-offload-backend || '' }}" in wf
    assert "remote-tokenizer: ${{ matrix.config.remote-tokenizer || '' }}" in wf


# A Kubernetes ingress serves a self-signed "Fake Certificate" until a real
# one is installed, so curl --fail dies with exit 60 and aiperf's own scrape
# would fail identically later. REMOTE_INSECURE_TLS has to cover both, and
# must stay off unless explicitly requested.
_CURL_STUB_RECORDING = r"""
    CURL_LOG=$(mktemp)
    curl() {
        printf '%s\n' "$*" >> "$CURL_LOG"
        for a in "$@"; do
            if [[ "$a" == "--write-out" ]]; then printf '200'; return 0; fi
        done
        return 0
    }
"""


def test_preflight_passes_insecure_to_curl_when_opted_in() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_INSECURE_TLS=true
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_RECORDING + r'''
        remote_bench_preflight
        # Every probe must carry it, not merely one of them: a partial
        # rollout still dies on whichever endpoint was left verifying.
        [[ -s "$CURL_LOG" ]]
        ! grep -qv -- "--insecure" "$CURL_LOG"
        grep -- "--insecure" "$CURL_LOG" | grep -q "worker-metrics"
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_disables_aiperf_ssl_verify_when_opted_in() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_INSECURE_TLS=true
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_OK + r'''
        remote_bench_preflight
        [[ "$AIPERF_HTTP_SSL_VERIFY" == "false" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_verifies_tls_by_default() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_RECORDING + r'''
        remote_bench_preflight
        ! grep -q -- "--insecure" "$CURL_LOG"
        [[ -z "${AIPERF_HTTP_SSL_VERIFY:-}" ]]
    ''')
    assert result.returncode == 0, result.stderr


def test_preflight_treats_non_true_insecure_flag_as_off() -> None:
    result = _run_bash(_REMOTE_ENV + r'''
        export REMOTE_INSECURE_TLS=false
        source benchmarks/benchmark_lib.sh
    ''' + _CURL_STUB_RECORDING + r'''
        remote_bench_preflight
        ! grep -q -- "--insecure" "$CURL_LOG"
    ''')
    assert result.returncode == 0, result.stderr


def test_workflows_wire_the_insecure_tls_flag() -> None:
    tmpl = Path(".github/workflows/benchmark-tmpl.yml").read_text()
    assert "remote-insecure-tls:" in tmpl
    assert "REMOTE_INSECURE_TLS: ${{ inputs.remote-insecure-tls }}" in tmpl
    wf = Path(".github/workflows/remote-bench-e2e.yml").read_text()
    assert "remote-insecure-tls: ${{ matrix.config.remote-insecure-tls || false }}" in wf
