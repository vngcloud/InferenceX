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


# --------------------------------------------------------------------------
# Cancellation safety: the replay must not outlive the shell that started it.
#
# Run 30508401430 was cancelled at 02:33 UTC and `aiperf system_controller`
# (reparented to PID 1) kept replaying against the production gateway for
# another 35 minutes. Two things had to be true for that to happen: the replay
# ran as a foreground pipeline (bash defers traps until the foreground command
# finishes, so no handler could fire until the full --benchmark-duration
# elapsed), and nothing recorded a PID a later cleanup could act on. These
# tests pin both fixes.
# --------------------------------------------------------------------------

_REPLAY_HARNESS = r'''
    export INFMAX_CONTAINER_WORKSPACE=/tmp
    source benchmarks/benchmark_lib.sh
    AIPERF_PYTHON=true
    write_agentic_result_json() { :; }
'''


def test_replay_records_a_live_pid_while_running(tmp_path) -> None:
    # The stub polls for the state file because the parent writes it just after
    # forking the replay -- asserting on first look would be a race, not a
    # regression. `kill -0` on the recorded PID is the real assertion: a file
    # holding a stale or bogus number is worse than no file, since the reaper
    # would then skip a live tree.
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    marker = tmp_path / "stub-checked"
    result = _run_bash(_REPLAY_HARNESS + r'''
        fake_aiperf() {
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                [ -s "$AIPERF_REPLAY_PID_FILE" ] && break
                sleep 0.5
            done
            pid=$(head -1 "$AIPERF_REPLAY_PID_FILE")
            kill -0 "$pid" || return 1
            grep -q "conc=" "$AIPERF_REPLAY_PID_FILE" || return 1
            touch "''' + str(marker) + r'''"
        }
        AIPERF_CLI=fake_aiperf
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
    ''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists(), "replay stub never saw a live PID in the state file"


def test_replay_clears_its_pid_file_on_clean_exit(tmp_path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = _run_bash(_REPLAY_HARNESS + r'''
        AIPERF_CLI=true
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
        [ ! -f "$AIPERF_REPLAY_PID_FILE" ]
        [ -z "$AIPERF_REPLAY_PID" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_exit_code_still_propagates(tmp_path) -> None:
    # Guards the PIPESTATUS[0] -> `wait` swap: a failing replay must still be
    # reported as a failure after the outputs are written.
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = _run_bash(_REPLAY_HARNESS + r'''
        AIPERF_CLI=false
        build_replay_cmd "''' + str(result_dir) + r'''"
        # `|| rc=$?`, not a `set +e` / `$?` pair: the function re-enables set -e
        # internally (pre-existing behaviour), which would undo a caller's
        # `set +e` and abort this script at the call site.
        rc=0
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''" || rc=$?
        [ "$rc" = "1" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exited with code 1" in result.stdout + result.stderr


def test_replay_restores_a_recipe_signal_handler(tmp_path) -> None:
    # Single-node agentic recipes install `trap 'exit 143' TERM` so their EXIT
    # trap can tear down a local server. Arming the replay's own handler must
    # not leave that disarmed for the rest of the run.
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = _run_bash(_REPLAY_HARNESS + r'''
        trap 'exit 143' TERM
        AIPERF_CLI=true
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
        # Command substitution, not `trap -p TERM | grep`: a pipeline runs its
        # components in subshells, where bash resets non-ignored traps to
        # default, so the piped form reports nothing whatever the state is.
        [[ "$(trap -p TERM)" == *"exit 143"* ]]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_leaves_no_trap_when_recipe_had_none(tmp_path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = _run_bash(_REPLAY_HARNESS + r'''
        AIPERF_CLI=true
        build_replay_cmd "''' + str(result_dir) + r'''"
        run_agentic_replay_and_write_outputs "''' + str(result_dir) + r'''"
        [ -z "$(trap -p TERM)" ]
        [ -z "$(trap -p INT)" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr


# The orphan the reaper has to handle is a tree, not a process: the observed
# leak was `aiperf system_controller` plus 16 children (8 workers at CCU8, 5
# managers, 2 record processors). The stub therefore spawns children of its own,
# and the name carries "aiperf" because that substring is the recycled-PID veto
# -- ps reads /proc/pid/cmdline, which setproctitle rewrites, so a live aiperf
# tree still answers to it.
_ORPHAN_STUB = r'''
    mkdir -p "$FAKE_DIR"
    cat > "$FAKE_DIR/aiperf" <<'STUB'
#!/usr/bin/env bash
sleep 300 &
echo $! >> "$CHILD_PIDS"
sleep 300 &
echo $! >> "$CHILD_PIDS"
sleep 300
STUB
    chmod +x "$FAKE_DIR/aiperf"
    "$FAKE_DIR/aiperf" &
    ROOT_PID=$!
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        [ "$(wc -l < "$CHILD_PIDS" 2>/dev/null || echo 0)" -ge 2 ] && break
        sleep 0.5
    done
'''


def test_reaper_kills_an_orphaned_replay_tree(tmp_path) -> None:
    state_dir = tmp_path / "inferencex-agentic-orphan"
    state_dir.mkdir()
    child_pids = tmp_path / "child-pids"
    child_pids.touch()
    result = _run_bash(r'''
        export TMPDIR="''' + str(tmp_path) + r'''"
        export FAKE_DIR="''' + str(tmp_path / "bin") + r'''"
        export CHILD_PIDS="''' + str(child_pids) + r'''"
        source benchmarks/benchmark_lib.sh
    ''' + _ORPHAN_STUB + r'''
        printf '%s\nrun=test job=test conc=8 duration=3600\n' "$ROOT_PID" \
            > "''' + str(state_dir / "aiperf-replay.pid") + r'''"
        reap_orphan_agentic_replays
        ! kill -0 "$ROOT_PID" 2>/dev/null
        while read -r child; do
            ! kill -0 "$child" 2>/dev/null || { echo "child $child survived" >&2; exit 1; }
        done < "$CHILD_PIDS"
        [ ! -f "''' + str(state_dir / "aiperf-replay.pid") + r'''" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reaping orphaned replay tree" in result.stdout


def test_reaper_spares_a_recycled_pid(tmp_path) -> None:
    # A stale state file must never be a licence to kill whatever inherited the
    # number. The state file is still cleared, so the next sweep is quiet.
    state_dir = tmp_path / "inferencex-agentic-recycled"
    state_dir.mkdir()
    state_file = state_dir / "aiperf-replay.pid"
    result = _run_bash(r'''
        export TMPDIR="''' + str(tmp_path) + r'''"
        source benchmarks/benchmark_lib.sh
        sleep 300 &
        INNOCENT=$!
        printf '%s\nrun=test job=test conc=8 duration=3600\n' "$INNOCENT" \
            > "''' + str(state_file) + r'''"
        reap_orphan_agentic_replays
        kill -0 "$INNOCENT" || { echo "innocent process was killed" >&2; exit 1; }
        kill "$INNOCENT" 2>/dev/null || true
        [ ! -f "''' + str(state_file) + r'''" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "is not aiperf" in result.stdout


def test_reaper_discards_a_malformed_state_file(tmp_path) -> None:
    state_dir = tmp_path / "inferencex-agentic-garbage"
    state_dir.mkdir()
    state_file = state_dir / "aiperf-replay.pid"
    state_file.write_text("not-a-pid\n")
    result = _run_bash(r'''
        export TMPDIR="''' + str(tmp_path) + r'''"
        source benchmarks/benchmark_lib.sh
        reap_orphan_agentic_replays
        [ ! -f "''' + str(state_file) + r'''" ]
    ''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "malformed" in result.stdout


def test_reaper_script_is_wired_into_shared_resource_cleanup() -> None:
    # The pre-run and post-run steps share the &resource-cleanup anchor, so one
    # call covers both hooks. The post-run copy is the layer that survives a
    # SIGKILL of the benchmark shell, and it only runs on cancellation because
    # of `if: always()` -- assert all three facts, not just the call.
    script = Path("runners/reap_orphan_aiperf.sh")
    assert script.exists()
    body = script.read_text()
    assert "reap_orphan_agentic_replays" in body
    assert "IS_AGENTIC=0" in body, "sourcing the lib un-gated can exit 1 mid-cleanup"

    tmpl = Path(".github/workflows/benchmark-tmpl.yml").read_text()
    assert "bash runners/reap_orphan_aiperf.sh || true" in tmpl
    assert "run: &resource-cleanup |" in tmpl
    assert "run: *resource-cleanup" in tmpl
    post = tmpl.split("Resource cleanup (post-run)", 1)[1]
    assert "if: always()" in post.split("run:", 1)[0]


def test_sigterm_during_replay_kills_the_whole_tree(tmp_path) -> None:
    # The claim the whole fix rests on. Previously the replay ran as a
    # foreground pipeline, so bash deferred the trap until it finished -- a
    # SIGTERM at cancellation time did nothing until --benchmark-duration
    # elapsed. Backgrounding it and blocking in `wait` makes the handler run
    # immediately; the stub spawns children so this also covers the descendant
    # snapshot (the observed leak was 16 children reparented to PID 1).
    import os
    import signal
    import time

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    children_file = tmp_path / "children"
    children_file.touch()

    stub = tmp_path / "aiperf"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'sleep 300 & echo $! >> "{children_file}"\n'
        f'sleep 300 & echo $! >> "{children_file}"\n'
        "sleep 300\n"
    )
    stub.chmod(0o755)

    driver = tmp_path / "driver.sh"
    driver.write_text(
        "set -e\n" + _AGENTIC_ENV + _REPLAY_HARNESS
        + f'AIPERF_CLI={stub}\n'
        + f'build_replay_cmd "{result_dir}"\n'
        + f'run_agentic_replay_and_write_outputs "{result_dir}"\n'
    )

    # TMPDIR points AIPERF_RUNTIME_DIR (and so the state file) inside this
    # test's own directory, so the glob below cannot pick up another run's file.
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    proc = subprocess.Popen(["bash", str(driver)], cwd=Path.cwd(), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 20
        pid_files: list[Path] = []
        while time.time() < deadline:
            pid_files = [Path(p) for p in
                         glob.glob(str(tmp_path / "inferencex-agentic-*/aiperf-replay.pid"))]
            if pid_files and children_file.read_text().count("\n") >= 2:
                break
            time.sleep(0.25)
        assert pid_files, "replay never recorded a PID file"

        root_pid = int(pid_files[0].read_text().splitlines()[0])
        child_pids = [int(line) for line in children_file.read_text().split()]
        assert len(child_pids) >= 2

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=40) == 143

        def alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            return True

        deadline = time.time() + 20
        while time.time() < deadline and any(alive(p) for p in [root_pid, *child_pids]):
            time.sleep(0.25)

        survivors = [p for p in [root_pid, *child_pids] if alive(p)]
        assert not survivors, f"replay tree survived SIGTERM: {survivors}"
        assert not pid_files[0].exists(), "state file left behind after a clean teardown"
    finally:
        if proc.poll() is None:
            proc.kill()
        for line in children_file.read_text().split():
            try:
                os.kill(int(line), signal.SIGKILL)
            except (OSError, ValueError):
                pass
