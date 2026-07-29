# Remote-Bench Authenticated Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let InferenceX `remote-bench` benchmark an API-key-protected, externally-managed OpenAI-compatible gateway using the same aiperf agentic command it runs natively, without the key ever reaching the repository, an uploaded artifact, or a log file.

**Architecture:** All shared remote-bench logic moves into one new `remote_bench_preflight()` function in `benchmarks/benchmark_lib.sh`; the three `*-remote-bench.sh` recipes shrink to ~8 lines each and call it. `build_replay_cmd` gains two conditional flags (`--api-key`, `--tokenizer`) that are absent unless the corresponding env var is set, so native recipes emit a byte-identical command. The key travels GitHub secret → workflow env → process env → aiperf argv, and is redacted at the one place InferenceX writes it to disk.

**Tech Stack:** Bash 4+ (`benchmark_lib.sh`), GitHub Actions reusable workflows (YAML), pytest driving bash via `subprocess.run` (`utils/test_benchmark_lib.py`), aiperf CLI.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-29-remote-bench-authenticated-gateway-design.md`:

- The aiperf command deviates from native agentic in exactly two places, both operator-controlled: `--max-context-length` and `--unsafe-override`. Everything else stays byte-identical — `--scenario inferencex-agentx-mvp`, `--random-seed 42`, `--num-dataset-entries 393`, `--agentic-cache-warmup-duration 600`, the trajectory-start ratios, `--use-server-token-count`, `--slice-duration 1.0`, `--tokenizer-trust-remote-code`, `--failed-request-threshold`.
- `--tokenizer` is added to the builder but emitted only when `AIPERF_TOKENIZER` is set, which native recipes never do, so their command is unchanged.
- Engine metrics stay required; GPU telemetry becomes optional.
- The secret name is `GREENNODE_API_KEY` (repo secret in `vngcloud/InferenceX`). One secret; do not add a second.
- `$REPLAY_CMD` is word-split at exec. **Never quote a value appended to `REPLAY_CMD`** — quotes would be passed through as literal characters in the argv element. Existing lines like `REPLAY_CMD+=" --url ${REMOTE_BASE_URL:-http://localhost:$PORT}"` show the required style.
- Run all tests from the **repository root**, not from `utils/`: `pytest utils/test_benchmark_lib.py -v`. The existing tests in that file use paths relative to the repo root.
- Non-goal: auto-dispatching remote-bench results to InferenceX-app production ingest. Do not add a `trigger-agentic-ingest` job to `remote-bench-e2e.yml`.

**One deliberate deviation from the spec.** The spec proposed a standalone `benchmarks/test_remote_bench_secret.sh`. This plan puts every test in `utils/test_benchmark_lib.py` instead, because that file already exists and already tests `build_replay_cmd` by sourcing the bash lib from Python via `subprocess.run` — the same technique, in the established location. No new test harness is introduced.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `benchmarks/benchmark_lib.sh` | modify | `build_replay_cmd` gains `--api-key`/`--tokenizer`; new `redact_replay_cmd()` and `remote_bench_preflight()`; `run_agentic_replay_and_write_outputs` redacts before writing |
| `utils/test_benchmark_lib.py` | modify | All tests for the above, plus workflow-wiring assertions |
| `benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh` | create | fp8 GLM-5.2 recipe (the launcher derives the filename from `PRECISION`, so fp4's file cannot be reused) |
| `benchmarks/single_node/agentic/glm5.2_fp4_sglang-remote-bench.sh` | modify | reduced to call `remote_bench_preflight()` |
| `benchmarks/single_node/agentic/dsv2lite_fp8_sglang-remote-bench.sh` | modify | reduced to call `remote_bench_preflight()` |
| `.github/workflows/benchmark-tmpl.yml` | modify | `REMOTE_API_KEY` env from the secret; new `remote-tokenizer` input |
| `.github/workflows/remote-bench-e2e.yml` | modify | stop hardcoding `kv-offloading: none`; pass tokenizer and KV-offload fields from the matrix |
| `.claude/skills/create-remote-bench/SKILL.md` | modify | scrape-first serving-config interview; optional telemetry and context cap; auth section |

---

### Task 1: API key reaches aiperf, never reaches an artifact

This is the security-critical task. aiperf has no env-var binding for `api_key`, so the key must be an argv element. GitHub Actions masks secrets in **job logs** but not in **uploaded artifact files**, so `benchmark_command.txt` must be redacted explicitly, and shell xtrace must be suppressed because `$REPLAY_CMD 2>&1 | tee benchmark.log` redirects the shell's stderr into the log file.

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (inside `build_replay_cmd`, which starts at line 1736; and `run_agentic_replay_and_write_outputs`, which starts at line 1895)
- Test: `utils/test_benchmark_lib.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `redact_replay_cmd <command-string>` — prints the string to stdout with every occurrence of `$REMOTE_API_KEY`'s value replaced by the literal text `$REMOTE_API_KEY`; a no-op when `REMOTE_API_KEY` is unset or empty. Reads `REMOTE_API_KEY` from the environment. Task 3 and Task 4 rely on `REMOTE_API_KEY` being the env var name.

- [ ] **Step 1: Write the failing tests**

Append to `utils/test_benchmark_lib.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from the repo root:

```bash
pytest utils/test_benchmark_lib.py -v
```

Expected: the two `redact_replay_cmd` tests fail with `redact_replay_cmd: command not found`; `test_remote_api_key_is_passed_to_aiperf` and `test_remote_api_key_with_whitespace_is_rejected` fail because no `--api-key` is emitted and no whitespace guard exists. `test_replay_cmd_omits_api_key_when_unset` and `test_redact_replay_cmd_is_a_noop_without_a_key` may already pass — that is fine, they are regression guards.

- [ ] **Step 3: Add the `--api-key` flag to `build_replay_cmd`**

In `benchmarks/benchmark_lib.sh`, immediately after the existing line `REPLAY_CMD+=" --model $MODEL"`, insert:

```bash
    # Authenticated targets (e.g. a managed MaaS gateway). aiperf turns this
    # into "Authorization: Bearer <key>". aiperf has no env-var transport for
    # api_key, so it has to be an argv element; redact_replay_cmd keeps it out
    # of benchmark_command.txt, which is uploaded as an artifact and is not
    # masked by GitHub Actions the way job logs are.
    if [ -n "${REMOTE_API_KEY:-}" ]; then
        # $REPLAY_CMD is word-split at exec, so a key containing whitespace
        # would split into separate argv elements and authenticate with a
        # truncated credential. Fail loudly rather than send a broken header.
        if [[ "$REMOTE_API_KEY" == *[[:space:]]* ]]; then
            echo "ERROR: REMOTE_API_KEY must not contain whitespace" >&2
            return 1
        fi
        REPLAY_CMD+=" --api-key $REMOTE_API_KEY"
    fi
```

Note the deliberate absence of quotes around `$REMOTE_API_KEY` in the `REPLAY_CMD+=` line — see Global Constraints.

- [ ] **Step 4: Add `redact_replay_cmd`**

In `benchmarks/benchmark_lib.sh`, immediately **before** the `run_agentic_replay_and_write_outputs()` definition, insert:

```bash
# Strip REMOTE_API_KEY out of a command string before it is written anywhere
# that gets uploaded. GitHub Actions masks registered secrets in job logs but
# not inside artifact files, so benchmark_command.txt needs this explicitly.
# No-op when no key is set, which is every native (non-remote) recipe.
redact_replay_cmd() {
    local cmd="$1"
    if [ -n "${REMOTE_API_KEY:-}" ]; then
        cmd="${cmd//"$REMOTE_API_KEY"/\$REMOTE_API_KEY}"
    fi
    printf '%s' "$cmd"
}
```

- [ ] **Step 5: Redact the artifact write and suppress xtrace**

In `run_agentic_replay_and_write_outputs()`, replace these five lines:

```bash
    echo "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"

    set +e
    set -x
    $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
```

with:

```bash
    redact_replay_cmd "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"
    echo >> "$result_dir/benchmark_command.txt"

    set +e
    # xtrace of the pipeline below expands $REPLAY_CMD, and the 2>&1 redirects
    # the shell's stderr into tee — so with a key set the trace would write the
    # credential straight into benchmark.log. The command is already recorded
    # (redacted) above, so the trace is redundant when a key is present.
    if [ -z "${REMOTE_API_KEY:-}" ]; then
        set -x
    fi
    $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
```

`redact_replay_cmd` uses `printf` with no trailing newline, hence the separate `echo >>` to preserve the trailing newline `echo` previously produced.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest utils/test_benchmark_lib.py -v
```

Expected: PASS, all 7 tests (2 pre-existing + 5 new).

- [ ] **Step 7: Commit**

```bash
git add benchmarks/benchmark_lib.sh utils/test_benchmark_lib.py
git commit -m "feat(remote-bench): pass REMOTE_API_KEY to aiperf, redact it from artifacts"
```

---

### Task 2: Tokenizer decoupled from the served model name

A gateway may serve a model under an alias (`z-ai/glm-5.2`) while the tokenizer must resolve as its HuggingFace repo id (`zai-org/GLM-5.2-FP8`). aiperf uses `--model` for both unless `--tokenizer` is supplied.

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (inside `build_replay_cmd`, at the existing `--tokenizer-trust-remote-code` line)
- Test: `utils/test_benchmark_lib.py`

**Interfaces:**
- Consumes: `_run_bash` and `_AGENTIC_ENV` from Task 1.
- Produces: `build_replay_cmd` reads `AIPERF_TOKENIZER` from the environment and appends `--tokenizer <value>` when non-empty. Task 3's `remote_bench_preflight()` sets `AIPERF_TOKENIZER` from `REMOTE_TOKENIZER`.

- [ ] **Step 1: Write the failing tests**

Append to `utils/test_benchmark_lib.py`:

```python
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
```

The second test asserts both halves deliberately: `--tokenizer-trust-remote-code` must survive, and the substring match uses a trailing space so it does not accidentally match `--tokenizer-trust-remote-code`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest utils/test_benchmark_lib.py -v -k tokenizer
```

Expected: `test_aiperf_tokenizer_is_passed_when_set` FAILS (no `--tokenizer` emitted). `test_replay_cmd_omits_tokenizer_when_unset` passes already — it is the regression guard proving native commands are unchanged.

- [ ] **Step 3: Add the `--tokenizer` flag**

In `benchmarks/benchmark_lib.sh`, immediately **before** the existing line `REPLAY_CMD+=" --tokenizer-trust-remote-code"`, insert:

```bash
    # A gateway may alias the model (e.g. serve zai-org/GLM-5.2-FP8 as
    # z-ai/glm-5.2). aiperf's dataset manager loads a tokenizer by --model
    # unless --tokenizer overrides it, and an alias will not resolve on the
    # Hub. Unset for every native recipe, where model and tokenizer coincide.
    if [ -n "${AIPERF_TOKENIZER:-}" ]; then
        REPLAY_CMD+=" --tokenizer $AIPERF_TOKENIZER"
    fi
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest utils/test_benchmark_lib.py -v
```

Expected: PASS, all 9 tests.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_lib.sh utils/test_benchmark_lib.py
git commit -m "feat(remote-bench): allow tokenizer to differ from served model name"
```

---

### Task 3: Shared `remote_bench_preflight()`

Collapses the boilerplate currently duplicated across remote-bench recipes into one function, and adds the new behavior: URL normalization, optional GPU telemetry, optional context cap, and an authenticated reachability probe.

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (add the function immediately before `build_replay_cmd()`, which starts at line 1736)
- Test: `utils/test_benchmark_lib.py`

**Interfaces:**
- Consumes: `REMOTE_API_KEY` (Task 1), `AIPERF_TOKENIZER` (Task 2).
- Produces: `remote_bench_preflight` — takes no arguments, returns 0 on success and 1 on any failure. Reads `REMOTE_BASE_URL`, `REMOTE_ENGINE_METRICS_URL`, `REMOTE_RUNNER_TYPE` (required, validated by the caller's `check_env_vars`) and `REMOTE_API_KEY`, `REMOTE_TOKENIZER`, `REMOTE_GPU_TELEMETRY_URL`, `REMOTE_MAX_CONTEXT_LENGTH`, `REMOTE_RESET_URL` (optional). Exports `REMOTE_BASE_URL` (normalized), `RUNNER_TYPE`, `AIPERF_SERVER_METRICS_URLS`, and conditionally `AIPERF_GPU_TELEMETRY_URL`, `AIPERF_TOKENIZER`, `MAX_MODEL_LEN`. Task 4's recipes call it between `mkdir -p "$RESULT_DIR"` and `resolve_trace_source`.

- [ ] **Step 1: Write the failing tests**

Append to `utils/test_benchmark_lib.py`. The `curl` shell function shadows the real binary, so these tests do no network I/O:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest utils/test_benchmark_lib.py -v -k preflight
```

Expected: all 7 FAIL with `remote_bench_preflight: command not found`.

- [ ] **Step 3: Implement `remote_bench_preflight`**

In `benchmarks/benchmark_lib.sh`, immediately **before** the `build_replay_cmd()` definition, insert:

```bash
# Shared pre-flight for *-remote-bench.sh recipes: normalize the target URL,
# verify the endpoints we depend on, and export what build_replay_cmd and
# process_agentic_result.py read. Lives here rather than being copied into each
# recipe so authentication and optional telemetry apply to every remote-bench
# target, not only the newest one.
#
# Required (the calling recipe's check_env_vars enforces presence):
#   REMOTE_BASE_URL, REMOTE_ENGINE_METRICS_URL, REMOTE_RUNNER_TYPE
# Optional:
#   REMOTE_API_KEY             bearer token; switches the probe to /v1/models
#   REMOTE_TOKENIZER           HF repo id when the served name is an alias
#   REMOTE_GPU_TELEMETRY_URL   DCGM /metrics; absent => --no-gpu-telemetry
#   REMOTE_MAX_CONTEXT_LENGTH  trace-length cap; absent => no cap at all
#   REMOTE_RESET_URL           POSTed before each concurrency point
remote_bench_preflight() {
    # build_replay_cmd appends "--endpoint /v1/chat/completions", so a base URL
    # that already ends in /v1 would produce /v1/v1/chat/completions.
    if [[ "$REMOTE_BASE_URL" == */v1 ]]; then
        echo "NOTE: stripping trailing /v1 from REMOTE_BASE_URL; build_replay_cmd appends the endpoint path." >&2
        REMOTE_BASE_URL="${REMOTE_BASE_URL%/v1}"
    fi
    REMOTE_BASE_URL="${REMOTE_BASE_URL%/}"

    # Engine metrics are load-bearing: without KV, cache-hit and queue data a
    # remote-bench result is not interpretable, so fail here rather than let a
    # full-duration run finish without them. aiperf's own probing soft-fails,
    # which is the right default for a general-purpose client but not for us.
    if ! curl --output /dev/null --silent --fail --max-time 10 "$REMOTE_ENGINE_METRICS_URL"; then
        echo "ERROR: REMOTE_ENGINE_METRICS_URL ($REMOTE_ENGINE_METRICS_URL) is not reachable. Required for remote-bench." >&2
        return 1
    fi
    export AIPERF_SERVER_METRICS_URLS="$REMOTE_ENGINE_METRICS_URL"

    # GPU telemetry is optional: managed gateways rarely expose DCGM. A URL
    # that was explicitly supplied but does not answer is still an error —
    # that is a misconfiguration, not an absent capability.
    if [ -n "${REMOTE_GPU_TELEMETRY_URL:-}" ]; then
        if ! curl --output /dev/null --silent --fail --max-time 10 "$REMOTE_GPU_TELEMETRY_URL"; then
            echo "ERROR: REMOTE_GPU_TELEMETRY_URL ($REMOTE_GPU_TELEMETRY_URL) was supplied but is not reachable." >&2
            return 1
        fi
        export AIPERF_GPU_TELEMETRY_URL="$REMOTE_GPU_TELEMETRY_URL"
    else
        echo "NOTE: REMOTE_GPU_TELEMETRY_URL not set; running with --no-gpu-telemetry. GPU power and utilization will be absent from this result." >&2
    fi

    # Reachability probe. An authenticated gateway 401s on /health, so probe
    # /v1/models with the bearer token instead. This fails a wrong or expired
    # key in seconds rather than after a full-duration run of 401s.
    if [ -n "${REMOTE_API_KEY:-}" ]; then
        local probe_code
        probe_code=$(curl --output /dev/null --silent --max-time 10 \
            --write-out '%{http_code}' \
            --header "Authorization: Bearer $REMOTE_API_KEY" \
            "$REMOTE_BASE_URL/v1/models")
        if [ "$probe_code" != "200" ]; then
            echo "ERROR: authenticated probe of $REMOTE_BASE_URL/v1/models returned HTTP $probe_code (expected 200). Check REMOTE_API_KEY and REMOTE_BASE_URL." >&2
            return 1
        fi
    elif ! curl --output /dev/null --silent --fail --max-time 10 "$REMOTE_BASE_URL/health"; then
        echo "ERROR: REMOTE_BASE_URL ($REMOTE_BASE_URL) is not reachable at /health." >&2
        return 1
    fi

    if [ -n "${REMOTE_RESET_URL:-}" ]; then
        echo "Resetting remote engine state via REMOTE_RESET_URL before this concurrency point ..."
        curl --output /dev/null --silent --fail --max-time 30 -X POST "$REMOTE_RESET_URL"
    fi

    # Self-report the real hardware key for downstream ingest. RUNNER_TYPE is
    # otherwise the GH Actions runs-on label (cluster:remote-bench), which
    # hwToGpuKey() in InferenceX-app's ingest cannot resolve. Exporting it here,
    # in-process before aiperf runs, is what process_agentic_result.py
    # (os.environ.get("RUNNER_TYPE")) actually sees.
    export RUNNER_TYPE="$REMOTE_RUNNER_TYPE"
    export REMOTE_BASE_URL

    if [ -n "${REMOTE_TOKENIZER:-}" ]; then
        export AIPERF_TOKENIZER="$REMOTE_TOKENIZER"
    fi

    # benchmark_lib.sh unsets MAX_MODEL_LEN at source time for agentic scripts
    # so inherited workflow overrides never cap a local server's native
    # context. Setting it back here is deliberate — it is the only route to
    # build_replay_cmd's --max-context-length. Left unset when the target's
    # window exceeds the corpus, where no cap is wanted at all.
    if [ -n "${REMOTE_MAX_CONTEXT_LENGTH:-}" ]; then
        export MAX_MODEL_LEN="$REMOTE_MAX_CONTEXT_LENGTH"
    fi
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest utils/test_benchmark_lib.py -v
```

Expected: PASS, all 16 tests.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_lib.sh utils/test_benchmark_lib.py
git commit -m "feat(remote-bench): add shared remote_bench_preflight with optional telemetry"
```

---

### Task 4: The three recipes

Adds the fp8 GLM-5.2 recipe and reduces the two existing recipes to the same shape. The launcher formula in `runners/launch_bench-client.sh` is `benchmarks/single_node/agentic/${EXP_NAME%%_*}_${PRECISION}_${FRAMEWORK}-remote-bench.sh`, so `exp-name`'s first underscore-delimited segment must be `glm5.2` and the filename must match exactly.

**Files:**
- Create: `benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh`
- Modify: `benchmarks/single_node/agentic/glm5.2_fp4_sglang-remote-bench.sh` (replace lines 39–92)
- Modify: `benchmarks/single_node/agentic/dsv2lite_fp8_sglang-remote-bench.sh` (replace lines 39–92)
- Test: `utils/test_benchmark_lib.py`

**Interfaces:**
- Consumes: `remote_bench_preflight` (Task 3).
- Produces: `glm5.2_fp8_sglang-remote-bench.sh`, dispatched via `exp-name` values beginning `glm5.2_` with `precision: fp8`, `framework: sglang`.

- [ ] **Step 1: Write the failing test**

Append to `utils/test_benchmark_lib.py`:

```python
import glob

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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest utils/test_benchmark_lib.py -v -k recipe
```

Expected: `test_glm52_fp8_remote_bench_recipe_exists` FAILS (file absent); `test_remote_bench_recipes_use_shared_preflight` FAILS on the two existing recipes, which still list the optional vars in `check_env_vars` and do not call `remote_bench_preflight`.

- [ ] **Step 3: Create the fp8 GLM-5.2 recipe**

Write `benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

# Remote-bench recipe: benchmarks an already-running, externally-managed
# SGLang endpoint instead of launching one on this box. No local server, no
# router, no nvidia-smi — this runner only needs network access to the target
# and a Python venv for aiperf. Hardware-agnostic by design: nothing here
# launches a server, so there is one file per model+precision+framework
# combo, not one per hardware target.
#
# Built for GreenNode's MaaS gateway, which is authenticated and exposes
# neither /health nor a DCGM exporter. All of that is handled by
# remote_bench_preflight in benchmark_lib.sh; see that function's header for
# the full env-var contract, and
# docs/superpowers/specs/2026-07-29-remote-bench-authenticated-gateway-design.md
# for why each requirement is what it is.
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE

mkdir -p "$RESULT_DIR"

remote_bench_preflight

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
```

Then make it executable:

```bash
chmod +x benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh
```

- [ ] **Step 4: Reduce the two existing recipes**

In `benchmarks/single_node/agentic/glm5.2_fp4_sglang-remote-bench.sh`, delete lines 39–92 (everything from `check_env_vars` to the end of the file) and replace with:

```bash
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE

mkdir -p "$RESULT_DIR"

remote_bench_preflight

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
```

Apply the identical replacement to `benchmarks/single_node/agentic/dsv2lite_fp8_sglang-remote-bench.sh` (same line range, same content — the two files' bodies are identical today).

In both files, replace the whole `# Required inputs, all self-reported …` comment block (the run of comment lines from `# Required inputs` down to the last `#` line before `check_env_vars`) with:

```bash
# Required and optional inputs, all self-reported by whoever owns the remote
# endpoint, are documented on remote_bench_preflight() in benchmark_lib.sh —
# that function is the authoritative contract. In short: REMOTE_BASE_URL,
# REMOTE_ENGINE_METRICS_URL and REMOTE_RUNNER_TYPE are required;
# REMOTE_GPU_TELEMETRY_URL, REMOTE_MAX_CONTEXT_LENGTH, REMOTE_API_KEY,
# REMOTE_TOKENIZER and REMOTE_RESET_URL are optional. Behavior changes belong
# in remote_bench_preflight(), not here — a per-recipe fix silently misses
# every other remote target.
```

Leave lines 1–14 (the shebang, the `source`, and the "Remote-bench recipe:" paragraph) untouched in both files.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest utils/test_benchmark_lib.py -v
bash -n benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh
bash -n benchmarks/single_node/agentic/glm5.2_fp4_sglang-remote-bench.sh
bash -n benchmarks/single_node/agentic/dsv2lite_fp8_sglang-remote-bench.sh
```

Expected: pytest PASS, all 18 tests. Each `bash -n` (syntax check only, no execution) prints nothing and exits 0.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/single_node/agentic/
git commit -m "feat(remote-bench): add glm5.2 fp8 recipe, move recipes onto shared preflight"
```

---

### Task 5: Workflow wiring

Carries the secret and the new fields from dispatch to the runner. Also fixes an existing defect: `remote-bench-e2e.yml` hardcodes `kv-offloading: none`, so a target with HiCache enabled would be ingested as having no offloading.

Note that `benchmark_lib.sh` validates the KV-offload pair at source time for agentic scripts: `KV_OFFLOADING=dram` requires a non-empty `KV_OFFLOAD_BACKEND`, and `KV_OFFLOADING=none` requires an empty one. Passing `kv-offloading` without `kv-offload-backend` would fail the job at source time, which is why both move together.

**Files:**
- Modify: `.github/workflows/benchmark-tmpl.yml` (inputs block ending at `remote-max-context-length`, around line 161; env block, around line 210)
- Modify: `.github/workflows/remote-bench-e2e.yml` (the `with:` block, lines 60–87)
- Test: `utils/test_benchmark_lib.py`

**Interfaces:**
- Consumes: `REMOTE_TOKENIZER` and `REMOTE_API_KEY` env names (Tasks 1–3).
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Append to `utils/test_benchmark_lib.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest utils/test_benchmark_lib.py -v -k "tmpl or e2e"
```

Expected: both FAIL — no `REMOTE_API_KEY` env, no `remote-tokenizer` input, and `kv-offloading: none` still hardcoded.

- [ ] **Step 3: Add the input and env to `benchmark-tmpl.yml`**

In the inputs block, immediately after the `remote-max-context-length` entry, insert:

```yaml
      remote-tokenizer:
        description: "HuggingFace tokenizer repo id for the remote target, when the endpoint serves the model under a different name (e.g. gateway alias z-ai/glm-5.2 backed by zai-org/GLM-5.2-FP8). Optional; defaults to the --model value."
        required: false
        type: string
        default: ""
```

In the env block, immediately after `REMOTE_MAX_CONTEXT_LENGTH:`, insert:

```yaml
  REMOTE_TOKENIZER: ${{ inputs.remote-tokenizer }}
  # Bearer token for authenticated remote targets. Empty for every other
  # recipe, which leaves build_replay_cmd's --api-key branch inert.
  REMOTE_API_KEY: ${{ secrets.GREENNODE_API_KEY }}
```

Also update two existing descriptions in the same file to match reality:
- `remote-gpu-telemetry-url`: change "(required for remote-bench)" to "(optional; absent means the run uses --no-gpu-telemetry and reports no GPU power or utilization)".
- `remote-max-context-length`: change "(required for remote-bench)" to "(optional; omit when the target's context window exceeds the trace corpus, set it when the window is smaller)". Keep the rest of the existing description, including the auto-truncate hang note.

- [ ] **Step 4: Pass the fields through in `remote-bench-e2e.yml`**

In the `with:` block, replace the line `kv-offloading: none` with:

```yaml
      kv-offloading: ${{ matrix.config.kv-offloading || 'none' }}
      kv-offload-backend: ${{ matrix.config.kv-offload-backend || '' }}
      kv-offload-backend-metadata: ${{ matrix.config.kv-offload-backend-metadata || '' }}
```

and immediately after the `remote-max-context-length` line, add:

```yaml
      remote-tokenizer: ${{ matrix.config.remote-tokenizer || '' }}
```

Update the `configs` input description at the top of the file to list the new supported keys, appending `kv-offloading`, `kv-offload-backend`, `kv-offload-backend-metadata`, and `remote-tokenizer` to the existing enumeration.

- [ ] **Step 5: Run the tests and validate the YAML parses**

```bash
pytest utils/test_benchmark_lib.py -v
python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ['.github/workflows/benchmark-tmpl.yml','.github/workflows/remote-bench-e2e.yml']]; print('yaml ok')"
```

Expected: pytest PASS, all 20 tests; the python command prints `yaml ok`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/benchmark-tmpl.yml .github/workflows/remote-bench-e2e.yml utils/test_benchmark_lib.py
git commit -m "feat(remote-bench): wire GREENNODE_API_KEY, tokenizer and kv-offloading through dispatch"
```

---

### Task 6: Update the `create-remote-bench` skill

The skill currently instructs an operator to treat GPU telemetry and the context cap as required, tells them to copy a recipe wholesale, and has no interview step. All three are now wrong.

**Files:**
- Modify: `.claude/skills/create-remote-bench/SKILL.md`

**Interfaces:**
- Consumes: the env-var contract established in Tasks 1–5.
- Produces: nothing code depends on.

- [ ] **Step 1: Add step 0, "Interview the operator"**

Insert a new section before the existing `## 1. Find out what's actually behind the endpoint`:

````markdown
## 0. Interview the operator (scrape first, then ask)

Most of what a benchmark config needs is already in the engine's own
`/metrics`. Scrape it before asking anything:

```bash
curl -sk "$REMOTE_ENGINE_METRICS_URL" > /tmp/wm.txt
grep -o '^sglang:[a-z_0-9]*' /tmp/wm.txt | sort -u          # what's exposed
grep -o 'model_name="[^"]*"' /tmp/wm.txt | sort -u           # model + precision
grep '^sglang:context_len{' /tmp/wm.txt | head -1            # engine window
grep '^sglang:max_total_num_tokens{' /tmp/wm.txt | head -1   # KV pool per rank
grep -o 'dp_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l     # DP degree
grep -o 'tp_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l     # TP degree
grep -o 'moe_ep_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l # EP degree
grep '^sglang:hicache_host_total_tokens{' /tmp/wm.txt | head -1  # >0 => dram
grep '^sglang:spec_accept_length{' /tmp/wm.txt | head -1      # present => spec on
```

That gives you, without asking:

| Derived | From |
|---|---|
| model, precision | `model_name` label (`zai-org/GLM-5.2-FP8` → fp8) |
| engine context window | `sglang:context_len` |
| KV pool → CCU ladder | `sglang:max_total_num_tokens` × DP-rank count |
| TP / DP / EP | cardinality of the `*_rank` labels |
| `kv-offloading` | `sglang:hicache_host_total_tokens` > 0 → `dram` |
| spec decoding on/off | presence of `sglang:spec_accept_length` |

Show the operator that table and have them confirm it. Then ask only for what
metrics cannot reveal:

- **container image** actually deployed — recorded verbatim into the ingested
  artifact's `image` field; never leave it a placeholder
- **hardware string** for `REMOTE_RUNNER_TYPE` — must be `GPU_KEYS`-resolvable
  (e.g. `h200-nv`), not the runner label
- **`ep`**, when `moe_ep_rank` is not broken out per rank
- **spec-decoding algorithm name** — metrics prove it is on, not which one
- **`kv-offload-backend` name** — `native` for SGLang HiCache; existing values
  in `configs/nvidia-master.yaml` are `native`, `vllm-simple`, `mooncake`
- **gateway URL and served model name**, when the target is behind a gateway
- **tokenizer repo id**, when the served name is an alias → `remote-tokenizer`
- **API-key secret name** — currently `GREENNODE_API_KEY`
- **gateway-enforced context limit**, which may be *lower* than the engine's
  `context_len` and is not discoverable from metrics. This is the number that
  matters for `remote-max-context-length`, not the engine's.
````

- [ ] **Step 2: Rewrite the required-parameters table in step 2**

Replace the `REMOTE_GPU_TELEMETRY_URL` and `REMOTE_MAX_CONTEXT_LENGTH` rows in the "Required parameters" table. Move both into the Optional table, and add the new optional vars. The required set becomes `REMOTE_BASE_URL`, `REMOTE_ENGINE_METRICS_URL`, `REMOTE_RUNNER_TYPE`. Add to the Optional table:

```markdown
| `REMOTE_GPU_TELEMETRY_URL` | DCGM `/metrics`-style endpoint. Absent means the run uses `--no-gpu-telemetry` and reports no GPU power or utilization — acceptable for a managed gateway that does not expose DCGM. A URL that *is* supplied but does not answer still fails the job: that is a misconfiguration, not an absent capability. |
| `REMOTE_MAX_CONTEXT_LENGTH` | Trace-length cap. Three cases: target window **above** the corpus → omit it entirely; **below** the corpus → set it to the real enforced limit; run **hangs** at the nominal window → binary-search the value downward. Capping below the corpus's shortest trace fails outright with `DatasetLoaderError: All N traces exceed --max-context-length`. |
| `REMOTE_API_KEY` | Bearer token, supplied by `benchmark-tmpl.yml` from the `GREENNODE_API_KEY` repo secret. Never commit a key. When set, the pre-flight probes `GET $REMOTE_BASE_URL/v1/models` with the token instead of `/health`, so a wrong key fails in seconds. |
| `REMOTE_TOKENIZER` | HuggingFace repo id, when the endpoint serves the model under an alias. aiperf's dataset manager loads a tokenizer by `--model` unless this overrides it, and an alias will not resolve on the Hub. |
```

Keep the existing `REMOTE_RESET_URL` row and the existing incident notes about the auto-truncate hang — they are still accurate, just no longer attached to a required parameter.

- [ ] **Step 3: Replace the copy-a-recipe instruction in step 3**

The shared logic now lives in `remote_bench_preflight()`. Replace the "Copy an existing one and rename" paragraph with:

````markdown
The body is model-, framework- and hardware-agnostic: everything shared lives
in `remote_bench_preflight()` in `benchmark_lib.sh`. A new recipe is:

```bash
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE
mkdir -p "$RESULT_DIR"
remote_bench_preflight
resolve_trace_source
install_agentic_deps
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
```

One file per model+precision+framework combo; do not create one per
hardware/cluster. Behavior changes belong in `remote_bench_preflight()`, not in
a recipe — a per-recipe fix silently misses every other target.
````

- [ ] **Step 4: Add the corpus-selection warning to step 5**

Add to the "Check the KV pool before picking a CCU ladder" section:

````markdown
**Check which corpus your `model-prefix` selects.** `resolve_trace_source()`
picks the trace corpus from `MODEL_PREFIX`, not from the target's actual
deployed window:

```
dsv4*|glm5.2*|minimaxm3*)  → semianalysis_cc_traces_weka_062126        # unfiltered
*)                         → semianalysis_cc_traces_weka_062126_256k   # 256k-capped
```

So a `glm5.2`-prefixed target inherits the **unfiltered** corpus regardless of
what the deployment actually supports, because glm5.2's native window is ~1M.
If your target's window is smaller than the corpus, prefer
`REMOTE_MAX_CONTEXT_LENGTH` — it keeps the same corpus and filters only the
traces above the cap, which stays comparable to a native run. Reach for
`WEKA_LOADER_OVERRIDE` to swap corpora only when the enforced limit is at or
below 256k; that is a different dataset, and results are then comparable only
against other runs on the same corpus.
````

- [ ] **Step 5: Add an authenticated-endpoints section**

Insert after the existing step 6 (Dispatch):

````markdown
## 6b. Authenticated endpoints

The key lives in exactly one place: the `GREENNODE_API_KEY` repo secret in
`vngcloud/InferenceX`. `benchmark-tmpl.yml` maps it to `REMOTE_API_KEY`, which
the runner inherits as process environment. Never put a key in a config, a
dispatch payload, a recipe, or a commit.

aiperf has no env-var transport for `api_key`, so it must be an argv element.
Two consequences, both handled:

- `benchmark_command.txt` is uploaded as an artifact and GitHub Actions does
  **not** mask artifact contents, so `redact_replay_cmd()` substitutes the
  literal `$REMOTE_API_KEY` before the file is written.
- Shell xtrace of the replay pipeline would land in `benchmark.log` via the
  `2>&1 | tee`, so `run_agentic_replay_and_write_outputs` skips `set -x` when a
  key is set.

Job logs are masked by Actions itself. Anyone with `ps` access on the
controller box during a run can read the key from argv; this is accepted
because `bench-client_01` is a dedicated single-tenant controller and aiperf
offers no alternative.

After any change here, re-run the guard tests:

```bash
pytest utils/test_benchmark_lib.py -v -k "api_key or redact"
```
````

- [ ] **Step 6: Update the dispatch examples in step 6**

Update both `gh workflow run` examples so the JSON objects show the current field set: drop `remote-gpu-telemetry-url` and `remote-max-context-length` from the required-looking list, and add `remote-tokenizer`, `kv-offloading`, `kv-offload-backend`. Replace the smoke-test example with the block below. The full-sweep example directly beneath it keeps its existing shape — one object per `conc` value, every object carrying this same field set, with only `conc` and `exp-name` varying — so update its objects to match this field set too.

The angle-bracketed values below are genuinely operator-supplied at dispatch time (see step 0), not placeholders to resolve while editing the skill.

```bash
gh workflow run remote-bench-e2e.yml -R vngcloud/InferenceX --ref <branch> \
  -f configs='[
    {"exp-name": "glm5.2_smoke", "conc": "1", "duration": "300",
     "image": "<real deployed image>", "model": "z-ai/glm-5.2",
     "model-prefix": "glm5.2", "framework": "sglang", "precision": "fp8",
     "tp": "8", "dp-attn": "true",
     "kv-offloading": "dram", "kv-offload-backend": "native",
     "remote-base-url": "https://maas-llm-aiplatform-hcm.api.vngcloud.vn",
     "remote-tokenizer": "zai-org/GLM-5.2-FP8",
     "remote-engine-metrics-url": "https://49.213.86.184.nip.io/glm/sglang/worker-metrics",
     "remote-runner-type": "<hw string>"}
  ]'
```

- [ ] **Step 7: Verify and commit**

Re-read the edited `SKILL.md` end to end and confirm no section still calls `REMOTE_GPU_TELEMETRY_URL` or `REMOTE_MAX_CONTEXT_LENGTH` required, and no section still says to copy a recipe file wholesale.

```bash
grep -n "required for remote-bench\|Copy an existing one" .claude/skills/create-remote-bench/SKILL.md
```

Expected: no output.

```bash
git add .claude/skills/create-remote-bench/SKILL.md
git commit -m "docs(create-remote-bench): scrape-first interview, optional telemetry and context cap"
```

---

## Validation before dispatch

After all six tasks, run the whole guard suite and a syntax check:

```bash
pytest utils/test_benchmark_lib.py -v
for f in benchmarks/single_node/agentic/*-remote-bench.sh; do bash -n "$f" || echo "FAIL $f"; done
python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/benchmark-tmpl.yml','.github/workflows/remote-bench-e2e.yml']]; print('yaml ok')"
```

Then, per the spec's testing section, in order:

1. Add the `GREENNODE_API_KEY` secret to `vngcloud/InferenceX`.
2. Run the recipe by hand on the `bench-client_01` controller with env exported manually, per the skill's existing debug loop. Confirm the authenticated pre-flight passes and that `grep -c "$GREENNODE_API_KEY" "$RESULT_DIR"/benchmark_command.txt "$RESULT_DIR"/benchmark.log` returns 0 for both files.
3. Dispatch one config at `conc: 1`, `duration: 300` — which triggers `--unsafe-override`, so `submission_valid` will be false, as expected for a smoke test.
4. Confirm `agg_*.json` has a `GPU_KEYS`-resolvable `hw`, the real deployed `image`, non-zero throughput, and populated `server_metrics.kv_cache`.
5. Dispatch the full ladder: `conc` 1, 2, 4, 8, 16, 32 at `duration: 3600`.

`workflow_dispatch` reads the workflow file from the default branch, so the Task 5 changes must merge to `main` before steps 3–5 can use the new matrix fields. Steps 1–2 do not depend on that.
