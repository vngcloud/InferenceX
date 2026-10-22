# Serving-YAML Dispatch (V1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent writes one `serving/<name>.yaml` (image, model, serve command, scenario, concurrencies) and runs `utils/serving/dispatch.sh serving/<name>.yaml`; GitHub Actions runs an AgentX benchmark for it on `h200-greennode-slurm-1x_01` with no config key, no hand-written recipe and no SHA juggling.

**Architecture:** `e2e-tests.yml` gets an optional `serving` input. When it is set, `get-jobs` calls `python -m infx.serving.to_matrix`, which validates the YAML and emits (a) normal generator-shaped matrix rows and (b) one rendered `serving` JSON object. The agentic job passes (b) to `benchmark-tmpl.yml` as a new input, which exports it as env. The 1×H200 launcher sees `SERVE_CMD` and runs one generic recipe, `serve_generic.sh`, which starts the engine and runs aiperf with `--config scenarios/<scenario>.yaml` plus per-run overrides (native aiperf YAML).

**Tech Stack:** Python 3.12 + pydantic v2 + PyYAML (run through `uv`), bash, GitHub Actions, Slurm + enroot on the 1×H200 node, aiperf (git submodule `utils/aiperf`, pinned `faf3f8b3`).

**Spec:** `docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md` (read the whole file, including the **Amendments** section at the end, which overrides earlier sections where they conflict).

## How to start (fresh session)

You are on the vngcloud fork of InferenceX. Nothing from the planning session is in your context; this file plus the spec is everything.

```bash
cd /home/thanglq5/greennode-code/InferenceOptimization/InferenceX
git fetch origin
git checkout feat/serving-config-dispatch          # local branch; created from vng-benchmark @ 6c54ee117, holds the spec commits
git log --oneline -4                               # expect the spec commits on top of 6c54ee117
git submodule update --init utils/aiperf           # needed by Task 1 (aiperf source); pinned faf3f8b3
git -C utils/aiperf rev-parse --short HEAD         # expect faf3f8b3
```

If you use a worktree (superpowers:using-git-worktrees), run the `git submodule update --init utils/aiperf` inside it.

Test command used throughout (run from the repo root):

```bash
PYTEST="uv run -q --python 3.12 --with pytest --with pyyaml --with pydantic --no-project python -m pytest -c utils/pytest.ini --rootdir utils"
$PYTEST utils/test_serving_schema.py
```

Known baseline: 4 tests in `utils/test_benchmark_lib.py` (`*api_key*`) already fail on this branch because their env lacks `INFMAX_CONTAINER_WORKSPACE`/`PORT`. Not yours; do not fix them, do not count them.

## Global Constraints

- Git author: `-c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai` on every commit. `git add` only the files of the task (never `-A`/`.`); never commit `.serena/` or other tool-local dirs.
- **Ask the user before any `git push`**, before cancelling/resubmitting the runner daemon (Task 9), and before writing on the node. Never push to `vng-benchmark` or `main`.
- Always pass `--repo vngcloud/InferenceX` to `gh` (the local default may be `SemiAnalysisAI/InferenceX`).
- Target runner: `h200-greennode-slurm-1x_01` = node `hoanq3-h200-1x-han-3`, 1×H200, launcher `runners/launch_h200-greennode-slurm-1x.sh`, partition `test`, account `greennode`. Submit host: `ssh slurm-client`.
- V1 is engine only. A `router:` key is rejected with: `router: not supported in V1 (engine only); remove the router block`.
- No ingest: the serving path adds no ingest job or step.
- `model` is an HF repo id; `serve` must not contain `/models/`, `/mnt/`, `/data/`.
- Serving YAML keys: `name runner image model gpus precision env serve scenario conc` (no `duration`: the scenario owns it).
- Scenarios: `scenarios/agentx-cc-062126.yaml` (profiling duration 3600 s) and `scenarios/agentx-cc-062126-smoke.yaml` (180 s), otherwise identical.
- Matrix rows must pass `infx.workflows.benchmark_schema.validate_matrix` unchanged (its models are `extra="forbid"`): rendered serving data never goes into matrix rows.
- Runner `.env` variables use the `IX_` prefix: `IX_HF_CACHE_HOST`, `IX_SCRATCH_ROOT`, `IX_SLURM_NODELIST`. Without a `.env` the launcher behaves exactly as today.
- Write files for the user inside the repo; report to the user in Vietnamese, concisely.

## Review Focus

Inputs the spec implies but that are easy to break; each has a test in the owning task:

1. A pasted serve command with backslash line continuations and a single-quoted JSON argument (`--default-chat-template-kwargs '{"enable_thinking": true}'`) must reach the engine with that argument intact → Task 3 `test_render_keeps_quoted_json_and_continuations`.
2. `--port 8889`, `--port=8889`, `--host 0.0.0.0`, `--host=0.0.0.0` in any position are all removed (the recipe owns host/port) → Task 3 `test_render_strips_host_and_port_forms`.
3. A command prefixed with env assignments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve …`) works; the assignments move into the env → Task 3 `test_render_moves_leading_env_assignments`.
4. YAML env values written unquoted (`SGLANG_DP_USE_GATHERV: 1`, `NCCL_TIMEOUT: 1.5`) are accepted as strings; unquoted booleans are rejected with a fix hint → Task 2 `test_env_numbers_become_strings`, `test_env_bool_rejected`.
5. `scenario: agentx-cc-062126-smoke.yaml` (with suffix or `scenarios/` prefix) and `conc: 8` (scalar) are accepted → Task 2 `test_scenario_suffix_and_scalar_conc`.

---

### Task 1: Scenario YAMLs + golden equivalence check

Native aiperf YAML for the reference AgentX command, verified equal to the CLI form that produced the green qwen38 smoke run `37733664587`.

**Files:**
- Create: `scenarios/agentx-cc-062126.yaml`
- Create: `scenarios/agentx-cc-062126-smoke.yaml`
- Create: `utils/serving/check_scenario_golden.py`

**Interfaces:**
- Produces: the two scenario files (read by Tasks 2, 4, 5); the per-run override list that Task 5's `build_replay_cmd_yaml` must emit. Task 5 re-runs this script against the real bash function.

Background. The reference command (copied from the job log of run `37733664587`):

```
aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:8888 --endpoint /v1/chat/completions --endpoint-type chat --streaming --model Qwen/Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B --concurrency 8 --benchmark-duration 90 --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --gpu-telemetry http://localhost:9400/metrics benchmarks/single_node/agentic/qwen38_bf16_h200.gpu_metrics.csv --tokenizer-trust-remote-code --max-context-length 262144 --num-dataset-entries 393 --slice-duration 1.0 --server-metrics http://localhost:8888/metrics --output-artifact-dir /workspace/results/aiperf_artifacts --unsafe-override --public-dataset semianalysis_cc_traces_weka_062126
```

aiperf resolves both CLI-only and `--config file + flags` through `aiperf.config.flags.resolver.resolve_config(cli_config, config_file)`. A planning spike showed the YAML below + per-run flags resolves to a config identical to the CLI one (0 differing fields), and a negative check caught changed `entries`/`concurrency`. The GPU-metrics CSV path in the reference is dropped: the generic recipe has no sidecar CSV.

- [ ] **Step 1: Write the full scenario**

`scenarios/agentx-cc-062126.yaml`:

```yaml
# AgentX replay of the SemiAnalysis CC traces (062126 corpus), the workload of
# build_replay_cmd in benchmarks/benchmark_lib.sh, as native aiperf YAML.
# Per-run values (url, model, tokenizer, concurrency, metrics/telemetry URLs,
# artifact dir) are CLI overrides added by build_replay_cmd_yaml.
# Equivalence with the CLI form: utils/serving/check_scenario_golden.py.
schemaVersion: "2.0"
benchmark:
  scenario: inferencex-agentx-mvp
  unsafeOverride: true
  endpoint:
    type: chat
    path: /v1/chat/completions
    streaming: true
    useServerTokenCount: true
  datasets:
  - name: main
    type: public
    dataset: semianalysis_cc_traces_weka_062126
    entries: 393
    randomSeed: 42
    traceIdleGapCapSeconds: 300.0
    maxContextLength: 262144
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 3600.0
    gracePeriod: 30.0
    failedRequestThreshold: 0.1
    trajectoryStartMinRatio: 0.25
    trajectoryStartMaxRatio: 0.75
    warmupRequestsPerLane: 10
    agenticWarmupGracePeriod: 1800.0
  artifacts:
    sliceDuration: 1.0
  tokenizer:
    trustRemoteCode: true
  runtime:
    statsInterval: 30.0
randomSeed: 42
```

Note: `unsafeOverride: true` matches the reference command (`build_replay_cmd` always passes `--unsafe-override` because the trace source is a local/public alias); it stamps `submission_valid=false`, as all existing fork runs are.

- [ ] **Step 2: Write the smoke scenario**

`scenarios/agentx-cc-062126-smoke.yaml`: the same file with two changes: the header comment's first line becomes `# Smoke variant of agentx-cc-062126.yaml: 180 s profiling, for bring-up only.` followed by `# inferencex-agentx-mvp locks a 900 s minimum; unsafeOverride turns that into a warning.`, and `duration: 3600.0` becomes `duration: 180.0`. Copy the rest verbatim.

- [ ] **Step 3: Write the golden check**

`utils/serving/check_scenario_golden.py`:

```python
"""Golden check: scenarios/<name>.yaml + per-run overrides == the reference AgentX CLI.

Run inside the aiperf project env (needs aiperf + cyclopts), from the repo root:

    (cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)

For each scenario it resolves (a) the reference CLI command that produced the
green qwen38 smoke run 37733664587, with --benchmark-duration set to the
scenario's duration, and (b) aiperf profile --config <scenario> + the overrides
that build_replay_cmd_yaml emits (taken by running that bash function), then
compares the two resolved configs field by field. Exit 1 on any difference.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from cyclopts import App

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("agentx-cc-062126", "agentx-cc-062126-smoke")
MODEL = "Qwen/Qwen3.8-27B"


def reference_argv(duration: int, artifact_dir: str) -> list[str]:
    return shlex.split(
        "--scenario inferencex-agentx-mvp --url http://localhost:8888 "
        "--endpoint /v1/chat/completions --endpoint-type chat --streaming "
        f"--model {MODEL} --tokenizer {MODEL} --concurrency 8 "
        f"--benchmark-duration {duration} --stats-interval 30 --random-seed 42 "
        "--failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 "
        "--trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 "
        "--trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 "
        "--use-server-token-count --gpu-telemetry http://localhost:9400/metrics "
        "--tokenizer-trust-remote-code --max-context-length 262144 "
        "--num-dataset-entries 393 --slice-duration 1.0 "
        "--server-metrics http://localhost:8888/metrics "
        f"--output-artifact-dir {artifact_dir}/aiperf_artifacts --unsafe-override "
        "--public-dataset semianalysis_cc_traces_weka_062126"
    )


def yaml_argv(scenario: str, artifact_dir: str) -> list[str]:
    """The argv build_replay_cmd_yaml produces, minus the leading `aiperf profile`."""
    script = (
        "set -e\n"
        "export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=0\n"
        f"export INFMAX_CONTAINER_WORKSPACE={shlex.quote(str(REPO))}\n"
        f"export AIPERF_CLI=aiperf MODEL={MODEL} FRAMEWORK=vllm SERVING_SCENARIO={scenario}\n"
        "export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics\n"
        "export AIPERF_SERVER_METRICS_URLS=http://localhost:8888/metrics\n"
        "source benchmarks/benchmark_lib.sh\n"
        f"build_replay_cmd_yaml {shlex.quote(artifact_dir)}\n"
        'printf "%s" "$REPLAY_CMD"\n'
    )
    out = subprocess.run(["bash", "-c", script], cwd=REPO, check=True, capture_output=True, text=True).stdout
    argv = shlex.split(out)
    assert argv[:2] == ["aiperf", "profile"], argv[:2]
    return argv[2:]


def parse(argv: list[str]) -> CLIConfig:
    captured: dict[str, CLIConfig] = {}
    app = App(name="golden")

    @app.default
    def _capture(*, cli_config: CLIConfig) -> None:
        captured["cli"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:  # cyclopts exits 0 after running the default command
        if exc.code not in (0, None):
            raise
    return captured["cli"]


def resolved(argv: list[str]) -> dict:
    cli = parse(argv)
    return resolve_config(cli, cli.config_file).model_dump(mode="json", by_alias=True)


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += diff(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: cli={json.dumps(a)} yaml={json.dumps(b)}"]


def main() -> int:
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in SCENARIOS:
            data = yaml.safe_load((REPO / "scenarios" / f"{scenario}.yaml").read_text())
            duration = int(data["benchmark"]["phases"][0]["duration"])
            differences = diff(resolved(reference_argv(duration, tmp)), resolved(yaml_argv(scenario, tmp)))
            if differences:
                failed = True
                print(f"FAIL {scenario}:")
                for line in differences:
                    print("  " + line)
            else:
                print(f"OK {scenario} ({duration}s): YAML + overrides == reference CLI")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("AIPERF_LOG_LEVEL", "WARNING")
    sys.exit(main())
```

- [ ] **Step 4: Run it — expect failure**

Run: `(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)`
Expected: FAIL with `subprocess.CalledProcessError` (the bash stderr says `build_replay_cmd_yaml: command not found`; Task 5 adds it). This proves the script really calls the bash function. Leave it failing; Task 5 Step 6 makes it pass.

- [ ] **Step 5: Commit**

```bash
git add scenarios/agentx-cc-062126.yaml scenarios/agentx-cc-062126-smoke.yaml utils/serving/check_scenario_golden.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): AgentX scenarios as native aiperf YAML + golden check"
```

---

### Task 2: Serving YAML schema and validation

**Files:**
- Create: `infx/serving/__init__.py` (empty docstring module)
- Create: `infx/serving/schema.py`
- Test: `utils/test_serving_schema.py`

**Interfaces:**
- Consumes: `configs/runners.yaml` (`labels: {pool: [runner names]}`, `hardware: {pool: {gpus-per-node: N}}`), `scenarios/*.yaml` from Task 1.
- Produces:
  - `class ServingError(ValueError)` — message is one or more lines, each `"<field>: <problem and fix>"`.
  - `class ServingSpec(BaseModel)` with fields `name: str, runner: str, image: str, model: str, gpus: int, precision: str = "bf16", env: dict[str, str], serve: str, scenario: str, conc: list[int]`.
  - `def load_serving(text: str, repo: Path = REPO_ROOT) -> ServingSpec` (raises `ServingError`).
  - `def available_scenarios(repo: Path = REPO_ROOT) -> list[str]`.
  - `REPO_ROOT: Path` (repo root).

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_schema.py`:

```python
import textwrap

import pytest

from infx.serving.schema import ServingError, available_scenarios, load_serving

GOOD = textwrap.dedent("""\
    name: qwen38-bf16-tp1
    runner: h200-greennode-slurm-1x_01
    image: vllm/vllm-openai:v0.29.0
    model: Qwen/Qwen3.8-27B
    gpus: 1
    serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144
    scenario: agentx-cc-062126-smoke
    conc: [4, 8]
""")


def with_line(old: str, new: str) -> str:
    assert old in GOOD, old
    return GOOD.replace(old, new)


def errors(text: str) -> str:
    with pytest.raises(ServingError) as info:
        load_serving(text)
    return str(info.value)


def test_good_yaml_loads_with_defaults():
    spec = load_serving(GOOD)
    assert spec.name == "qwen38-bf16-tp1"
    assert spec.precision == "bf16"
    assert spec.env == {}
    assert spec.conc == [4, 8]


def test_available_scenarios_lists_both():
    assert {"agentx-cc-062126", "agentx-cc-062126-smoke"} <= set(available_scenarios())


def test_router_rejected_with_v1_message():
    assert errors(GOOD + "router: {image: x}\n") == "router: not supported in V1 (engine only); remove the router block"


def test_duration_rejected_with_hint():
    assert "the scenario fixes the duration" in errors(GOOD + "duration: 90\n")


def test_unknown_key_names_allowed_keys():
    message = errors(GOOD + "imagee: x\n")
    assert message.startswith("imagee: unknown key")
    assert "image" in message


def test_missing_image_is_reported():
    assert "image: Field required" in errors(with_line("image: vllm/vllm-openai:v0.29.0\n", ""))


def test_name_with_underscore_rejected():
    assert errors(with_line("name: qwen38-bf16-tp1", "name: qwen38_bf16")).startswith("name:")


def test_local_model_path_rejected():
    assert "model: use the HF repo id" in errors(with_line("model: Qwen/Qwen3.8-27B", "model: /models/Qwen3.8-27B"))


def test_local_path_in_serve_rejected():
    text = with_line("serve: vllm serve Qwen/Qwen3.8-27B", "serve: vllm serve /models/PhalaCloud/GLM")
    assert "serve: local path containing '/models/'" in errors(text)


def test_shell_variable_in_serve_rejected():
    assert "serve: shell variables" in errors(with_line("--max-model-len 262144", "--port $PORT"))


def test_unknown_runner_rejected():
    assert errors(with_line("runner: h200-greennode-slurm-1x_01", "runner: nope")).startswith("runner: unknown runner 'nope'")


def test_too_many_gpus_for_runner():
    assert "gpus: 8 > 1 GPUs per node" in errors(with_line("gpus: 1", "gpus: 8"))


def test_eight_gpus_ok_on_hardware_hcm():
    text = with_line("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01").replace("gpus: 1", "gpus: 8")
    assert load_serving(text).gpus == 8


def test_missing_scenario_lists_available():
    message = errors(with_line("scenario: agentx-cc-062126-smoke", "scenario: nope"))
    assert "scenario: no scenarios/nope.yaml; available:" in message
    assert "agentx-cc-062126-smoke" in message


def test_scenario_suffix_and_scalar_conc():
    text = with_line("scenario: agentx-cc-062126-smoke", "scenario: scenarios/agentx-cc-062126-smoke.yaml")
    text = text.replace("conc: [4, 8]", "conc: 8")
    spec = load_serving(text)
    assert spec.scenario == "agentx-cc-062126-smoke"
    assert spec.conc == [8]


def test_duplicate_conc_rejected():
    assert "conc: duplicate values" in errors(with_line("conc: [4, 8]", "conc: [8, 8]"))


def test_env_numbers_become_strings():
    spec = load_serving(GOOD + "env:\n  SGLANG_DP_USE_GATHERV: 1\n  NCCL_TIMEOUT: 1.5\n  NCCL_P2P_LEVEL: NVL\n")
    assert spec.env == {"SGLANG_DP_USE_GATHERV": "1", "NCCL_TIMEOUT": "1.5", "NCCL_P2P_LEVEL": "NVL"}


def test_env_bool_rejected():
    assert "quote boolean values" in errors(GOOD + "env:\n  HF_HUB_OFFLINE: true\n")


def test_env_bad_name_rejected():
    assert "env.BAD-NAME: not a valid environment variable name" in errors(GOOD + "env:\n  BAD-NAME: x\n")


def test_not_a_mapping():
    assert errors("- a\n- b\n") == "serving: the top level must be a mapping of keys"


def test_multiple_problems_one_line_each():
    text = with_line("model: Qwen/Qwen3.8-27B", "model: /models/x").replace("gpus: 1", "gpus: 8")
    lines = errors(text).splitlines()
    assert len(lines) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_schema.py`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'infx.serving'`.

- [ ] **Step 3: Implement**

`infx/serving/__init__.py`:

```python
"""Serving-YAML dispatch (V1). Spec: docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md"""
```

`infx/serving/schema.py`:

```python
"""Serving YAML (V1): the one file an agent writes to dispatch an AgentX benchmark.

Spec: docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md
Every error is one line "<field>: <problem and fix>" so an agent can act on it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
LOCAL_PATH_MARKERS = ("/models/", "/mnt/", "/data/")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ServingError(ValueError):
    """Validation failure; each line names a field and the fix."""


class ServingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    runner: str = Field(min_length=1)
    image: str = Field(min_length=1)
    model: str = Field(min_length=1)
    gpus: int = Field(gt=0)
    precision: str = Field(default="bf16", pattern=r"^[a-z0-9]+$")
    env: dict[str, str] = Field(default_factory=dict)
    serve: str = Field(min_length=1)
    scenario: str = Field(pattern=NAME_PATTERN)
    conc: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)

    @field_validator("conc", mode="before")
    @classmethod
    def _scalar_conc(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        return value

    @field_validator("scenario", mode="before")
    @classmethod
    def _scenario_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.removeprefix("scenarios/")
            for suffix in (".yaml", ".yml"):
                value = value.removesuffix(suffix)
        return value

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            if isinstance(item, bool):
                raise ValueError(f'{key}: quote boolean values, e.g. "1" or "true"')
            out[key] = str(item) if isinstance(item, (int, float)) else item
        return out


def available_scenarios(repo: Path = REPO_ROOT) -> list[str]:
    return sorted(path.stem for path in (repo / "scenarios").glob("*.yaml"))


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "serving"
        message = item["msg"]
        if item["type"] == "extra_forbidden":
            message = "unknown key (allowed: " + ", ".join(ServingSpec.model_fields) + ")"
        lines.append(f"{location}: {message}")
    return "\n".join(lines)


def _gpus_per_node(runner: str, repo: Path) -> int:
    runners = yaml.safe_load((repo / "configs" / "runners.yaml").read_text())
    for pool, names in (runners.get("labels") or {}).items():
        if isinstance(names, list) and runner in names:
            gpus = ((runners.get("hardware") or {}).get(pool) or {}).get("gpus-per-node")
            if gpus is None:
                raise ServingError(f"runner: pool {pool} has no gpus-per-node in configs/runners.yaml")
            return int(gpus)
    raise ServingError(
        f"runner: unknown runner {runner!r}; use a runner name from configs/runners.yaml "
        "(V1 target: h200-greennode-slurm-1x_01)"
    )


def load_serving(text: str, repo: Path = REPO_ROOT) -> ServingSpec:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ServingError(f"serving: not valid YAML ({error})") from error
    if not isinstance(raw, dict):
        raise ServingError("serving: the top level must be a mapping of keys")
    if "router" in raw:
        raise ServingError("router: not supported in V1 (engine only); remove the router block")
    if "duration" in raw:
        raise ServingError(
            "duration: not a serving key; the scenario fixes the duration "
            "(pick the -smoke scenario for a short run)"
        )
    try:
        spec = ServingSpec.model_validate(raw)
    except ValidationError as error:
        raise ServingError(_format_validation_error(error)) from error

    problems: list[str] = []
    if spec.model.startswith("/"):
        problems.append("model: use the HF repo id (e.g. Qwen/Qwen3.8-27B), not a local path")
    for marker in LOCAL_PATH_MARKERS:
        if marker in spec.serve:
            problems.append(f"serve: local path containing {marker!r}; replace it with the HF repo id from model")
            break
    if "$" in spec.serve:
        problems.append("serve: shell variables ($...) are not expanded; write literal values")
    for key, value in spec.env.items():
        if not ENV_NAME.fullmatch(key):
            problems.append(f"env.{key}: not a valid environment variable name")
        if "\n" in value:
            problems.append(f"env.{key}: value must be a single line")
    if len(set(spec.conc)) != len(spec.conc):
        problems.append("conc: duplicate values")
    try:
        gpus_per_node = _gpus_per_node(spec.runner, repo)
        if spec.gpus > gpus_per_node:
            problems.append(
                f"gpus: {spec.gpus} > {gpus_per_node} GPUs per node on runner {spec.runner}; pick a bigger runner"
            )
    except ServingError as error:
        problems.append(str(error))
    if not (repo / "scenarios" / f"{spec.scenario}.yaml").is_file():
        problems.append(
            f"scenario: no scenarios/{spec.scenario}.yaml; available: {', '.join(available_scenarios(repo))}"
        )
    if problems:
        raise ServingError("\n".join(problems))
    return spec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_schema.py`
Expected: all PASS. If `test_missing_image_is_reported` fails only on wording, print `errors(...)` and match pydantic's actual "Field required" text; do not loosen other assertions.

- [ ] **Step 5: Commit**

```bash
git add infx/serving/__init__.py infx/serving/schema.py utils/test_serving_schema.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): serving YAML schema with one-line, agent-actionable errors"
```

---

### Task 3: Render the serve command

**Files:**
- Create: `infx/serving/render.py`
- Test: `utils/test_serving_render.py`

**Interfaces:**
- Consumes: `ServingSpec`, `ServingError` (Task 2).
- Produces: `def render(spec: ServingSpec) -> dict[str, str]` with keys `framework` (`"vllm"`|`"sglang"`), `serve-cmd` (shell-quoted argv via `shlex.join`, **without** `--host/--port`; `serve_generic.sh` appends them), `serve-env` (`KEY=VALUE` lines joined by `\n`, spec `env` overrides command-prefix assignments), `served-model-name` (value of `--served-model-name`, or `""`).
- Also produces helpers used by tests: `split_serve(serve: str) -> list[str]`, `infer_framework(argv: list[str]) -> str`.

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_render.py`:

```python
import shlex
import textwrap

import pytest

from infx.serving.render import infer_framework, render, split_serve
from infx.serving.schema import ServingError, ServingSpec


def spec(serve: str, **extra) -> ServingSpec:
    fields = dict(
        name="x", runner="h200-greennode-slurm-1x_01", image="img", model="Qwen/Qwen3.8-27B",
        gpus=1, serve=serve, scenario="agentx-cc-062126-smoke", conc=[8],
    )
    fields.update(extra)
    return ServingSpec.model_validate(fields)


def test_render_keeps_quoted_json_and_continuations():
    serve = textwrap.dedent("""\
        vllm serve Qwen/Qwen3.8-27B \\
          --max-model-len 262144 \\
          --default-chat-template-kwargs '{"enable_thinking": true}'
    """)
    out = render(spec(serve))
    assert shlex.split(out["serve-cmd"]) == [
        "vllm", "serve", "Qwen/Qwen3.8-27B", "--max-model-len", "262144",
        "--default-chat-template-kwargs", '{"enable_thinking": true}',
    ]


@pytest.mark.parametrize("flags", [
    "--host 0.0.0.0 --port 8889",
    "--port=8889 --host=0.0.0.0",
    "--port 8889 --tp 1 --host 127.0.0.1",
])
def test_render_strips_host_and_port_forms(flags):
    argv = shlex.split(render(spec(f"vllm serve Qwen/Qwen3.8-27B {flags} --max-num-seqs 64"))["serve-cmd"])
    assert not [a for a in argv if a.startswith(("--host", "--port"))]
    assert "8889" not in argv and "0.0.0.0" not in argv and "127.0.0.1" not in argv
    assert argv[-2:] == ["--max-num-seqs", "64"]


def test_render_moves_leading_env_assignments():
    out = render(spec("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FOO=1 vllm serve Qwen/Qwen3.8-27B",
                      env={"FOO": "2"}))
    assert shlex.split(out["serve-cmd"])[:2] == ["vllm", "serve"]
    assert out["serve-env"].splitlines() == ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "FOO=2"]


def test_sglang_gets_enable_metrics_once():
    out = render(spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --tp 1"))
    assert out["framework"] == "sglang"
    assert shlex.split(out["serve-cmd"]).count("--enable-metrics") == 1
    already = render(spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --enable-metrics"))
    assert shlex.split(already["serve-cmd"]).count("--enable-metrics") == 1


def test_served_model_name_both_forms():
    assert render(spec("vllm serve Qwen/Qwen3.8-27B --served-model-name q38"))["served-model-name"] == "q38"
    assert render(spec("vllm serve Qwen/Qwen3.8-27B --served-model-name=q38"))["served-model-name"] == "q38"
    assert render(spec("vllm serve Qwen/Qwen3.8-27B"))["served-model-name"] == ""


@pytest.mark.parametrize("argv,framework", [
    (["vllm", "serve", "m"], "vllm"),
    (["python3", "-m", "sglang.launch_server", "--model-path", "m"], "sglang"),
    (["python", "-m", "sglang.launch_server"], "sglang"),
    (["sglang", "serve", "--model-path", "m"], "sglang"),
])
def test_infer_framework(argv, framework):
    assert infer_framework(argv) == framework


def test_unknown_engine_rejected():
    with pytest.raises(ServingError, match="serve: unknown engine"):
        render(spec("trtllm-serve Qwen/Qwen3.8-27B"))


def test_vllm_disable_log_stats_rejected():
    with pytest.raises(ServingError, match="--disable-log-stats"):
        render(spec("vllm serve Qwen/Qwen3.8-27B --disable-log-stats"))


def test_unbalanced_quote_rejected():
    with pytest.raises(ServingError, match="serve: cannot parse"):
        split_serve("vllm serve m --x '{")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_render.py`
Expected: FAIL `ModuleNotFoundError: No module named 'infx.serving.render'`.

- [ ] **Step 3: Implement**

`infx/serving/render.py`:

```python
"""Turn a validated ServingSpec's launch command into what serve_generic.sh runs.

serve-cmd is shlex-quoted so `eval "set -- $SERVE_CMD"` in bash restores the
exact argv; --host/--port are removed here and appended by the recipe, which
owns the port.
"""

from __future__ import annotations

import re
import shlex

from infx.serving.schema import ServingError, ServingSpec

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_OWNED_FLAGS = ("--host", "--port")


def split_serve(serve: str) -> list[str]:
    joined = re.sub(r"\\[ \t]*\r?\n", " ", serve)
    try:
        return shlex.split(joined)
    except ValueError as error:
        raise ServingError(f"serve: cannot parse the command ({error}); check the quotes") from error


def _pop_env_prefix(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    index = 0
    while index < len(argv) and _ENV_ASSIGNMENT.match(argv[index]):
        key, value = argv[index].split("=", 1)
        env[key] = value
        index += 1
    return env, argv[index:]


def infer_framework(argv: list[str]) -> str:
    if argv[:2] == ["vllm", "serve"]:
        return "vllm"
    if argv[:2] == ["sglang", "serve"] or "sglang.launch_server" in argv:
        return "sglang"
    raise ServingError(
        "serve: unknown engine; the command must start with `vllm serve`, "
        "`python3 -m sglang.launch_server` or `sglang serve`"
    )


def _strip_owned_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg in _OWNED_FLAGS:
            skip_value = True
            continue
        if arg.startswith(tuple(f"{flag}=" for flag in _OWNED_FLAGS)):
            continue
        out.append(arg)
    return out


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def render(spec: ServingSpec) -> dict[str, str]:
    prefix_env, argv = _pop_env_prefix(split_serve(spec.serve))
    if not argv:
        raise ServingError("serve: empty command")
    framework = infer_framework(argv)
    if framework == "vllm" and "--disable-log-stats" in argv:
        raise ServingError("serve: drop --disable-log-stats; the benchmark scrapes vLLM metrics")
    argv = _strip_owned_flags(argv)
    if framework == "sglang" and "--enable-metrics" not in argv:
        argv.append("--enable-metrics")
    env = {**prefix_env, **spec.env}
    return {
        "framework": framework,
        "serve-cmd": shlex.join(argv),
        "serve-env": "\n".join(f"{key}={value}" for key, value in env.items()),
        "served-model-name": _flag_value(argv, "--served-model-name") or "",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_render.py utils/test_serving_schema.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add infx/serving/render.py utils/test_serving_render.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): render the serve command (host/port owned by the recipe)"
```

---

### Task 4: `to_matrix` — matrix rows + rendered serving JSON

**Files:**
- Create: `infx/serving/to_matrix.py`
- Test: `utils/test_serving_to_matrix.py`

**Interfaces:**
- Consumes: `load_serving`, `ServingError`, `REPO_ROOT` (Task 2); `render` (Task 3); scenario files (Task 1); `infx.workflows.benchmark_schema.validate_matrix` (existing).
- Produces:
  - `def scenario_facts(scenario: str, repo: Path = REPO_ROOT) -> tuple[int, str]` → (profiling duration s, public dataset loader).
  - `def build(text: str, repo: Path = REPO_ROOT) -> dict` → `{"matrix": [row, ...], "serving": {...}, "title": str}`.
  - CLI: `python -P -m infx.serving.to_matrix <file>` prints `{"matrix": [...], "serving": {...}}` (one line); `--check <file>` prints only the run title; on error prints `ERROR <line>` per problem to stderr, exit 2.
  - `serving` object keys (consumed verbatim by Task 7's `benchmark-tmpl.yml` env): `serve-cmd`, `serve-env`, `served-model-name`, `scenario`, `loader`, `yaml`.
  - Matrix row keys (exactly): `image model model-prefix precision framework runner tp pp dcp-size pcp-size ep dp-attn spec-decoding conc kv-offloading total-cpu-dram-gb duration exp-name scenario-type`. `tp` = `gpus` (benchmark-tmpl computes `GPU_COUNT = TP*PP*PCP`), `exp-name` = `<name>_c<conc>`.
  - Title: `serving <name> <scenario> (<duration>s) c<conc,...> <runner>`.

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_to_matrix.py`:

```python
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from infx.serving.to_matrix import build, scenario_facts
from infx.workflows.benchmark_schema import validate_matrix

REPO = Path(__file__).resolve().parents[1]
YAML = textwrap.dedent("""\
    name: qwen38-bf16-tp1
    runner: h200-greennode-slurm-1x_01
    image: vllm/vllm-openai:v0.29.0
    model: Qwen/Qwen3.8-27B
    gpus: 1
    env: {VLLM_X: "1"}
    serve: vllm serve Qwen/Qwen3.8-27B --port 8000 --served-model-name q38
    scenario: agentx-cc-062126-smoke
    conc: [4, 8]
""")
ROW_KEYS = {
    "image", "model", "model-prefix", "precision", "framework", "runner", "tp", "pp", "dcp-size",
    "pcp-size", "ep", "dp-attn", "spec-decoding", "conc", "kv-offloading", "total-cpu-dram-gb",
    "duration", "exp-name", "scenario-type",
}


def test_scenario_facts():
    assert scenario_facts("agentx-cc-062126") == (3600, "semianalysis_cc_traces_weka_062126")
    assert scenario_facts("agentx-cc-062126-smoke") == (180, "semianalysis_cc_traces_weka_062126")


def test_one_row_per_conc_with_generator_shape():
    result = build(YAML)
    rows = result["matrix"]
    assert [row["conc"] for row in rows] == [4, 8]
    for row in rows:
        assert set(row) == ROW_KEYS
    assert rows[1] == {
        "image": "vllm/vllm-openai:v0.29.0", "model": "Qwen/Qwen3.8-27B", "model-prefix": "qwen38-bf16-tp1",
        "precision": "bf16", "framework": "vllm", "runner": "h200-greennode-slurm-1x_01", "tp": 1, "pp": 1,
        "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False, "spec-decoding": "none", "conc": 8,
        "kv-offloading": "none", "total-cpu-dram-gb": 0, "duration": 180, "exp-name": "qwen38-bf16-tp1_c8",
        "scenario-type": "agentic-coding",
    }


def test_rows_pass_the_workflow_schema():
    validate_matrix(build(YAML)["matrix"])  # raises ValueError on any mismatch


def test_serving_object():
    serving = build(YAML)["serving"]
    assert serving == {
        "serve-cmd": "vllm serve Qwen/Qwen3.8-27B --served-model-name q38",
        "serve-env": "VLLM_X=1",
        "served-model-name": "q38",
        "scenario": "agentx-cc-062126-smoke",
        "loader": "semianalysis_cc_traces_weka_062126",
        "yaml": YAML,
    }


def test_title():
    assert build(YAML)["title"] == "serving qwen38-bf16-tp1 agentx-cc-062126-smoke (180s) c4,8 h200-greennode-slurm-1x_01"


def run_cli(*args):
    return subprocess.run([sys.executable, "-P", "-m", "infx.serving.to_matrix", *args],
                          cwd=REPO, capture_output=True, text=True, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})


def test_cli_json_and_check(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(YAML)
    out = run_cli(str(path))
    assert out.returncode == 0, out.stderr
    assert set(json.loads(out.stdout)) == {"matrix", "serving"}
    check = run_cli("--check", str(path))
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip().startswith("serving qwen38-bf16-tp1 ")


def test_cli_errors_one_per_line(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(YAML.replace("gpus: 1", "gpus: 8").replace("model: Qwen/Qwen3.8-27B", "model: /models/q"))
    out = run_cli(str(path))
    assert out.returncode == 2
    assert [line.split(":")[0] for line in out.stderr.splitlines()] == ["ERROR model", "ERROR gpus"]


def test_cli_missing_file():
    out = run_cli("/nonexistent/serving.yaml")
    assert out.returncode == 2
    assert out.stderr.startswith("ERROR serving: cannot read")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_to_matrix.py`
Expected: FAIL `ModuleNotFoundError: No module named 'infx.serving.to_matrix'`.

- [ ] **Step 3: Implement**

`infx/serving/to_matrix.py`:

```python
"""Serving YAML -> e2e-tests matrix rows + the rendered serving object.

    python -P -m infx.serving.to_matrix serving.yaml          # {"matrix": [...], "serving": {...}}
    python -P -m infx.serving.to_matrix --check serving.yaml  # the run title only

Rows have exactly the keys the generator emits for single-node agentic rows, so
they pass infx.workflows.benchmark_schema (extra="forbid"). Everything
arm-specific travels in "serving", which e2e-tests.yml hands to benchmark-tmpl.yml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from infx.serving.render import render
from infx.serving.schema import REPO_ROOT, ServingError, ServingSpec, load_serving


def scenario_facts(scenario: str, repo: Path = REPO_ROOT) -> tuple[int, str]:
    data = yaml.safe_load((repo / "scenarios" / f"{scenario}.yaml").read_text())
    benchmark = data["benchmark"]
    profiling = [phase for phase in benchmark["phases"] if phase.get("kind", phase.get("name")) == "profiling"]
    if len(profiling) != 1:
        raise ServingError(f"scenario: scenarios/{scenario}.yaml must have exactly one profiling phase")
    datasets = benchmark["datasets"]
    if len(datasets) != 1 or datasets[0].get("type") != "public":
        raise ServingError(f"scenario: scenarios/{scenario}.yaml must have exactly one public dataset")
    return int(profiling[0]["duration"]), datasets[0]["dataset"]


def _row(spec: ServingSpec, framework: str, conc: int, duration: int) -> dict:
    return {
        "image": spec.image,
        "model": spec.model,
        "model-prefix": spec.name,
        "precision": spec.precision,
        "framework": framework,
        "runner": spec.runner,
        "tp": spec.gpus,
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 1,
        "dp-attn": False,
        "spec-decoding": "none",
        "conc": conc,
        "kv-offloading": "none",
        "total-cpu-dram-gb": 0,
        "duration": duration,
        "exp-name": f"{spec.name}_c{conc}",
        "scenario-type": "agentic-coding",
    }


def build(text: str, repo: Path = REPO_ROOT) -> dict:
    spec = load_serving(text, repo)
    rendered = render(spec)
    duration, loader = scenario_facts(spec.scenario, repo)
    return {
        "matrix": [_row(spec, rendered["framework"], conc, duration) for conc in spec.conc],
        "serving": {
            "serve-cmd": rendered["serve-cmd"],
            "serve-env": rendered["serve-env"],
            "served-model-name": rendered["served-model-name"],
            "scenario": spec.scenario,
            "loader": loader,
            "yaml": text,
        },
        "title": (
            f"serving {spec.name} {spec.scenario} ({duration}s) "
            f"c{','.join(str(conc) for conc in spec.conc)} {spec.runner}"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true", help="validate and print the run title only")
    args = parser.parse_args(argv)
    try:
        try:
            text = args.path.read_text()
        except OSError as error:
            raise ServingError(f"serving: cannot read {args.path} ({error.strerror})") from error
        result = build(text)
    except ServingError as error:
        for line in str(error).splitlines():
            print(f"ERROR {line}", file=sys.stderr)
        return 2
    if args.check:
        print(result["title"])
    else:
        print(json.dumps({"matrix": result["matrix"], "serving": result["serving"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_to_matrix.py utils/test_serving_render.py utils/test_serving_schema.py`
Expected: all PASS. If `test_rows_pass_the_workflow_schema` fails, read the pydantic message: fix the row, never the schema.

- [ ] **Step 5: Commit**

```bash
git add infx/serving/to_matrix.py utils/test_serving_to_matrix.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): to_matrix — schema-valid matrix rows + rendered serving object"
```

---

### Task 5: `build_replay_cmd_yaml` + generic recipe `serve_generic.sh`

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` — add `build_replay_cmd_yaml` right after the closing `}` of `build_replay_cmd` (the line before `write_agentic_result_json() {`, currently ~line 3484). Do not change `build_replay_cmd`.
- Create: `benchmarks/single_node/agentic/serve_generic.sh` (mode 755)
- Test: `utils/test_serving_replay.py`

**Interfaces:**
- Consumes env: `INFMAX_CONTAINER_WORKSPACE MODEL PORT CONC SERVING_SCENARIO AIPERF_CLI`, optional `SERVED_MODEL_NAME AIPERF_GPU_TELEMETRY_URL AIPERF_GPU_TELEMETRY_METRICS_CSV AIPERF_SERVER_METRICS_URLS`; recipe also `FRAMEWORK RESULT_DIR SERVE_CMD SERVE_ENV SERVING_LOADER SERVING_YAML`.
- Produces: `build_replay_cmd_yaml <result_dir>` sets `REPLAY_CMD` (consumed by the existing `run_agentic_replay_and_write_outputs`), exactly:
  `$AIPERF_CLI profile --config $INFMAX_CONTAINER_WORKSPACE/scenarios/$SERVING_SCENARIO.yaml --url http://localhost:$PORT --model ${SERVED_MODEL_NAME:-$MODEL} --tokenizer $MODEL --concurrency $CONC [--gpu-telemetry URL [CSV] | --no-gpu-telemetry] [--server-metrics URL...] --output-artifact-dir <result_dir>/aiperf_artifacts`

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_replay.py`:

```python
import os
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


def test_serve_generic_is_valid_bash_and_executable():
    script = REPO / "benchmarks/single_node/agentic/serve_generic.sh"
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for needle in ('eval "set -- $SERVE_CMD"', '--host 0.0.0.0 --port "$PORT"', "build_replay_cmd_yaml",
                   "run_agentic_replay_and_write_outputs", 'WEKA_LOADER_OVERRIDE="$SERVING_LOADER"',
                   "resolve_trace_source", 'serving.yaml'):
        assert needle in text, needle
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_replay.py`
Expected: FAIL (`build_replay_cmd_yaml: command not found`, serve_generic.sh missing).

- [ ] **Step 3: Add `build_replay_cmd_yaml` to `benchmarks/benchmark_lib.sh`**

Insert after the closing `}` of `build_replay_cmd` (keep one blank line on each side):

```bash
# Serving-YAML path (docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md):
# the workload lives in scenarios/$SERVING_SCENARIO.yaml (native aiperf YAML);
# only per-run values are flags here. utils/serving/check_scenario_golden.py
# proves YAML + these flags resolve to the same config as build_replay_cmd.
build_replay_cmd_yaml() {
    check_env_vars INFMAX_CONTAINER_WORKSPACE MODEL PORT CONC SERVING_SCENARIO AIPERF_CLI
    local result_dir="$1"
    local scenario_file="$INFMAX_CONTAINER_WORKSPACE/scenarios/${SERVING_SCENARIO}.yaml"
    if [ ! -f "$scenario_file" ]; then
        echo "ERROR: scenario file not found: scenarios/${SERVING_SCENARIO}.yaml" >&2
        return 1
    fi
    # Same timeouts/UI env as build_replay_cmd (dataset configuration on H200).
    export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    export AIPERF_UI_REALTIME_METRICS_ENABLED=true
    REPLAY_CMD="$AIPERF_CLI profile --config $scenario_file"
    REPLAY_CMD+=" --url http://localhost:$PORT"
    REPLAY_CMD+=" --model ${SERVED_MODEL_NAME:-$MODEL}"
    REPLAY_CMD+=" --tokenizer $MODEL"
    REPLAY_CMD+=" --concurrency $CONC"
    if [ -n "${AIPERF_GPU_TELEMETRY_URL:-}" ]; then
        REPLAY_CMD+=" --gpu-telemetry $AIPERF_GPU_TELEMETRY_URL"
        if [ -n "${AIPERF_GPU_TELEMETRY_METRICS_CSV:-}" ]; then
            REPLAY_CMD+=" $AIPERF_GPU_TELEMETRY_METRICS_CSV"
        fi
    else
        REPLAY_CMD+=" --no-gpu-telemetry"
    fi
    if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
        REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS//,/ }"
    fi
    REPLAY_CMD+=" --output-artifact-dir $result_dir/aiperf_artifacts"
}
```

- [ ] **Step 4: Create `benchmarks/single_node/agentic/serve_generic.sh`**

```bash
#!/usr/bin/env bash
set -eo pipefail
set -x

# Generic engine recipe for the serving-YAML path
# (docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md).
# The launcher runs this instead of a per-arm recipe when SERVE_CMD is set.
# Everything arm-specific comes from infx.serving.to_matrix via env:
#   SERVE_CMD          shell-quoted engine argv, without --host/--port
#   SERVE_ENV          KEY=VALUE lines exported before the engine starts
#   SERVED_MODEL_NAME  --served-model-name from the command, if any
#   SERVING_SCENARIO   scenarios/<name>.yaml (aiperf workload, incl. duration)
#   SERVING_LOADER     public trace dataset of that scenario
#   SERVING_YAML       the user's serving YAML, saved next to the results

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL CONC PORT RESULT_DIR FRAMEWORK SERVE_CMD SERVING_SCENARIO SERVING_LOADER

mkdir -p "$RESULT_DIR"
printf '%s\n' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"

if [[ -n "${SERVE_ENV:-}" ]]; then
    while IFS= read -r assignment; do
        if [[ -n "$assignment" ]]; then
            export "$assignment"
        fi
    done <<< "$SERVE_ENV"
fi

nvidia-smi

# Validates the loader name and pre-downloads the trace dataset; the
# --public-dataset flag it builds is unused (the scenario YAML names the dataset).
export WEKA_LOADER_OVERRIDE="$SERVING_LOADER"
resolve_trace_source
install_agentic_deps
"$AIPERF_HF_CLI" download "$MODEL"

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
case "$FRAMEWORK" in
    vllm) export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:" ;;
    sglang) export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:" ;;
    *) echo "ERROR: serve_generic.sh supports vllm and sglang, got FRAMEWORK=$FRAMEWORK" >&2; exit 1 ;;
esac

SERVER_LOG="$RESULT_DIR/server.log"
eval "set -- $SERVE_CMD"
SERVER_CMD=("$@" --host 0.0.0.0 --port "$PORT")
write_command "$RESULT_DIR/server_command.txt" "${SERVER_CMD[@]}"
"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

build_replay_cmd_yaml "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
```

Then `chmod 755 benchmarks/single_node/agentic/serve_generic.sh`.

`AIPERF_HF_CLI` is set when `benchmark_lib.sh` is sourced, and `resolve_trace_source` → `ensure_hf_cli` → `install_agentic_deps` installs it, so `"$AIPERF_HF_CLI" download "$MODEL"` works after `resolve_trace_source` (verified in `benchmark_lib.sh` during planning). The model lands in `$HF_HUB_CACHE`, which the launcher mounts from `IX_HF_CACHE_HOST`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_replay.py`
Expected: all PASS.

- [ ] **Step 6: Run the golden check (Task 1) — must pass now**

Run: `(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)`
Expected:
```
OK agentx-cc-062126 (3600s): YAML + overrides == reference CLI
OK agentx-cc-062126-smoke (180s): YAML + overrides == reference CLI
```
Any `FAIL` line names the differing field: fix the scenario YAML (Task 1 files) or the flags in `build_replay_cmd_yaml`, never the reference argv.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/benchmark_lib.sh benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_replay.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): build_replay_cmd_yaml + serve_generic.sh recipe"
```

---

### Task 6: 1×H200 launcher — runner `.env` knobs + generic recipe switch

**Files:**
- Modify: `runners/launch_h200-greennode-slurm-1x.sh`
- Test: `utils/test_serving_launcher.py`

**Interfaces:**
- Consumes: runner `.env` vars `IX_HF_CACHE_HOST`, `IX_SCRATCH_ROOT`, `IX_SLURM_NODELIST` (all optional); `SERVE_CMD` from `benchmark-tmpl.yml` (Task 7).
- Produces: with `SERVE_CMD` set, `BENCH_SCRIPT=benchmarks/single_node/agentic/serve_generic.sh`; without any `IX_*`, behavior identical to today.

Current code to change (read the file first; line numbers are from `6c54ee117`):
- lines ~24–33: `case "$(hostname -s)" in ... esac` setting `SLURM_NODELIST`
- line ~36: `NODE_ROOT="/data/inferencex-$(id -u)"`
- the workspace step: `srun ... bash -c "umask 077; mkdir -p $WS $NODE_ROOT/hf $NODE_ROOT/uv $NODE_ROOT/home && tar -xz -C $WS"`
- `MOUNTS="-m $WS:/workspace -m $NODE_ROOT/hf:$HF_HUB_CACHE -m ..."`
- the `BENCH_SCRIPT` block ending with `BENCH_SCRIPT="${BENCH_BASE}.sh"` / `fi`, followed by `GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"`

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_launcher.py`:

```python
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "runners/launch_h200-greennode-slurm-1x.sh"


def text() -> str:
    return LAUNCHER.read_text()


def test_launcher_is_valid_bash():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_scratch_root_knob_keeps_uid_suffix_and_default():
    assert 'NODE_ROOT="${IX_SCRATCH_ROOT:-/data/inferencex}-$(id -u)"' in text()


def test_hf_cache_knob_defaults_to_node_root():
    t = text()
    assert 'HF_CACHE_HOST="${IX_HF_CACHE_HOST:-$NODE_ROOT/hf}"' in t
    assert "-m $HF_CACHE_HOST:$HF_HUB_CACHE" in t
    assert "-m $NODE_ROOT/hf:" not in t


def test_nodelist_knob_overrides_hostname_map():
    t = text()
    knob = t.index('if [[ -n "${IX_SLURM_NODELIST:-}" ]]; then')
    assert knob < t.index('case "$(hostname -s)" in') < t.index("salloc ")


def test_serving_switch_sits_between_formula_and_gpu_csv():
    t = text()
    formula = t.index('BENCH_SCRIPT="${BENCH_BASE}.sh"')
    switch = t.index('BENCH_SCRIPT="benchmarks/single_node/agentic/serve_generic.sh"')
    csv = t.index('GPU_METRICS_CSV="${BENCH_SCRIPT%.sh}.gpu_metrics.csv"')
    assert formula < switch < csv
    assert 'if [[ -n "${SERVE_CMD:-}" ]]; then' in t
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_launcher.py`
Expected: `test_launcher_is_valid_bash` PASS, the other four FAIL.

- [ ] **Step 3: Edit the launcher**

(a) Replace the hostname `case` block with:

```bash
# Pin salloc to the runner's own node: one runner per node, one GPU per node.
# A runner's .env may set IX_SLURM_NODELIST; otherwise map the hostname.
if [[ -n "${IX_SLURM_NODELIST:-}" ]]; then
  SLURM_NODELIST="$IX_SLURM_NODELIST"
else
  case "$(hostname -s)" in
    hoanq3-h200-1x-han-3) SLURM_NODELIST="hoanq3-h200-1x-han-3" ;;
    hoanq3-h200-1x-han-3-1) SLURM_NODELIST="hoanq3-h200-1x-han-3-1" ;;
    *)
      echo "ERROR: unrecognized 1xH200 host $(hostname -s), add it to the map in $0 or set IX_SLURM_NODELIST in the runner's .env" >&2
      exit 1
      ;;
  esac
fi
```

Keep the existing comment lines above the old `case` (about han-3 and nvidia-container-cli) right above this block.

(b) Replace `NODE_ROOT="/data/inferencex-$(id -u)"` with:

```bash
NODE_ROOT="${IX_SCRATCH_ROOT:-/data/inferencex}-$(id -u)"
# Host dir mounted as the container's HF cache. A runner's .env can point it
# at a cache shared by every runner uid (e.g. /data/hf-cache, mode 2777 + default ACL).
HF_CACHE_HOST="${IX_HF_CACHE_HOST:-$NODE_ROOT/hf}"
```

(c) In the workspace `srun ... bash -c "umask 077; mkdir -p ..."` line, replace `$NODE_ROOT/hf` with `$HF_CACHE_HOST`. (`mkdir -p` leaves an existing shared dir's mode alone.)

(d) In `MOUNTS=...`, replace `-m $NODE_ROOT/hf:$HF_HUB_CACHE` with `-m $HF_CACHE_HOST:$HF_HUB_CACHE`.

(e) Right after the `BENCH_SCRIPT` formula block (after its closing `fi`) and before `GPU_METRICS_CSV=...`, insert:

```bash
# Serving-YAML path: one generic recipe; the arm lives in SERVE_CMD & co.
if [[ -n "${SERVE_CMD:-}" ]]; then
  BENCH_SCRIPT="benchmarks/single_node/agentic/serve_generic.sh"
fi
```

No change is needed to pass `SERVE_*`/`SERVING_*` into the container: the launcher's `export -p | grep -vE ...` env file already carries every exported job variable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_launcher.py`
Expected: all PASS. Also run `python3 runners/test_launch_h200_greennode.py` (existing check; expect `OK: ...`).

- [ ] **Step 5: Commit**

```bash
git add runners/launch_h200-greennode-slurm-1x.sh utils/test_serving_launcher.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(runners): 1xH200 launcher reads runner .env knobs and runs serve_generic.sh for serving YAML"
```

---

### Task 7: Workflows — `serving` input through `e2e-tests.yml` into `benchmark-tmpl.yml`

**Files:**
- Modify: `.github/workflows/e2e-tests.yml`
- Modify: `.github/workflows/benchmark-tmpl.yml`
- Test: `utils/test_serving_workflows.py`

**Interfaces:**
- Consumes: `python -P -m infx.serving.to_matrix <file>` output `{"matrix", "serving"}` (Task 4).
- Produces: env in every serving job: `SERVE_CMD SERVE_ENV SERVED_MODEL_NAME SERVING_SCENARIO SERVING_LOADER SERVING_YAML` (consumed by Tasks 5–6). Empty strings for every non-serving job.

Facts you need (verified on this branch): `e2e-tests.yml` has `workflow_dispatch.inputs` (19 inputs; GitHub's limit is 25) and a `workflow_call.inputs` block; the `get-jobs` job has an `outputs:` map and one step `id: get-jobs` whose `run:` script is `if [ changelog refs ]; then …; else … generate_sweep_configs.py …; fi`, then pipes `CONFIG_JSON` through `python -P -m infx.workflows.benchmark_schema` and `score_matrix`, then writes outputs. The agentic fan-out job is `test-sweep-agentic` (`uses: $/.github/workflows/benchmark-tmpl.yml`, keep that `uses:` line as is). `benchmark-tmpl.yml` has `on.workflow_call.inputs` and a top-level `env:` block (e.g. `MODEL: ${{ fromJSON(inputs.config).model }}`). PyYAML parses the key `on:` as boolean `True`.

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_workflows.py`:

```python
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github/workflows"


def load(name: str) -> dict:
    data = yaml.safe_load((WF / name).read_text())
    data["on"] = data.pop(True, data.get("on"))
    return data


def test_e2e_dispatch_and_call_accept_serving():
    on = load("e2e-tests.yml")["on"]
    for trigger in ("workflow_dispatch", "workflow_call"):
        serving = on[trigger]["inputs"]["serving"]
        assert serving["type"] == "string" and serving["required"] is False and serving["default"] == ""
    assert len(on["workflow_dispatch"]["inputs"]) <= 25


def test_get_jobs_exports_serving_and_agentic_job_passes_it():
    jobs = load("e2e-tests.yml")["jobs"]
    assert jobs["get-jobs"]["outputs"]["serving"] == "${{ steps.get-jobs.outputs.serving }}"
    step = next(s for s in jobs["get-jobs"]["steps"] if s.get("id") == "get-jobs")
    assert step["env"]["SERVING_YAML"] == "${{ inputs.serving }}"
    assert step["env"]["DURATION_OVERRIDE"] == "${{ inputs.duration-override }}"
    script = step["run"]
    assert "python -P -m infx.serving.to_matrix" in script
    assert "${{" not in script.split("infx.serving.to_matrix")[0].split('if [ -n "$SERVING_YAML" ]')[1]
    assert 'echo "serving=${SERVING_JSON:-}"' in script
    assert jobs["test-sweep-agentic"]["with"]["serving"] == "${{ needs.get-jobs.outputs.serving }}"


def test_benchmark_tmpl_maps_serving_to_env():
    tmpl = load("benchmark-tmpl.yml")
    serving = tmpl["on"]["workflow_call"]["inputs"]["serving"]
    assert serving["type"] == "string" and serving["default"] == ""
    env = tmpl["env"]
    for var, key in (("SERVE_CMD", "serve-cmd"), ("SERVE_ENV", "serve-env"),
                     ("SERVED_MODEL_NAME", "served-model-name"), ("SERVING_SCENARIO", "scenario"),
                     ("SERVING_LOADER", "loader"), ("SERVING_YAML", "yaml")):
        assert env[var] == f"${{{{ inputs.serving != '' && fromJSON(inputs.serving).{key} || '' }}}}"


def test_no_ingest_in_e2e_tests():
    jobs = load("e2e-tests.yml")["jobs"]
    assert not [name for name in jobs if "ingest" in name.lower()]
    assert "ingest" not in yaml.safe_dump(jobs).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_workflows.py`
Expected: FAIL (`KeyError: 'serving'`), except `test_no_ingest_in_e2e_tests` which already passes (it guards the "no ingest" requirement from now on).

- [ ] **Step 3: Edit `e2e-tests.yml`**

(a) In **both** `workflow_dispatch.inputs` and `workflow_call.inputs`, add after `generate-cli-command` (match the file's 4-space indentation):

```yaml
            serving:
                description: "Serving YAML (serving-yaml path, see .agents/skills/serving-bench). When set, replaces generate-cli-command; duration comes from the scenario."
                required: false
                type: string
                default: ""
```

(b) In `get-jobs.outputs` add:

```yaml
            serving: ${{ steps.get-jobs.outputs.serving }}
```

(c) In the `get-jobs` step `env:` add:

```yaml
                  SERVING_YAML: ${{ inputs.serving }}
                  DURATION_OVERRIDE: ${{ inputs.duration-override }}
```

(d) In its `run:` script, turn the first `if` into a three-way branch. Replace the line

```bash
                  if [ -n "$CHANGELOG_BASE_REF" ] || [ -n "$CHANGELOG_HEAD_REF" ]; then
```

with

```bash
                  SERVING_JSON=""
                  if [ -n "$SERVING_YAML" ]; then
                    if [ -n "$DURATION_OVERRIDE" ]; then
                      echo "duration-override cannot be combined with serving: the scenario fixes the duration" >&2
                      exit 1
                    fi
                    SERVING_FILE="${RUNNER_TEMP}/serving.yaml"
                    printf '%s\n' "$SERVING_YAML" > "$SERVING_FILE"
                    SERVING_OUT=$(env PYTHONPATH="$GITHUB_WORKSPACE" \
                      uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
                      python -P -m infx.serving.to_matrix "$SERVING_FILE")
                    CONFIG_JSON=$(python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["matrix"]))' <<<"$SERVING_OUT")
                    SERVING_JSON=$(python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["serving"]))' <<<"$SERVING_OUT")
                  elif [ -n "$CHANGELOG_BASE_REF" ] || [ -n "$CHANGELOG_HEAD_REF" ]; then
```

Leave the rest of the `if … else … fi` unchanged (its `else` branch still handles `generate-cli-command`). The YAML text only ever reaches the shell through `$SERVING_YAML` (env), never through `${{ }}` inside `run:`.

(e) In the final `{ … } >> "$GITHUB_OUTPUT"` group add one line:

```bash
                    echo "serving=${SERVING_JSON:-}"
```

(`json.dumps` output is a single line, so a plain `key=value` output is safe.)

(f) In `test-sweep-agentic.with` add:

```yaml
            serving: ${{ needs.get-jobs.outputs.serving }}
```

- [ ] **Step 4: Edit `benchmark-tmpl.yml`**

(a) Under `on.workflow_call.inputs` add (match its 6-space indentation):

```yaml
      serving:
        description: "Rendered serving object from infx.serving.to_matrix (serving-yaml path only). Empty for every other job."
        required: false
        type: string
        default: ""
```

(b) In the top-level `env:` block, after `WEKA_LOADER_OVERRIDE: ...`, add:

```yaml
  # Serving-YAML path (infx.serving). Empty unless e2e-tests got a serving input;
  # runners/launch_h200-greennode-slurm-1x.sh switches to serve_generic.sh on SERVE_CMD.
  SERVE_CMD: ${{ inputs.serving != '' && fromJSON(inputs.serving).serve-cmd || '' }}
  SERVE_ENV: ${{ inputs.serving != '' && fromJSON(inputs.serving).serve-env || '' }}
  SERVED_MODEL_NAME: ${{ inputs.serving != '' && fromJSON(inputs.serving).served-model-name || '' }}
  SERVING_SCENARIO: ${{ inputs.serving != '' && fromJSON(inputs.serving).scenario || '' }}
  SERVING_LOADER: ${{ inputs.serving != '' && fromJSON(inputs.serving).loader || '' }}
  SERVING_YAML: ${{ inputs.serving != '' && fromJSON(inputs.serving).yaml || '' }}
```

Before adding `SERVED_MODEL_NAME`, run `grep -rn "SERVED_MODEL_NAME" runners/ benchmarks/*.sh .github/` and confirm every existing use is `${SERVED_MODEL_NAME:-...}` or `[ -n ... ]`-style, so an empty value behaves like unset. If some launcher sets it unconditionally from another source, note it in the commit message; do not change that launcher.

- [ ] **Step 5: Run tests + lint**

Run: `$PYTEST utils/test_serving_workflows.py`
Expected: all PASS.
Then, if available: `actionlint .github/workflows/e2e-tests.yml .github/workflows/benchmark-tmpl.yml` (install: `go install github.com/rhysd/actionlint/cmd/actionlint@latest`, or skip and say so). Pre-existing actionlint findings unrelated to your lines are not yours.

Also simulate the get-jobs serving branch locally:

```bash
SERVING_OUT=$(env PYTHONPATH="$PWD" uv run -q --no-project --python 3.12 --with pydantic --with pyyaml python -P -m infx.serving.to_matrix serving/qwen38-bf16-tp1.yaml 2>&1) || echo "$SERVING_OUT"
```

(`serving/qwen38-bf16-tp1.yaml` is created in Task 8; if you run this before Task 8, use the YAML from `utils/test_serving_to_matrix.py` in a temp file.) Then pipe the matrix through the same checks CI runs:

```bash
python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["matrix"]))' <<<"$SERVING_OUT" \
  | env PYTHONPATH="$PWD" uv run -q --no-project --python 3.12 --with pydantic --with pyyaml python -P -m infx.workflows.benchmark_schema >/dev/null && echo SCHEMA_OK
```

Expected: `SCHEMA_OK`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/e2e-tests.yml .github/workflows/benchmark-tmpl.yml utils/test_serving_workflows.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "ci(serving): serving input in e2e-tests, rendered serving env in benchmark-tmpl"
```

---

### Task 8: `dispatch.sh`, example serving YAML, skill `serving-bench`

**Files:**
- Create: `utils/serving/dispatch.sh` (mode 755)
- Create: `serving/qwen38-bf16-tp1.yaml`
- Create: `.agents/skills/serving-bench/SKILL.md`
- Create: symlink `.claude/skills/serving-bench -> ../../.agents/skills/serving-bench`
- Test: `utils/test_serving_dispatch.py`

**Interfaces:**
- Consumes: `python -P -m infx.serving.to_matrix --check` (Task 4), the `serving` + `test-name` dispatch inputs (Task 7).
- Produces: `utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]` → validates, dispatches, prints the run URL. Default ref = the current git branch, which must exist on `origin` with the same commit as local `HEAD` (CI runs the pushed code, so a stale or unpushed branch is an error).

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_dispatch.py`:

```python
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DISPATCH = REPO / "utils/serving/dispatch.sh"


def test_dispatch_is_executable_bash():
    assert os.access(DISPATCH, os.X_OK)
    subprocess.run(["bash", "-n", str(DISPATCH)], check=True)


def test_dispatch_stops_on_invalid_yaml_before_calling_gh(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nrouter: {}\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "gh").write_text("#!/bin/sh\necho GH_CALLED >&2\nexit 99\n")
    (fake_bin / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    out = subprocess.run([str(DISPATCH), str(bad), "--ref", "some-branch"], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 2
    assert "ERROR router: not supported in V1" in out.stderr
    assert "GH_CALLED" not in out.stderr


def test_example_serving_yaml_is_valid():
    out = subprocess.run(
        ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "pydantic", "--with", "pyyaml",
         "python", "-P", "-m", "infx.serving.to_matrix", "--check", "serving/qwen38-bf16-tp1.yaml"],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("serving qwen38-bf16-tp1 agentx-cc-062126-smoke (180s) c8 ")


def test_skill_is_self_contained():
    skill = (REPO / ".agents/skills/serving-bench/SKILL.md").read_text()
    assert skill.startswith("---\nname: serving-bench\n")
    for banned in ("nvidia-master", "refs/bench", "generate-cli-command", "inferencex-dispatch"):
        assert banned not in skill, banned
    assert "utils/serving/dispatch.sh" in skill
    assert (REPO / ".claude/skills/serving-bench").resolve() == (REPO / ".agents/skills/serving-bench").resolve()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYTEST utils/test_serving_dispatch.py`
Expected: FAIL (files missing).

- [ ] **Step 3: Create `utils/serving/dispatch.sh`**

```bash
#!/usr/bin/env bash
# Dispatch one serving YAML as an AgentX benchmark on vngcloud/InferenceX.
#   utils/serving/dispatch.sh serving/<name>.yaml [--ref <branch>]
# Validates locally first (same code CI runs), then starts e2e-tests.yml with the
# YAML as the `serving` input and prints the run URL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GH_REPO="vngcloud/InferenceX"

if [[ $# -lt 1 ]]; then
    echo "usage: utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]" >&2
    exit 2
fi
yaml_file="$(realpath "$1")"
shift
ref=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) ref="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

title=$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" \
    uv run -q --no-project --python 3.12 --with pydantic --with pyyaml \
    python -P -m infx.serving.to_matrix --check "$yaml_file") || exit 2

if [[ -z "$ref" ]]; then
    ref="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    remote_sha="$(git -C "$REPO_ROOT" ls-remote origin "refs/heads/$ref" | cut -f1)"
    if [[ -z "$remote_sha" ]]; then
        echo "ERROR branch $ref is not on origin; push it first (CI runs the pushed code)" >&2
        exit 2
    fi
    if [[ "$remote_sha" != "$(git -C "$REPO_ROOT" rev-parse HEAD)" ]]; then
        echo "ERROR local HEAD differs from origin/$ref; push or pull first (CI runs the pushed code)" >&2
        exit 2
    fi
fi

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gh workflow run e2e-tests.yml --repo "$GH_REPO" --ref "$ref" \
    -f serving="$(cat "$yaml_file")" \
    -f test-name="$title"

for _ in $(seq 1 30); do
    sleep 4
    run_id=$(gh run list --repo "$GH_REPO" -w e2e-tests.yml -L 20 \
        --json databaseId,displayTitle,createdAt \
        --jq "[.[] | select(.displayTitle == \"e2e Test - $title\" and .createdAt >= \"$started\")][0].databaseId // empty")
    if [[ -n "$run_id" ]]; then
        echo "https://github.com/$GH_REPO/actions/runs/$run_id"
        exit 0
    fi
done
echo "ERROR dispatched, but the run did not show up within 2 minutes; check: gh run list --repo $GH_REPO -w e2e-tests.yml -L 5" >&2
exit 1
```

`chmod 755 utils/serving/dispatch.sh`. The `run-name` of `e2e-tests.yml` is `e2e Test - ${{ inputs.test-name || ... }}`; confirm with `sed -n 2p .github/workflows/e2e-tests.yml` and adjust the `displayTitle` prefix if it differs.

- [ ] **Step 4: Create `serving/qwen38-bf16-tp1.yaml`**

The serve command is the qwen38 smoke recipe's (`git show 295131eb9:benchmarks/single_node/agentic/qwen38_bf16_h200.sh`, the `VLLM_CMD` array) with TP fixed to 1; the image is the one the green run `37733664587` used:

```yaml
# Qwen3.8-27B BF16, TP1, on one H200 — the arm of the green smoke run 37733664587,
# expressed as a serving YAML. Dispatch: utils/serving/dispatch.sh serving/qwen38-bf16-tp1.yaml
name: qwen38-bf16-tp1
runner: h200-greennode-slurm-1x_01
image: vllm/vllm-openai:v0.29.0
model: Qwen/Qwen3.8-27B
gpus: 1
precision: bf16
serve: |
  vllm serve Qwen/Qwen3.8-27B
  --served-model-name Qwen/Qwen3.8-27B
  --trust-remote-code
  --kv-cache-dtype fp8
  --tensor-parallel-size 1
  --max-model-len 262144
  --gpu-memory-utilization 0.90
  --enable-auto-tool-choice
  --enable-prefix-caching
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
  --max-num-batched-tokens 32768
  --attention-backend FLASHINFER
  --async-scheduling
  --default-chat-template-kwargs '{"enable_thinking": true}'
scenario: agentx-cc-062126-smoke
conc: [8]
```

(Newlines inside `serve: |` are plain whitespace to `shlex.split`; no backslashes needed.)

- [ ] **Step 5: Create `.agents/skills/serving-bench/SKILL.md` and the symlink**

```markdown
---
name: serving-bench
description: Benchmark an LLM serving config with the AgentX agentic-coding scenario on a VNG GPU runner, starting from a vllm/sglang launch command, a docker-compose file or a k8s manifest. Use when the user says "benchmark agentx với config…", "chạy agentx cho config này", "bench agentx", or pastes a serve command and asks for a benchmark. Writes one serving YAML and dispatches it with one command.
---

# serving-bench

You turn the user's serving config into **one YAML file** and run **one command**. Do not write recipes, edit configs, or call `gh workflow run` yourself.

## 1. Write `serving/<name>.yaml`

```yaml
name: qwen38-bf16-tp1               # [a-z0-9.-], no "_"; becomes the artifact name prefix
runner: h200-greennode-slurm-1x_01  # 1xH200 (V1 target). 8xH200: hardware-hcm-8x_01 (not validated in V1)
image: vllm/vllm-openai:v0.29.0     # required
model: Qwen/Qwen3.8-27B             # HF repo id, never a local path
gpus: 1                             # GPUs the command needs
precision: bf16                     # optional (bf16 default), naming only
env:                                # optional, the engine's env vars; quote booleans
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
serve: |                            # the launch command, as given
  vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --max-model-len 262144
  --default-chat-template-kwargs '{"enable_thinking": true}'
scenario: agentx-cc-062126-smoke    # a file name in scenarios/ (without .yaml)
conc: [4, 8]                        # one benchmark job per value
```

Full working example: `serving/qwen38-bf16-tp1.yaml`.

Translation rules:
- **Model path → HF id.** `--model-path /models/PhalaCloud/GLM-5.3-W4AFP8` becomes `--model-path PhalaCloud/GLM-5.3-W4AFP8` and `model: PhalaCloud/GLM-5.3-W4AFP8`. If you cannot tell the HF id from the path, ask.
- **Host/port:** leave them or drop them; they are replaced by the benchmark's own.
- **Env vars** (lines like `NCCL_P2P_LEVEL=NVL`, compose `environment:`, k8s `env:`) go under `env:`.
- **gpus:** tensor-parallel size (`--tp`, `--tensor-parallel-size`) × pipeline-parallel size. With sglang `--enable-dp-attention`, `--dp` does not add GPUs (`--tp 8 --dp 4` = 8 GPUs).
- **docker-compose / k8s:** `image` from `image:`; `serve` from `command:` + `args:` joined; `env` from `environment:` / `env:`.
- No shell variables (`$X`) in `serve`: write literal values.

## 2. Ask the user (do not guess) when

- the image is not given → ask for the exact image tag;
- the scenario is not given → run `ls scenarios/`, list the names (`agentx-cc-062126` = full 3600 s run, `agentx-cc-062126-smoke` = 180 s bring-up) and ask which;
- a router is part of the config → say V1 benchmarks the engine only and ask whether to continue without the router;
- `gpus` is larger than 1 for runner `h200-greennode-slurm-1x_01` → say V1 targets the 1×H200 runner and ask.

## 3. Dispatch

```bash
utils/serving/dispatch.sh serving/<name>.yaml
```

It validates first. On `ERROR <field>: <fix>` lines, fix the YAML and rerun; nothing was dispatched. On success it prints the run URL. The current git branch must be pushed (the script says so if not); do not push without the user's OK.

## 4. Results

- Status: `gh run view <run-id> --repo vngcloud/InferenceX`. One job per `conc`. Expect roughly 20 min of image pull, model load and warmup on top of the scenario duration (180 s smoke ≈ 25 min).
- Artifacts (`gh run download <run-id> --repo vngcloud/InferenceX -n <artifact>`), names start with `<name>_c<conc>`:
  - `agentic_<name>_c<conc>_…`: `results/` with `serving.yaml`, `server_command.txt`, `server.log`, `benchmark_command.txt`, `benchmark.log`, `aiperf_artifacts/`
  - `bmk_agentic_<name>_c<conc>_…`: aggregate result JSON
  - `server_logs_<name>_c<conc>_…`: server logs
- A failed job: read `server.log` first (engine flags, OOM, parser errors), then `benchmark.log` (aiperf). `gh run view <run-id> --repo vngcloud/InferenceX --log-failed | tail -100` shows the launcher side (slurm, enroot).
- Report to the user: run URL, per-conc success/failure, and where the artifacts are.
```

Then:

```bash
ln -s ../../.agents/skills/serving-bench .claude/skills/serving-bench
ls -l .claude/skills/serving-bench
```

(Check the existing links' form first with `ls -l .claude/skills/ | head`; use the same relative form.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `$PYTEST utils/test_serving_dispatch.py`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add utils/serving/dispatch.sh serving/qwen38-bf16-tp1.yaml .agents/skills/serving-bench/SKILL.md .claude/skills/serving-bench utils/test_serving_dispatch.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): dispatch.sh, qwen38 example, serving-bench skill"
```

---

### Task 9: Node setup on han-3 — shared HF cache + runner `.env` (needs user OK)

No repo files. Every command below runs on `ssh slurm-client`; node-local paths on han-3 (`/data`) are not visible from slurm-client, so node work goes through `sbatch`. **Ask the user before Step 1 (writes on the node) and before Step 3 (restarts the runner daemon).**

Facts: runner `h200-greennode-slurm-1x_01` runs as Slurm job **2362** (`gh-daemon-1x01`, user `thanglq5`, script `/shared/thanglq5/daemon-1x01.sh` = `cd /data/thanglq5/runner-1x_01 && exec ./run.sh`, 7-day limit, ends around 2026-10-15). The runner reads `/data/thanglq5/runner-1x_01/.env` at start.

- [ ] **Step 1: Create the shared cache and the `.env` (one sbatch)**

```bash
ssh slurm-client 'cat > /shared/thanglq5/ix-serving-setup.sh' <<'EOF'
#!/bin/bash
set -x
mkdir -p /data/hf-cache
chmod 2777 /data/hf-cache
if setfacl -R -d -m u::rwx,g::rwx,o::rwx /data/hf-cache; then echo ACL_OK; else echo ACL_FAIL; fi
getfacl -p /data/hf-cache
df -h /data
ENVF=/data/thanglq5/runner-1x_01/.env
touch "$ENVF"
for kv in IX_HF_CACHE_HOST=/data/hf-cache IX_SCRATCH_ROOT=/data/inferencex IX_SLURM_NODELIST=hoanq3-h200-1x-han-3; do
  grep -q "^${kv%%=*}=" "$ENVF" || echo "$kv" >> "$ENVF"
done
echo "--- $ENVF"; cat "$ENVF"
EOF
ssh slurm-client 'chmod +x /shared/thanglq5/ix-serving-setup.sh && sbatch -p test -w hoanq3-h200-1x-han-3 -c 1 --mem=1G -t 5 -J ix-serving-setup -o /shared/thanglq5/ix-serving-setup-%j.out /shared/thanglq5/ix-serving-setup.sh'
```

Wait for it (`ssh slurm-client 'squeue -u thanglq5'`; it pends while a 15-CPU bench holds the node), then `ssh slurm-client 'cat /shared/thanglq5/ix-serving-setup-<jobid>.out'`.
Expected: `ACL_OK`, a default ACL listed by `getfacl`, and the `.env` showing the three `IX_` lines (plus any lines that were there before, untouched).
If `ACL_FAIL`: add `umask 0000` as the first command after `set -x` in `benchmarks/single_node/agentic/serve_generic.sh`, with the comment `# /data on this node has no ACL support: keep the shared HF cache writable for every runner uid.`, rerun Task 5's tests, and commit that one-line change.

- [ ] **Step 2: Check the runner is idle**

```bash
gh api repos/vngcloud/InferenceX/actions/runners --paginate --jq '.runners[]|select(.name=="h200-greennode-slurm-1x_01")|"\(.status) busy=\(.busy)"'
```

Expected: `online busy=false`. If busy, wait; restarting kills the running job.

- [ ] **Step 3: Restart the daemon so the runner reads `.env` (ask the user first)**

```bash
ssh slurm-client 'scancel 2362; sleep 20; sbatch -p test -t 7-00:00:00 -c 1 --mem=4G --nodelist=hoanq3-h200-1x-han-3 --job-name=gh-daemon-1x01 -o /shared/thanglq5/gh-daemon-1x01-%j.out /shared/thanglq5/daemon-1x01.sh'
```

Note the new job id (it replaces 2362; the 7-day clock restarts). Then wait up to 2 minutes and check:

```bash
gh api repos/vngcloud/InferenceX/actions/runners --paginate --jq '.runners[]|select(.name=="h200-greennode-slurm-1x_01")|.status'
```

Expected: `online`. If it stays offline, read `/shared/thanglq5/gh-daemon-1x01-<newid>.out` on slurm-client.

- [ ] **Step 4: Record**

Tell the user (Vietnamese, short): cache path, ACL result, `.env` content, new daemon job id and its end time (`ssh slurm-client 'squeue -j <id> -o "%i %L"'`).

---

### Task 10: Push the branch (needs user OK) and run the smoke

- [ ] **Step 1: Full local check**

```bash
$PYTEST utils/test_serving_schema.py utils/test_serving_render.py utils/test_serving_to_matrix.py utils/test_serving_replay.py utils/test_serving_launcher.py utils/test_serving_workflows.py utils/test_serving_dispatch.py
(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)
python3 runners/test_launch_h200_greennode.py
git status --short     # only intended files; nothing untracked that belongs to the feature
```

Expected: all tests PASS, golden prints two `OK` lines, launcher check `OK`.

- [ ] **Step 2: Push (ask the user first; feature branch only)**

```bash
git push -u origin feat/serving-config-dispatch
```

- [ ] **Step 3: Dispatch the smoke**

```bash
utils/serving/dispatch.sh serving/qwen38-bf16-tp1.yaml
```

Expected: one URL `https://github.com/vngcloud/InferenceX/actions/runs/<id>`. The run title is `e2e Test - serving qwen38-bf16-tp1 agentx-cc-062126-smoke (180s) c8 h200-greennode-slurm-1x_01`.

- [ ] **Step 4: Watch it**

`get-jobs` finishes in about 1 minute; check it first:

```bash
gh run view <id> --repo vngcloud/InferenceX --json jobs --jq '.jobs[]|"\(.name) \(.status) \(.conclusion)"'
```

If `get-jobs` failed: `gh run view <id> --repo vngcloud/InferenceX --log-failed | tail -60`, fix, commit, ask before pushing again, redispatch. The agentic job takes about 25 minutes (pull, model download into `/data/hf-cache` on first run, load, warmup, 180 s profile). Use `gh run watch <id> --repo vngcloud/InferenceX --exit-status` in the background, or check every ~10 minutes; do not poll in a tight loop.

Failure triage, in this order:
- launcher (slurm/enroot) → `--log-failed` output; compare with the symptom table in `.agents/skills/inferencex-dispatch/SKILL.md` ("Symptom → fix");
- engine → `server.log` in the `server_logs_qwen38-bf16-tp1_c8_*` artifact;
- aiperf → `benchmark.log` in the `agentic_qwen38-bf16-tp1_c8_*` artifact (a `--config` / schema error shows at the top).

- [ ] **Step 5: Verify the success criteria**

```bash
mkdir -p /tmp/ix-smoke && cd /tmp/ix-smoke
gh run view <id> --repo vngcloud/InferenceX --json conclusion --jq .conclusion          # success
gh api repos/vngcloud/InferenceX/actions/runs/<id>/artifacts --jq '.artifacts[].name'   # agentic_/bmk_agentic_/server_logs_ qwen38-bf16-tp1_c8...
gh run download <id> --repo vngcloud/InferenceX -p 'agentic_qwen38-bf16-tp1_c8*'
find . -name serving.yaml -o -name server_command.txt -o -name benchmark_command.txt -o -name profile_export.jsonl
grep -o -- '--config [^ ]*' */results/benchmark_command.txt
grep -c '"benchmark_phase": *"profiling"' */results/aiperf_artifacts/profile_export.jsonl
```

Expected: conclusion `success`; the three artifacts exist; `serving.yaml` equals `serving/qwen38-bf16-tp1.yaml`; `benchmark_command.txt` contains `--config /workspace/scenarios/agentx-cc-062126-smoke.yaml`; profiling records > 0. (If the jsonl field name differs, open one line and count by the phase field you see.) For reference the old 90 s CLI smoke `37733664587` finished 25/25 requests with 0 errors; a 180 s run should show more requests and no errors.

- [ ] **Step 6: Report** run URL, conclusion, request/error counts, and anything that needed fixing.

---

### Task 11: Acceptance — a stateless GLM agent drives it end to end

Use the `use-greennode-glm` skill (it runs GLM-5.3 in the `opencode` CLI). Give GLM a fresh session in this repo checkout on `feat/serving-config-dispatch`, with only the `serving-bench` skill as guidance.

- [ ] **Step 1: Prompt without an image (it must ask)**

Prompt (verbatim):

```
Giúp tôi benchmark agentx với config sau, trên runner H200 1 GPU:

vllm serve Qwen/Qwen3.8-27B \
  --served-model-name Qwen/Qwen3.8-27B --trust-remote-code --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --enable-prefix-caching --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --max-num-batched-tokens 32768 --attention-backend FLASHINFER \
  --async-scheduling --default-chat-template-kwargs '{"enable_thinking": true}' \
  --host 0.0.0.0 --port 8000

Dùng concurrency 8, scenario smoke.
```

Pass: GLM asks for the image and does not dispatch.

- [ ] **Step 2: Answer with the image**

Reply: `image: vllm/vllm-openai:v0.29.0`.
Pass: GLM writes `serving/<name>.yaml` that `python -P -m infx.serving.to_matrix --check` accepts, runs `utils/serving/dispatch.sh serving/<name>.yaml`, and reports the run URL. (Its YAML must not contain `/models/`, the old `--port 8000` may be present or dropped.)

- [ ] **Step 3: Verify the run** as in Task 10 Step 5. Pass = conclusion `success`.

- [ ] **Step 4: Record the outcome** — what GLM asked, the YAML it wrote (`cat serving/<name>.yaml`), run URL, conclusion; any place it went wrong becomes a fix in `SKILL.md` (commit, then rerun this task). Do not commit GLM's YAML unless the user asks.

---

### Task 12: Wrap-up

- [ ] **Step 1:** Run the superpowers:verification-before-completion checklist against Tasks 1–11 (fresh test run + the run URLs as evidence).
- [ ] **Step 2:** Write a short handoff `/home/thanglq5/greennode-code/InferenceOptimization/docs/handoffs/<today YYYY-MM-DD>-serving-yaml-dispatch-v1.md` (Vietnamese, ≤ 60 lines; that folder sits outside the InferenceX repo — follow its `CLAUDE.md`, which keeps an index table, and add a row there): what landed (commit list), smoke + acceptance run URLs, `.env`/cache state on han-3 and the daemon job id + expiry, known gaps (router, hardware-hcm 8×H200, dashboard importer, merging into `vng-benchmark`), and that `dispatch.sh`'s default ref is the current branch.
- [ ] **Step 3:** Ask the user whether to push the last commits and whether to open a PR from `feat/serving-config-dispatch` into `vng-benchmark`. Do not merge.
