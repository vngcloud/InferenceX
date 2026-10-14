# Generic Router Sidecar (Serving-YAML V3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `router:` in a serving YAML accepts any router (image + command + env). The launcher wires it to an HTTP engine with placeholders; Dynamo keeps working unedited as an auto-detected profile.

**Architecture:** `render.py` classifies the router as profile `dynamo` (engine `dynamo.sglang` + router `dynamo.frontend`, today's behavior) or `generic` (everything else). Two new matrix fields (`router-profile`, `router-metrics-port`) travel `to_matrix` → `benchmark-tmpl.yml` env → launcher/recipes. For `generic`, the launcher allocates `ENGINE_PORT`, the engine listens there, and `serve_router.sh` expands `{{router_port}}`, `{{engine_url}}`, `{{engine_port}}` in the router command/env with plain bash string replacement.

**Tech Stack:** Python 3.12 (pydantic, pyyaml, pytest), bash, GitHub Actions workflow YAML, Slurm/pyxis launcher.

**Spec:** `docs/superpowers/specs/2026-10-14-serving-router-generic-design.md` (amended by Task 1, Step 8). Builds on `2026-10-13-serving-router-dynamo-design.md`.

**Run tests with** (no venv in this repo; verified 122 passed at baseline, ~65 s for the whole serving suite):
`uv run --with pytest --with pyyaml --with pydantic python -m pytest <paths> -q`

## Global Constraints

- Existing Dynamo YAMLs (`serving/glm53-w4afp8-dynamo*.yaml`) need no edit, and every existing field of their `build()` output stays byte-identical. Only `router-profile` and `router-metrics-port` are added (spec Success 2).
- Placeholders allowed in `router.serve` and `router.env` values: `{{router_port}}`, `{{engine_url}}`, `{{engine_port}}`. `{{engine_url}}` = `http://127.0.0.1:<ENGINE_PORT>`.
- A placeholder is exactly `{{` + lowercase letters/underscore + `}}` (no spaces). `{{ x }}` (with spaces) or `{%` is left alone (inline Jinja).
- `metrics_port`: integer 1-65535.
- Error format stays one line per problem: `<field>: <problem and fix>`.
- `router:` keeps requiring a `hardware-hcm-*` runner. One engine behind one router.
- `--numa-node`, `$VAR`, local model paths, reserved env keys stay rejected in `serve` and `router.serve`.
- `RESERVED_ENV` gains `ENGINE_PORT`, `ROUTER_PROFILE`, `ROUTER_METRICS_PORT`.
- No secrets in YAML. Never `git add -A`; add the files each task lists. Do not push without the user's OK (Task 5).

## Review Focus

Inputs the spec implies but a first pass would miss, most likely first. Each has a test in the task named.

1. Inline Jinja `{{ spaced }}` inside `router.serve` must not be rejected or rewritten (Task 1 schema test; Task 2 expand test keeps it intact).
2. Dynamo router paired with a plain engine, and a `dynamo.sglang` engine paired with a non-Dynamo router: one clear error each, not a hang at runtime (Task 1 render tests).
3. `metrics_port` set on a Dynamo router would be silently ignored: reject it (Task 1 render test).
4. Generic path launched without `ENGINE_PORT` (launcher bug) must fail loudly, not start a router pointing at `http://127.0.0.1:` (Task 2 test).
5. Router step dies early (bad command, no `bash` in a slim image): the `.router_exited` marker still appears so readiness fails fast (Task 3 test asserts the marker on the generic path).
6. `router.metrics_port` unset on a generic router falls back to the router port (Task 2 `generic-default-metrics` case).

---

### Task 1: Schema, render, matrix fields

**Files:**
- Modify: `infx/serving/schema.py` (RouterSpec ~81-92, RESERVED_ENV ~29-55, `load_serving` ~205-240)
- Modify: `infx/serving/render.py` (`_render_router` ~101-120)
- Modify: `infx/serving/to_matrix.py` (`build` serving object ~133-135)
- Create: `utils/testdata/serving_dynamo_build_golden.json`
- Test: `utils/test_serving_schema.py`, `utils/test_serving_render.py`, `utils/test_serving_to_matrix.py`
- Modify: `docs/superpowers/specs/2026-10-14-serving-router-generic-design.md` (amendments)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (later tasks rely on these exact names):
  - `RouterSpec.metrics_port: int | None`
  - `render(spec)` returns, besides V2 keys, `"router-profile"` (`"dynamo"` | `"generic"` | `""` when no router) and `"router-metrics-port"` (`"18888"`-style string, or `""`).
  - `build(text)["serving"]` carries both keys with the same names.
  - For profile `generic`, `router-cmd` keeps `--host`/`--port` flags and the literal `{{...}}` placeholders (shlex-quoted, so a placeholder token appears as `'--port={{router_port}}'`). For `dynamo`, `--http-host`/`--http-port` are still stripped (unchanged).

- [ ] **Step 1: Capture the golden of the existing Dynamo YAMLs (before any code change)**

```bash
mkdir -p utils/testdata
uv run --with pyyaml --with pydantic python - <<'PY'
import json, pathlib
from infx.serving.to_matrix import build
out = {}
for path in sorted(pathlib.Path("serving").glob("*.yaml")):
    result = build(path.read_text())
    if result["serving"]["router-cmd"]:
        out[path.stem] = result
pathlib.Path("utils/testdata/serving_dynamo_build_golden.json").write_text(
    json.dumps(out, indent=1, sort_keys=True) + "\n")
print(sorted(out))
PY
```
Expected: prints `['glm53-w4afp8-dynamo', 'glm53-w4afp8-dynamo-agentx', 'glm53-w4afp8-dynamo-agentx-c32']`.

- [ ] **Step 2: Write the failing tests**

`utils/test_serving_to_matrix.py`: add near the other tests (the file already imports `json`, `textwrap`, `pytest`, `build`, `validate_matrix`; add `REPO` only if the file lacks it — check with `grep -n "^REPO" utils/test_serving_to_matrix.py` and add `REPO = Path(__file__).resolve().parents[1]` if missing).

```python
GOLDEN = REPO / "utils/testdata/serving_dynamo_build_golden.json"
NEW_SERVING_FIELDS = ("router-profile", "router-metrics-port")


@pytest.mark.parametrize("name", sorted(json.loads(GOLDEN.read_text())))
def test_dynamo_yamls_render_as_before(name):
    result = build((REPO / "serving" / f"{name}.yaml").read_text())
    assert result["serving"]["router-profile"] == "dynamo"
    assert result["serving"]["router-metrics-port"] == ""
    for key in NEW_SERVING_FIELDS:
        del result["serving"][key]
    assert result == json.loads(GOLDEN.read_text())[name]


SGLANG_ROUTER = textwrap.dedent("""\
    name: glm53-sglr
    runner: hardware-hcm-8x_02
    image: lmsysorg/sglang:v0.5.18
    model: PhalaCloud/GLM-5.3-W4AFP8
    gpus: 8
    precision: w4afp8
    serve: python3 -m sglang.launch_server --model-path PhalaCloud/GLM-5.3-W4AFP8 --served-model-name zai-org/GLM-5.3 --tp 8 --dp 4 --ep 8 --enable-metrics
    router:
      image: sglang-router-patched:v0.5.18-slim-noargcanon
      metrics_port: 18888
      serve: python3 -m sglang_router.launch_router --port={{router_port}} --prometheus-port=18888 --worker-urls={{engine_url}}
    scenario: agentx-cc-062126
    conc: [8, 24, 32, 48]
""")


def test_generic_router_serving_object_and_rows():
    result = build(SGLANG_ROUTER)
    serving = result["serving"]
    assert serving["router-profile"] == "generic"
    assert serving["router-metrics-port"] == "18888"
    assert "{{engine_url}}" in serving["router-cmd"] and "{{router_port}}" in serving["router-cmd"]
    assert "--port" not in serving["serve-cmd"].split()
    validate_matrix(result["matrix"])
    assert [row["conc"] for row in result["matrix"]] == [8, 24, 32, 48]
    assert {row["framework"] for row in result["matrix"]} == {"sglang"}
```

In the same file update `test_serving_object`: add `"router-profile": "",` and `"router-metrics-port": "",` after `"router-env": "",` in the expected dict.

`utils/test_serving_render.py`: edit existing tests.

In `test_dynamo_pair_renders_router_fields` append:
```python
    assert out["router-profile"] == "dynamo"
    assert out["router-metrics-port"] == ""
```
Replace `test_engine_only_has_empty_router_fields` body's last line with:
```python
    assert (out["router-image"], out["router-cmd"], out["router-env"]) == ("", "", "")
    assert (out["router-profile"], out["router-metrics-port"]) == ("", "")
```
Replace `test_router_with_plain_engine_rejected` and `test_router_must_be_dynamo_frontend` with:
```python
def test_dynamo_router_with_plain_engine_rejected():
    with pytest.raises(ServingError, match=r"^serve: python3 -m dynamo.frontend needs a python3 -m dynamo.sglang engine"):
        render(dyn(engine="python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B"))


def test_dynamo_engine_with_other_router_rejected():
    with pytest.raises(ServingError, match=r"^router.serve: a python3 -m dynamo.sglang engine needs python3 -m dynamo.frontend as its router"):
        render(dyn(router="python3 -m sglang_router.launch_router --port={{router_port}}"))


def test_dynamo_router_rejects_metrics_port():
    with pytest.raises(ServingError, match=r"^router.metrics_port: not used with python3 -m dynamo.frontend"):
        render(dyn(metrics_port=18888))
```
Add:
```python
GENERIC_ROUTER = (
    "python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}} "
    "--prometheus-port=18888 --worker-urls={{engine_url}} --dp-aware --policy=cache_aware"
)


def sgl_router(router: str = GENERIC_ROUTER, **router_extra) -> ServingSpec:
    return spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --tp 8 --port 30000",
                runner="hardware-hcm-8x_02", gpus=8,
                router=RouterSpec(image="img-r", serve=router, **router_extra))


def test_generic_router_keeps_command_and_placeholders():
    out = render(sgl_router(metrics_port=18888, env={"A": "{{engine_url}}"}))
    assert out["framework"] == "sglang"
    assert out["router-profile"] == "generic"
    assert out["router-metrics-port"] == "18888"
    assert shlex.split(out["router-cmd"]) == shlex.split(GENERIC_ROUTER)
    assert out["router-env"] == "A={{engine_url}}"
    assert "--port" not in shlex.split(out["serve-cmd"])


def test_generic_router_without_metrics_port():
    assert render(sgl_router())["router-metrics-port"] == ""
```

`utils/test_serving_schema.py`: add after `test_router_image_pattern`:
```python
GENERIC_ROUTER_YAML = textwrap.dedent("""\
    router:
      image: sglang-router-patched:v0.5.18-slim-noargcanon
      metrics_port: 18888
      serve: python3 -m sglang_router.launch_router --port={{router_port}} --worker-urls={{engine_url}}
""")


def generic(serve_line: str | None = None) -> str:
    text = dynamo().removesuffix(ROUTER) + GENERIC_ROUTER_YAML
    return text.replace("--worker-urls={{engine_url}}", serve_line) if serve_line else text


def test_generic_router_block_loads():
    spec = load_serving(generic())
    assert spec.router.metrics_port == 18888
    assert "{{engine_url}}" in spec.router.serve


def test_router_unknown_placeholder_rejected():
    out = errors(generic("--worker-urls={{foo}}"))
    assert "router.serve: unknown placeholder {{foo}}; allowed: router_port, engine_url, engine_port" in out
    out = errors(generic() + "  env:\n    X: '{{bar}}'\n")
    assert "router.env.X: unknown placeholder {{bar}}" in out


def test_router_inline_jinja_is_not_a_placeholder():
    load_serving(generic("--chat-template '{{ messages }}'"))


def test_router_metrics_port_range():
    assert "router.metrics_port:" in errors(generic().replace("18888", "70000"))


def test_engine_port_reserved_env():
    out = errors(generic() + "  env:\n    ENGINE_PORT: '1'\n")
    assert "router.env.ENGINE_PORT: reserved, the benchmark recipe sets it; remove it from env" in out
```
Note: in `generic()` the router block is the last block of the text and has 2-space indented keys, so `  env:` appended with 2 spaces lands inside `router:`.

- [ ] **Step 3: Run the tests, verify they fail**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_schema.py utils/test_serving_render.py utils/test_serving_to_matrix.py -q`
Expected: FAIL (unknown key `metrics_port`, `KeyError: 'router-profile'`, old error messages, etc.).

- [ ] **Step 4: Implement the schema changes**

`infx/serving/schema.py`:

1. Add to `RESERVED_ENV` after `"ROUTER_ENV",`:
```python
    "ROUTER_PROFILE",
    "ROUTER_METRICS_PORT",
    "ENGINE_PORT",
```
2. After `ROUTER_RUNNER_PREFIX = "hardware-hcm-"` add:
```python
# {{name}} (lowercase, no spaces) in router.serve / router.env; {{ x }} is inline Jinja and stays literal.
ROUTER_PLACEHOLDERS = ("router_port", "engine_url", "engine_port")
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
```
3. Replace the `RouterSpec` class docstring and fields:
```python
class RouterSpec(BaseModel):
    """A router sidecar in front of the engine (spec 2026-10-14 §1)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)
```
(keep the existing `_env_strings` validator).
4. In `load_serving`, inside the existing `if spec.router:` block (after the `ROUTER_RUNNER_PREFIX` check) add:
```python
        texts = [("router.serve", spec.router.serve)] + [
            (f"router.env.{key}", value) for key, value in spec.router.env.items()
        ]
        for field, text in texts:
            for name in PLACEHOLDER.findall(text):
                if name not in ROUTER_PLACEHOLDERS:
                    problems.append(
                        f"{field}: unknown placeholder {{{{{name}}}}}; "
                        f"allowed: {', '.join(ROUTER_PLACEHOLDERS)}"
                    )
```
Also update the module docstring's V2 sentence to: `V2 adds an optional router: block (RouterSpec); V3 lets it be any router (placeholders, metrics_port).` and add `docs/superpowers/specs/2026-10-14-serving-router-generic-design.md (V3).` to the Specs line.

- [ ] **Step 5: Implement render + to_matrix**

`infx/serving/render.py`: replace `_render_router` with:
```python
def _render_router(spec: ServingSpec, engine_argv: list[str]) -> dict[str, str]:
    is_dynamo_engine = _python_module(engine_argv) == "dynamo.sglang"
    if spec.router is None:
        if is_dynamo_engine:
            raise ServingError(
                "router: required for python3 -m dynamo.sglang (a Dynamo engine has no HTTP port); "
                "add a router block running python3 -m dynamo.frontend"
            )
        return {
            "router-image": "", "router-cmd": "", "router-env": "",
            "router-profile": "", "router-metrics-port": "",
        }
    prefix_env, argv = _pop_env_prefix(split_serve(spec.router.serve, "router.serve"))
    if not argv:
        raise ServingError("router.serve: empty command")
    is_dynamo_router = _python_module(argv) == "dynamo.frontend"
    if is_dynamo_router and not is_dynamo_engine:
        raise ServingError("serve: python3 -m dynamo.frontend needs a python3 -m dynamo.sglang engine")
    if is_dynamo_engine and not is_dynamo_router:
        raise ServingError(
            "router.serve: a python3 -m dynamo.sglang engine needs python3 -m dynamo.frontend as its router"
        )
    _check_json_values("router.serve", argv)
    if is_dynamo_router:
        if spec.router.metrics_port is not None:
            raise ServingError(
                "router.metrics_port: not used with python3 -m dynamo.frontend "
                "(metrics come from the frontend port and the Dynamo system port); remove it"
            )
        argv = _strip_flags(argv, _ROUTER_OWNED_FLAGS)
    env = {**prefix_env, **spec.router.env}
    return {
        "router-image": spec.router.image,
        "router-cmd": shlex.join(argv),
        "router-env": "\n".join(f"{key}={value}" for key, value in env.items()),
        "router-profile": "dynamo" if is_dynamo_router else "generic",
        "router-metrics-port": "" if spec.router.metrics_port is None else str(spec.router.metrics_port),
    }
```
Also update the module docstring: after "appends them." add ` Any other router is the "generic" profile: its command is kept whole, {{router_port}} / {{engine_url}} / {{engine_port}} are expanded at runtime by serve_router.sh.`

`infx/serving/to_matrix.py`: in `build`, after `"router-env": rendered["router-env"],` add:
```python
            "router-profile": rendered["router-profile"],
            "router-metrics-port": rendered["router-metrics-port"],
```

- [ ] **Step 6: Run the serving Python tests, verify they pass**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_schema.py utils/test_serving_render.py utils/test_serving_to_matrix.py -q`
Expected: all pass, including the 3 golden cases.

- [ ] **Step 7: Run the whole serving suite**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_*.py -q`
Expected: everything passes (the shell/launcher/workflow tests do not read the new fields yet).

- [ ] **Step 8: Amend the spec**

Append to `docs/superpowers/specs/2026-10-14-serving-router-generic-design.md`:
```markdown

## Amendments (implementation plan, 2026-10-14)

- **`router-profile`** (`dynamo` | `generic` | `''`) is a second new serving-object field next to `router-metrics-port`. Profile `dynamo` = engine `python3 -m dynamo.sglang` with router `python3 -m dynamo.frontend`; the pairing is enforced (a Dynamo router needs a Dynamo engine and the reverse). Every other router is `generic`. This replaces the spec's "engine is dynamo.* **or** router is dynamo.frontend".
- **Golden:** the existing Dynamo YAMLs keep every existing field byte-identical; the two new fields are added (`utils/testdata/serving_dynamo_build_golden.json`).
- **Placeholder grammar:** exactly `{{` + `[a-z_]+` + `}}`. `{{ x }}` with spaces is inline Jinja and stays literal. `shlex.join` quotes a placeholder token, e.g. `'--port={{router_port}}'`; substitution happens before the launcher's `eval`.
- **Readiness:** generic routers use the existing `wait_for_frontend_model` (router `/v1/models` lists the served model). The separate engine `/health` wait from §3 is dropped: the router listing the model already implies a healthy engine, and the guards for engine-exited / router-exited stay.
- **Metrics:** no dedupe needed; router metrics port (or router port) and `ENGINE_PORT` never coincide.
- `metrics_port` with a Dynamo router is an error (it would be ignored).
```

- [ ] **Step 9: Commit**

```bash
git add infx/serving/schema.py infx/serving/render.py infx/serving/to_matrix.py \
  utils/testdata/serving_dynamo_build_golden.json \
  utils/test_serving_schema.py utils/test_serving_render.py utils/test_serving_to_matrix.py \
  docs/superpowers/specs/2026-10-14-serving-router-generic-design.md
git commit -m "feat(serving): generic router profile in schema/render (placeholders, metrics_port)"
```

---

### Task 2: Router and engine recipes (shell)

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (add `expand_router_placeholders` right after `wait_for_frontend_model`, ~line 700+)
- Modify: `benchmarks/single_node/agentic/serve_router.sh`
- Modify: `benchmarks/single_node/agentic/serve_generic.sh`
- Test: `utils/test_serving_router.py`

**Interfaces:**
- Consumes: env `ROUTER_PROFILE` (`dynamo`|`generic`), `ROUTER_METRICS_PORT` (string or empty), `ENGINE_PORT`, `PORT`, `ROUTER_CMD`, `ROUTER_ENV`, `DYN_SYSTEM_PORT` (Dynamo only). Task 3 provides them.
- Produces: `expand_router_placeholders TEXT` (stdout = TEXT with `{{router_port}}`, `{{engine_port}}`, `{{engine_url}}` replaced from `$PORT` / `$ENGINE_PORT`).

- [ ] **Step 1: Write the failing tests**

In `utils/test_serving_router.py`:

(a) In `test_serve_router_exports_env_appends_port_and_records_command` and `test_serve_router_never_prints_router_env_values`, add `"ROUTER_PROFILE": "dynamo",` to the env dict passed to `subprocess.run` (next to `"ROUTER_CMD": cmd`).

(b) Add:
```python
def run_generic_router(tmp_path, **extra):
    cmd = ("python3 -c 'import os,sys; print(os.environ[\"FOO\"], *sys.argv[1:])' "
           "--port={{router_port}} --worker-urls={{engine_url}} --tmpl '{{ keep }}'")
    env = {
        "PATH": "/usr/bin:/bin", "ROUTER_CMD": cmd, "ROUTER_ENV": "FOO=w{{engine_port}}\n",
        "ROUTER_PROFILE": "generic", "PORT": "9999", "ENGINE_PORT": "9200",
        "RESULT_DIR": str(tmp_path), "KV_OFFLOADING": "none",
    } | extra
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(["bash", str(ROUTER)], cwd=REPO, capture_output=True, text=True, env=env)


def test_serve_router_generic_expands_placeholders_and_appends_nothing(tmp_path):
    out = run_generic_router(tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "w9200 --port=9999 --worker-urls=http://127.0.0.1:9200 --tmpl {{ keep }}"
    assert "--http-port" not in (tmp_path / "router_command.txt").read_text()
    assert "--port=9999" in (tmp_path / "router_command.txt").read_text()


def test_serve_router_generic_needs_engine_port(tmp_path):
    out = run_generic_router(tmp_path, ENGINE_PORT=None)
    assert out.returncode != 0
    assert "ENGINE_PORT" in out.stdout + out.stderr
```
(`ENGINE_PORT=None` drops the key via the filter.) If `check_env_vars` prints its error to a different stream than stdout/stderr combined, the assertion on both covers it.

(c) Replace `test_serve_generic_branches` parametrization and asserts (the stub lib, helper code and env dict stay as they are):
```python
@pytest.mark.parametrize("mode", ["dynamo", "generic", "generic-default-metrics", "engine-only"])
def test_serve_generic_branches(tmp_path, mode):
```
Replace `if router:` (the env block) with:
```python
    if mode == "dynamo":
        env |= {"ROUTER_CMD": "python3 -m dynamo.frontend", "ROUTER_PROFILE": "dynamo", "DYN_SYSTEM_PORT": "9100"}
    elif mode.startswith("generic"):
        env |= {"ROUTER_CMD": "python3 -m sglang_router.launch_router", "ROUTER_PROFILE": "generic", "ENGINE_PORT": "9200"}
        if mode == "generic":
            env["ROUTER_METRICS_PORT"] = "18888"
```
and replace the final `if router: ... else: ...` assertion block with:
```python
    if mode == "dynamo":
        assert "WAIT frontend --port 8888 --model zai-org/GLM-5.3" in calls
        assert f"--router-exited {tmp_path}/results/.router_exited" in calls
        assert "METRICS http://localhost:8888/metrics,http://localhost:9100/metrics" in calls
        assert "--port" not in command
    elif mode.startswith("generic"):
        assert "WAIT frontend --port 8888 --model zai-org/GLM-5.3" in calls
        metrics = "18888" if mode == "generic" else "8888"
        assert f"METRICS http://localhost:{metrics}/metrics,http://localhost:9200/metrics" in calls
        assert command.strip().endswith("--host 0.0.0.0 --port 9200")
    else:
        assert "WAIT health --port 8888" in calls
        assert "METRICS http://localhost:8888/metrics" in calls
        assert command.strip().endswith("--host 0.0.0.0 --port 8888")
```

- [ ] **Step 2: Run, verify they fail**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_router.py -q`
Expected: FAIL (`expand_router_placeholders: command not found`, generic branches take the Dynamo path).

- [ ] **Step 3: Implement**

`benchmarks/benchmark_lib.sh`, directly after the closing `}` of `wait_for_frontend_model` add:
```bash
# Serving-YAML generic router: fill {{router_port}}, {{engine_port}}, {{engine_url}} in $1
# from $PORT and $ENGINE_PORT. Plain string replacement; nothing is evaluated, and any
# other {{...}} (inline Jinja) is left as it is.
expand_router_placeholders() {
    local text="$1"
    text="${text//\{\{router_port\}\}/$PORT}"
    text="${text//\{\{engine_port\}\}/$ENGINE_PORT}"
    text="${text//\{\{engine_url\}\}/http://127.0.0.1:$ENGINE_PORT}"
    printf '%s' "$text"
}
```

`benchmarks/single_node/agentic/serve_router.sh`: replace the header comment line `# Dynamo frontend for the serving-YAML router path` with `# Router sidecar for the serving-YAML router path (any router; Dynamo is the "dynamo" profile)`, and replace everything from `check_env_vars ROUTER_CMD PORT ...` to the end with:
```bash
#   ROUTER_PROFILE  dynamo: frontend argv without --http-host/--http-port, appended here;
#                   generic: the full command, {{router_port}} {{engine_url}} {{engine_port}} expanded here
# Dynamo discovery (DYN_DISCOVERY_BACKEND, DYN_FILE_KV) comes from the launcher; the
# generic profile gets ENGINE_PORT instead.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars ROUTER_CMD PORT RESULT_DIR
if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
    check_env_vars DYN_DISCOVERY_BACKEND DYN_FILE_KV
else
    check_env_vars ENGINE_PORT
fi

mkdir -p "$RESULT_DIR"
# xtrace off: ROUTER_ENV values may hold tokens; log only the names.
{ set +x; } 2>/dev/null
if [[ -n "${ROUTER_ENV:-}" ]]; then
    while IFS= read -r assignment; do
        if [[ -n "$assignment" ]]; then
            if [[ "${ROUTER_PROFILE:-}" != dynamo ]]; then
                assignment="$(expand_router_placeholders "$assignment")"
            fi
            export "$assignment"
            echo "router env: ${assignment%%=*}"
        fi
    done <<< "$ROUTER_ENV"
fi
set -x

if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
    eval "set -- $ROUTER_CMD"
    ROUTER_ARGV=("$@" --http-host 0.0.0.0 --http-port "$PORT")
else
    eval "set -- $(expand_router_placeholders "$ROUTER_CMD")"
    ROUTER_ARGV=("$@")
fi
write_command "$RESULT_DIR/router_command.txt" "${ROUTER_ARGV[@]}"
exec "${ROUTER_ARGV[@]}"
```
(Keep lines 1-3 shebang/`set` lines, and keep the existing comment lines `ROUTER_CMD` / `ROUTER_ENV` above the new `ROUTER_PROFILE` comment; delete the old two-line "Discovery ... comes from the launcher." comment since the new one replaces it.)

`benchmarks/single_node/agentic/serve_generic.sh`:
1. In the header comments, replace the `ROUTER_CMD` line with `#   ROUTER_CMD         set on the router path: the router runs in a sidecar (serve_router.sh); with ROUTER_PROFILE=dynamo the engine has no HTTP port, otherwise it listens on ENGINE_PORT` and add after it `#   ROUTER_PROFILE     dynamo | generic (launcher)\n#   ROUTER_METRICS_PORT generic router: its /metrics port, empty = the router port\n#   ENGINE_PORT        generic router: the engine's HTTP port (launcher)`.
2. Replace the metrics-URL block:
```bash
if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        check_env_vars DYN_SYSTEM_PORT
        # Frontend (dynamo_frontend_*) + engine (sglang:* via the Dynamo system port).
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics,http://localhost:${DYN_SYSTEM_PORT}/metrics"
    else
        check_env_vars ENGINE_PORT
        # Router (its own metrics port, else the router port) + the engine's own /metrics.
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics,http://localhost:${ENGINE_PORT}/metrics"
    fi
else
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
fi
```
3. Replace the `SERVER_CMD` block:
```bash
if [[ -n "${ROUTER_CMD:-}" && "${ROUTER_PROFILE:-}" == dynamo ]]; then
    SERVER_CMD=("$@")
elif [[ -n "${ROUTER_CMD:-}" ]]; then
    SERVER_CMD=("$@" --host 0.0.0.0 --port "$ENGINE_PORT")
else
    SERVER_CMD=("$@" --host 0.0.0.0 --port "$PORT")
fi
```
The readiness `if [[ -n "${ROUTER_CMD:-}" ]]` branch (`wait_for_frontend_model`) is unchanged.

- [ ] **Step 4: Run, verify they pass**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_router.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_lib.sh benchmarks/single_node/agentic/serve_router.sh \
  benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_router.py
git commit -m "feat(serving): generic router recipes (placeholder expansion, engine port, metrics urls)"
```

---

### Task 3: Launcher and workflow env

**Files:**
- Modify: `runners/launch_hardware-hcm.sh` (router block ~187-217)
- Modify: `.github/workflows/benchmark-tmpl.yml` (~229-232)
- Test: `utils/test_serving_launcher_hcm.py`, `utils/test_serving_workflows.py`

**Interfaces:**
- Consumes: serving-object fields `router-profile`, `router-metrics-port` (Task 1) as job env `ROUTER_PROFILE`, `ROUTER_METRICS_PORT`.
- Produces: for `generic`, an exported `ENGINE_PORT` (free TCP port) before the router step starts, inherited by the router and bench steps. For `dynamo`, exactly today's `DYN_*` behavior and no `ENGINE_PORT`.

- [ ] **Step 1: Write the failing tests**

`utils/test_serving_workflows.py`: replace the loop list in `test_benchmark_tmpl_maps_router_fields` with
```python
    for var, key in [
        ("ROUTER_IMAGE", "router-image"), ("ROUTER_CMD", "router-cmd"), ("ROUTER_ENV", "router-env"),
        ("ROUTER_PROFILE", "router-profile"), ("ROUTER_METRICS_PORT", "router-metrics-port"),
    ]:
```

`utils/test_serving_launcher_hcm.py`:

(a) In the `srun` stub string in `run_launcher`, append the engine port as a sixth column: change
`$(case "$*" in *serve_router.sh*) echo router;; *) echo other;; esac)" >> {envlog}'` to
`$(case "$*" in *serve_router.sh*) echo router;; *) echo other;; esac) ${{ENGINE_PORT:-none}}" >> {envlog}'`.

(b) Add `"ROUTER_PROFILE": "dynamo",` to the three existing `extra`/`extra |=` dicts that set `"ROUTER_CMD": "python3 -m dynamo.frontend"` (in `test_serving_path`, `test_dyn_system_port_fits_i16`, `test_dyn_system_port_not_passed_to_router_step`).

(c) In `test_dyn_system_port_not_passed_to_router_step`, after the existing router assertion add:
```python
    assert all(r[5] == "none" for r in rows), rows   # Dynamo profile never gets ENGINE_PORT
```
(`rows` there only holds rows whose first column is `file`, which is every row on the Dynamo path.)

(d) Add:
```python
def test_generic_router_gets_engine_port_and_no_dynamo_env(tmp_path):
    extra = {"SERVE_CMD": "python3 -m sglang.launch_server --model-path=m", "ROUTER_IMAGE": ROUTER_IMAGE,
             "ROUTER_CMD": "python3 -m sglang_router.launch_router", "ROUTER_PROFILE": "generic"}
    calls, envlog = run_launcher(tmp_path, extra)
    rows = [line.split() for line in envlog.splitlines()]
    assert rows and all(r[0] == "none" and r[1] == "none" and r[2] == "none" for r in rows), rows
    router = [r for r in rows if r[4] == "router"]
    other = [r for r in rows if r[4] == "other"]
    assert len(router) == 1 and router[0][5] != "none", rows
    assert other[-1][5] == router[0][5], rows       # the bench step sees the same ENGINE_PORT
    assert 0 < int(router[0][5]) < 65536
    assert (tmp_path / "ws/results/.router_exited").read_text().startswith("exit ")
```

- [ ] **Step 2: Run, verify they fail**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_launcher_hcm.py utils/test_serving_workflows.py -q`
Expected: FAIL (`ROUTER_PROFILE` not in tmpl env; generic launch exports `DYN_*` and no `ENGINE_PORT`).

- [ ] **Step 3: Implement the launcher**

In `runners/launch_hardware-hcm.sh`, replace the comment block above `ROUTER_SRUN_PID=""` and the opening of the router `if` through the `rm -rf` line with:
```bash
# Router sidecar (serving YAML with a router: block, spec 2026-10-13 §4, 2026-10-14 §3).
#   dynamo profile: a Dynamo frontend on $PORT; frontend and engine find each other through
#   Dynamo's file discovery in the shared workspace.
#   generic profile: any router on $PORT; the engine listens on $ENGINE_PORT and the router
#   command reaches it as {{engine_url}} (expanded by serve_router.sh).
# pyxis shares the host netns, so everything talks over localhost.
ROUTER_SRUN_PID=""
if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/workspace/results/.dynkv DYN_SYSTEM_ENABLED=true
        # engine/bench only: exported after the router srun forks so the frontend does not inherit it
        DYN_SYSTEM_PORT="$(free_tcp_port_below_32768)"
    else
        export ENGINE_PORT="$(free_tcp_port)"
    fi
    mkdir -p "$GITHUB_WORKSPACE/results"
    # A previous run's registrations would show up as ghost workers.
    rm -rf "$GITHUB_WORKSPACE/results/.dynkv" "$GITHUB_WORKSPACE/results/.router_exited"
```
(the remainder of the block — `ROUTER_SQUASH=...` through the subshell — stays.) Replace the final line of the block `    export DYN_SYSTEM_PORT` with:
```bash
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_SYSTEM_PORT
    fi
```

- [ ] **Step 4: Implement the workflow env**

In `.github/workflows/benchmark-tmpl.yml` change the comment `# Router sidecar (Dynamo frontend); only runners/launch_hardware-hcm.sh reads these.` to `# Router sidecar (any router, spec 2026-10-14); only runners/launch_hardware-hcm.sh reads these.` and after the `ROUTER_ENV:` line add:
```yaml
  ROUTER_PROFILE: ${{ inputs.serving != '' && fromJSON(inputs.serving).router-profile || '' }}
  ROUTER_METRICS_PORT: ${{ inputs.serving != '' && fromJSON(inputs.serving).router-metrics-port || '' }}
```

- [ ] **Step 5: Run the whole serving suite**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_*.py -q`
Expected: all pass (the baseline was 122; the new total is higher).

- [ ] **Step 6: Commit**

```bash
git add runners/launch_hardware-hcm.sh .github/workflows/benchmark-tmpl.yml \
  utils/test_serving_launcher_hcm.py utils/test_serving_workflows.py
git commit -m "feat(serving): launcher allocates ENGINE_PORT for generic routers; map router profile/metrics port"
```

---

### Task 4: Reference YAML, skill and docs

**Files:**
- Create: `serving/glm53-w4afp8-sglang-router.yaml`
- Modify: `.agents/skills/serving-bench/SKILL.md`
- Modify: `docs/serving-yaml-procedures.md` (line ~52), `docs/serving-yaml-procedures_zh.md` (line ~51)
- Test: `utils/test_serving_to_matrix.py::test_every_checked_in_serving_yaml_builds` (existing, picks the new file up)

**Interfaces:**
- Consumes: schema/render from Task 1 (the YAML must validate).
- Produces: a format reference for the sglang-router shape, and the skill rules the agent follows.

- [ ] **Step 1: Write the reference YAML**

Create `serving/glm53-w4afp8-sglang-router.yaml`:
```yaml
# GLM-5.3 W4AFP8 prod config (han, glm53-w4afp8-h200-8x) on hardware-hcm: SGLang 0.5.18 engine +
# sglang-router (cache_aware, dp-aware). Engine as pasted by the user on 2026-10-14, minus
# --numa-node/--host/--port, with the HF id as model path. The router reaches the engine through
# {{engine_url}} (the prod k8s worker-service URL has no meaning on the runner).
# NVSHMEM_REMOTE_TRANSPORT=none: same reason as serving/glm53-w4afp8-dynamo.yaml (single-node DeepEP off IB).
# Prod's SGLANG_TORCH_PROFILER_DIR / SGLANG_PROFILE_WITH_STACK are left out (profiling is off).
# router.image is the name as pasted; the registry prefix it needs on the runner is unconfirmed,
# see Task 5 of docs/superpowers/plans/2026-10-14-serving-router-generic.md.
# Format reference only: do not dispatch it or copy its values for another request.
name: glm53-w4afp8-sglang-router
runner: hardware-hcm-8x_02
image: lmsysorg/sglang:v0.5.18
model: PhalaCloud/GLM-5.3-W4AFP8
gpus: 8
precision: w4afp8
env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  SGLANG_DP_USE_GATHERV: "1"
  SGLANG_ENABLE_METRICS_DP_ATTENTION: "1"
  NCCL_P2P_LEVEL: NVL
  HF_HUB_OFFLINE: "1"
  NVSHMEM_REMOTE_TRANSPORT: none
serve: |
  python3 -m sglang.launch_server
  --model-path PhalaCloud/GLM-5.3-W4AFP8
  --quantization w4afp8
  --tp 8 --dp 4 --ep 8
  --enable-dp-attention --enable-dp-attention-local-control-broadcast --enable-dp-lm-head
  --tokenizer-worker-num 8
  --dist-init-addr 127.0.0.1:10888
  --moe-a2a-backend deepep
  --chunked-prefill-size 32768
  --tool-call-parser glm47 --reasoning-parser glm45
  --mem-fraction-static 0.75 --max-running-requests 256 --context-length 300000
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_sparse_q8
  --allow-auto-truncate
  --enable-metrics --enable-metrics-for-all-schedulers --enable-cache-report
  --enable-hierarchical-cache --hicache-size 128 --hicache-io-backend direct --hicache-write-policy write_back
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --schedule-policy dfs-weight --enable-prefill-delayer
  --served-model-name zai-org/GLM-5.3
router:
  image: sglang-router-patched:v0.5.18-slim-noargcanon
  metrics_port: 18888
  serve: |
    python3 -m sglang_router.launch_router
    --host=0.0.0.0 --port={{router_port}} --prometheus-port=18888
    --policy=cache_aware --dp-aware --worker-urls={{engine_url}}
    --cache-threshold=0.3 --balance-abs-threshold=100000 --balance-rel-threshold=2.0
    --max-tree-size=67108864 --eviction-interval=300
    --health-check-interval-secs=15 --health-check-timeout-secs=10 --health-failure-threshold=5
    --request-timeout-secs=900 --retry-max-retries=2
scenario: agentx-cc-062126
conc: [8, 24, 32, 48]
```

- [ ] **Step 2: Validate it**

Run: `uv run --with pyyaml --with pydantic python -P -m infx.serving.to_matrix --check serving/glm53-w4afp8-sglang-router.yaml`
Expected: prints `serving glm53-w4afp8-sglang-router agentx-cc-062126 (3600s) c8,24,32,48 hardware-hcm-8x_02`. If it prints `ERROR ...` lines, fix the YAML, not the schema.

- [ ] **Step 3: Update the skill**

In `.agents/skills/serving-bench/SKILL.md`:

1. In the YAML block, replace the line
`router:                             # optional; only a Dynamo frontend in front of python3 -m dynamo.sglang`
and its `image/env/serve` sub-lines (`  image: <frontend image>` … `    python3 -m dynamo.frontend --router-mode=kv ...`) with:
```yaml
router:                             # optional: any router in front of the engine (hardware-hcm runner)
  image: <router image>
  env: {}                           # the router's env vars
  metrics_port: 18888               # optional: the router's /metrics port; default = the router port
  serve: |                          # {{router_port}} = the port the benchmark calls, {{engine_url}} = the engine
    python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}}
    --worker-urls={{engine_url}} --policy=cache_aware
```
2. Replace the sentence `Full working examples: … and `serving/glm53-w4afp8-dynamo.yaml` (Dynamo router + engine). Format references only: …` with:
`Full working examples: `serving/qwen38-bf16-tp1.yaml` (engine only), `serving/glm53-w4afp8-dynamo.yaml` (Dynamo router + engine) and `serving/glm53-w4afp8-sglang-router.yaml` (sglang-router + SGLang engine). Format references only: never dispatch them or copy their values for a user's request.`
3. Replace the translation rule starting `- **Router (Dynamo):**` with:
```
- **Router:** a router part goes under `router:` with its own `image`; the engine stays in `serve`. Engine env → `env:`, router env → `router.env:`.
  - **Dynamo** (`python3 -m dynamo.frontend` + `python3 -m dynamo.sglang`): as before. Leave out discovery settings (`DYN_*`, `ETCD_ENDPOINTS`, NATS) and `--http-host/--http-port`: the benchmark wires them.
  - **Any other router** (sglang-router and others): keep the user's flags verbatim except the wiring. Listen host/port flags → `--host=0.0.0.0 --port={{router_port}}`. A worker/backend URL (k8s service, hostname) → `{{engine_url}}` (or `{{engine_port}}` where only a port fits). A literal metrics/prometheus port flag → also `router.metrics_port: <port>`. The engine's own `--host/--port` are dropped, as for any engine.
  - **Router command missing** (only flags + image): infer it from the flag family and say so in the reply so the user can correct it. `--worker-urls`, `--policy=cache_aware`, `--dp-aware`, `--prometheus-port` → `python3 -m sglang_router.launch_router`. Unknown family → ask.
  - Placeholders exist only in `router.serve` / `router.env`. No `$VAR` there either.
```
4. Add after the `**Drop `--numa-node …`**` rule:
```
- **DeepEP on `hardware-hcm-*`** (`--moe-a2a-backend deepep`): add `NVSHMEM_REMOTE_TRANSPORT: none` to `env:`. The container sees InfiniBand devices and single-node NVSHMEM would segfault on IBGDA/IBRC.
```
5. Replace the first bullet of "## 2. Ask the user" (`- the image is not given → ask …`) with:
```
- the image is not given and no version either → ask for the exact image tag. Only the user's message counts: never take the image from `serving/*.yaml`, the example above or past runs, and never reuse an existing `serving/` file for a new request — write a new one;
- a **version** is given but no image (e.g. "SGLang 0.5.18") → resolve a public image yourself, do not ask: SGLang → `lmsysorg/sglang:v<ver>`, vLLM → `vllm/vllm-openai:v<ver>`. Verify the tag exists before writing the YAML: `curl -s "https://hub.docker.com/v2/repositories/<repo>/tags?name=<tag>&page_size=50"` and check the exact tag is in `results[].name`. Variants (`-cu129`, `-cu130`, `-runtime`) exist → take the bare tag and name it in the reply. Tag not found → stop and ask. A router image is never resolved this way (it is the user's own build): use it as given; a bare name with no registry prefix → ask for the full reference;
```
6. Delete the bullet `- a router that is not `dynamo.frontend`, or an engine that is not `dynamo.sglang` next to it → say only the Dynamo pair is supported and ask;` and in the "Runner" rule keep `A router (`router:`) needs a `hardware-hcm-*` runner.` unchanged.
7. Add to §2 a last bullet: `- a Dynamo router paired with a non-Dynamo engine, or the reverse → the validator rejects it; tell the user the Dynamo frontend only works with `dynamo.sglang`, and ask.`

- [ ] **Step 4: Update the docs links**

`docs/serving-yaml-procedures.md` line ~52: after the sentence listing the two examples, change `and [`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml) (Dynamo router + engine).` to `[`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml) (Dynamo router + engine) and [`serving/glm53-w4afp8-sglang-router.yaml`](../serving/glm53-w4afp8-sglang-router.yaml) (sglang-router + SGLang engine).`
`docs/serving-yaml-procedures_zh.md` line ~51: change `与 [`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml)（Dynamo Router + 引擎）。` to `、[`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml)（Dynamo Router + 引擎）与 [`serving/glm53-w4afp8-sglang-router.yaml`](../serving/glm53-w4afp8-sglang-router.yaml)（sglang-router + SGLang 引擎）。`

- [ ] **Step 5: Run the suite**

Run: `uv run --with pytest --with pyyaml --with pydantic python -m pytest utils/test_serving_*.py -q`
Expected: all pass, including `test_every_checked_in_serving_yaml_builds[glm53-w4afp8-sglang-router.yaml]`.

- [ ] **Step 6: Commit**

```bash
git add serving/glm53-w4afp8-sglang-router.yaml .agents/skills/serving-bench/SKILL.md \
  docs/serving-yaml-procedures.md docs/serving-yaml-procedures_zh.md
git commit -m "docs(serving): sglang-router reference YAML; skill rules for generic routers and public image resolution"
```

---

### Task 5: Real verification (gated)

Not code. This task needs the user: pushing the branch and choosing the router image reference. **Stop and ask before any push** (user's global rule).

- [ ] **Step 1: Ask the user for two things**

1. The full registry reference of the router image (the paste only has `sglang-router-patched:v0.5.18-slim-noargcanon`; `enroot import` of a bare name goes to Docker Hub and will fail). If it is private on VCR, the runner's `60108-khanghd` account must be able to read it (see `docs/serving-yaml-procedures.md`).
2. OK to push `vng-benchmark` to origin (CI runs the pushed code).

- [ ] **Step 2: Build the smoke YAML**

```bash
sed -e 's#image: sglang-router-patched:v0.5.18-slim-noargcanon#image: <FULL REFERENCE FROM THE USER>#' \
    -e 's#^scenario: agentx-cc-062126$#scenario: agentx-cc-062126-smoke#' \
    -e 's#^conc: \[8, 24, 32, 48\]#conc: [8]#' \
    serving/glm53-w4afp8-sglang-router.yaml > /tmp/glm53-sglr-smoke.yaml
ssh slurm-client squeue -p hardware-hcm
```
Pick the free box (`hardware-hcm-8x_01` or `_02`, edit `runner:` in the temp file). If both are held, tell the user the run waits in the Slurm queue.

- [ ] **Step 3: Dispatch (after the user's push OK)**

```bash
git push origin vng-benchmark   # only after the user says yes
utils/serving/dispatch.sh /tmp/glm53-sglr-smoke.yaml
```

- [ ] **Step 4: Verify with evidence**

Download the artifacts (`gh run download <run-id> --repo vngcloud/InferenceX -n <artifact>`) and check, in this order: `router.log` shows the router started and registered the worker; `router_command.txt` holds `--port=<PORT>` and `--worker-urls=http://127.0.0.1:<ENGINE_PORT>` with no `{{`; `server_command.txt` ends with `--host 0.0.0.0 --port <ENGINE_PORT>`; `benchmark.log` shows `sglang:*` metrics were scraped from both URLs; the job is green. Failure reading order and the "do not edit the user's config" rule are in the skill's §4.
Then use `superpowers:verification-before-completion` before claiming done, and report the run URL.

- [ ] **Step 5: Real run (only if the user asks)**

Dispatch `serving/glm53-w4afp8-sglang-router.yaml` with the full router image (ladder `8 24 32 48`, scenario `agentx-cc-062126`) the same way; it takes about an hour plus ~20 min startup per conc.

---

## Self-review

**Spec coverage:** §1 schema → Task 1 (RouterSpec.metrics_port, placeholders, RESERVED_ENV; the "unknown placeholder" message matches the spec). §2 render/engine port → Task 1 (render, matrix fields) + Task 2 (`--port $ENGINE_PORT`). §3 launch → Task 2 (serve_router, metrics URLs, readiness reuse) + Task 3 (ENGINE_PORT, profile gating, workflow env). §4 tests → Tasks 1-3 (schema/render/golden/fixture build, shell expansion tests, launcher tests); real run → Task 5. §5 skill → Task 4. Risks (Docker Hub rate limit, router entrypoint, `--dp-aware`) are exercised only by Task 5; the slim-image `bash` risk is covered by Review Focus 5. Spec deviations are recorded in Task 1's amendments.

**Type consistency:** field names `router-profile` / `router-metrics-port` (serving object) ↔ `ROUTER_PROFILE` / `ROUTER_METRICS_PORT` (env) ↔ `RouterSpec.metrics_port`; `ENGINE_PORT` and `expand_router_placeholders` are named the same in Tasks 2 and 3; the profile values `dynamo` / `generic` are the same everywhere.
