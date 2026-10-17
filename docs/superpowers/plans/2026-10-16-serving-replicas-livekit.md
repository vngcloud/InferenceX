# Serving YAML replicas + LiveKit 111868 scenarios — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A serving YAML can run N engine replicas behind any router (`replicas: N`, `{{engine_urls}}`), and Gemma-4-31B is benchmarked for customer 111868 on `hardware-hcm` 8×H200 in two shapes (1 GPU direct; 2 replicas + vllm-router with `cache_aware` and with `consistent_hash`) on the LiveKit V1/V2 datasets.

**Architecture:** `replicas` is a new ServingSpec field. `to_matrix` passes it in the `serving` object (matrix `tp` stays = total `gpus` because `GPU_COUNT = TP×PP×PCP` drives `salloc`). `benchmark-tmpl.yml` exports `REPLICAS`; `launch_hardware-hcm.sh` allocates one engine port per replica (`ENGINE_PORTS`); `serve_generic.sh` starts N copies of `SERVE_CMD` on GPU slices; `expand_router_placeholders` fills `{{engine_urls}}`. Scenarios and lock entries for the dataset come last, after the dataset track publishes it.

**Tech Stack:** bash recipes, Python 3.12 (pydantic, pyyaml), aiperf fork `utils/aiperf` @ `faf3f8b3`, GitHub Actions (`dispatch.sh`), Slurm `hardware-hcm`, vLLM `v0.25.1`, vllm-router `0.1.14`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-serving-replicas-livekit-design.md` (read first). Dataset track: `vmonitor-gateway-datasets/docs/plans/0004-livekit-voice-agent-synth-traces-plan.md` (runs in parallel; hands over repo/revision/file/sha256 for Task 8).

## How to start (fresh session)

```bash
cd /home/thanglq5/greennode-code/InferenceOptimization/InferenceX
git fetch origin
git worktree add ../InferenceX-livekit -b feat/serving-replicas-livekit origin/vng-benchmark
mv docs/superpowers/specs/2026-10-16-serving-replicas-livekit-design.md ../InferenceX-livekit/docs/superpowers/specs/
mv docs/superpowers/plans/2026-10-16-serving-replicas-livekit.md ../InferenceX-livekit/docs/superpowers/plans/
cd ../InferenceX-livekit
git submodule update --init utils/aiperf      # must be faf3f8b35760a7e2f9eadcc04fae66ebdbd3b765
```

- The main checkout `InferenceOptimization/InferenceX` belongs to other sessions (it is behind origin and has an untracked `serving/glm53-w4afp8-sglang-router-c108120.yaml` that is not ours). After the two `mv` above, never edit or commit there.
- PYTEST (serving tests only):
  `uv run -q --python 3.12 --with pytest --with pyyaml --with pydantic --no-project python -m pytest -c utils/pytest.ini --rootdir utils utils/test_serving_*.py -q`
- GOLDEN: `(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)`
- Commits: `git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit ...`, subject `feat(serving): …` like the recent fork commits; `git add` only the task's files; never `git add -A`/`.`.
- **Never push without the user's OK** (`dispatch.sh` needs the branch pushed: ask once in Task 7, quoting the branch name). Never print a token.
- AGENTS.md test rule: tests run the real code with concrete inputs; no test that greps source or config text. Shell scripts are tested with stubbed binaries on `PATH`.

## Global Constraints

- `replicas: int`, default 1, `≥ 1`. `gpus % replicas == 0`. `replicas > 1` requires a `router:` block and not the Dynamo frontend.
- Placeholder `{{engine_urls}}` = space-separated `http://127.0.0.1:<port>` for every engine, in replica order; `{{engine_url}}` stays = the first engine.
- Matrix row `tp` = total `gpus` (not per replica). `serving["replicas"]` is a string (`"1"`, `"2"`), like the other serving fields.
- Replica 0 logs to `server.log` and records `server_command.txt` (the vLLM server-metrics adapter reads `GPU KV cache size` from `server.log`, prior art `gemma4tau2_fp8_h200.sh`); replica r>0 → `server.<r>.log`, `server_command.<r>.txt`.
- Replica r gets GPUs `[r·k, (r+1)·k)` of the job's GPU list (`CUDA_VISIBLE_DEVICES` if set, else `nvidia-smi --query-gpu=index`), `k = gpus/replicas`.
- Engine (all Gemma YAMLs): `vllm/vllm-openai:v0.25.1`, prod flags minus MTP (spec §6). Router: same image, `python3 -m pip install -q vllm-router==0.1.14` then `vllm-router … --prometheus-port 29000`, `router.metrics_port: 29000`.
- Dataset scenarios: `entries: 1217`, no `requests`, `duration` 1200 (smoke 180), think time on = `traceIdleGapCapSeconds: 300.0`, off = `0.0` (NOT `ignoreTraceDelays`: the mooncake loader ignores it).
- Lock keys `livekit-111868-v1` / `livekit-111868-v2`, repo `Green-Lab/2026-10-16-111868-agentic-traces`, files `trace.v1-append-ideal.jsonl` / `trace.v2-observed.jsonl`.
- SLO for sizing: TTFT p95 ≤ 0.5 s (negotiable); target 300 concurrent rooms.

---

### Task 0: Commit spec + plan on the new branch

- [ ] **Step 1:** in `../InferenceX-livekit`:

```bash
git add docs/superpowers/specs/2026-10-16-serving-replicas-livekit-design.md docs/superpowers/plans/2026-10-16-serving-replicas-livekit.md
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "docs(serving): spec + plan for serving-YAML replicas and LiveKit 111868 scenarios"
```

---

### Task 1: Schema — `replicas` and `{{engine_urls}}`

**Files:**
- Modify: `infx/serving/schema.py` (`RESERVED_ENV`, `ROUTER_PLACEHOLDERS`, `ServingSpec`, `load_serving`)
- Test: `utils/test_serving_schema.py`

**Interfaces:**
- Produces: `ServingSpec.replicas: int`; router placeholder name `engine_urls`; reserved env `REPLICAS`, `ENGINE_PORTS`.

- [ ] **Step 1: Write the failing tests** (append to `utils/test_serving_schema.py`; it already defines `GOOD`, `errors`, `dynamo`)

```python
REPLICAS_YAML = textwrap.dedent("""\
    name: gemma4-2rep
    runner: hardware-hcm-8x_01
    image: vllm/vllm-openai:v0.25.1
    model: RedHatAI/gemma-4-31B-it-FP8-block
    gpus: 2
    replicas: 2
    serve: vllm serve RedHatAI/gemma-4-31B-it-FP8-block --tensor-parallel-size 1
    router:
      image: vllm/vllm-openai:v0.25.1
      serve: vllm-router --port {{router_port}} --worker-urls {{engine_urls}} --policy consistent_hash
    scenario: agentx-cc-062126-smoke
    conc: [8]
""")


def test_replicas_default_one_and_router_yaml_loads():
    assert load_serving(GOOD).replicas == 1
    assert load_serving(REPLICAS_YAML).replicas == 2


@pytest.mark.parametrize("text, error", [
    (REPLICAS_YAML.replace("gpus: 2", "gpus: 3"), "replicas: gpus 3 is not a multiple of replicas 2"),
    (REPLICAS_YAML.split("router:")[0] + "scenario: agentx-cc-062126-smoke\nconc: [8]\n",
     "replicas: more than 1 replica needs a router: block"),
    (dynamo().replace("gpus: 8", "gpus: 8\nreplicas: 2"), "replicas: the Dynamo frontend discovers its engines itself"),
    (REPLICAS_YAML.replace("replicas: 2", "replicas: 0"), "replicas: Input should be greater than or equal to 1"),
    (REPLICAS_YAML.replace("gpus: 2\n", "gpus: 2\nenv: {ENGINE_PORTS: x}\n"), "env.ENGINE_PORTS: reserved"),
], ids=["indivisible", "no-router", "dynamo", "zero", "reserved-env"])
def test_replicas_errors(text, error):
    assert error in errors(text)
```

- [ ] **Step 2: Run, expect failure** — `<PYTEST> -k replicas` → `replicas` unknown key / missing errors.

- [ ] **Step 3: Implement** in `infx/serving/schema.py`:
  - `RESERVED_ENV`: after `"ENGINE_PORT",` add `"ENGINE_PORTS",` and `"REPLICAS",`.
  - `ROUTER_PLACEHOLDERS = ("router_port", "engine_url", "engine_port", "engine_urls", "storage_master_port", "storage_meta_port",)` — `engine_urls` goes **after** `engine_port` (the existing `test_router_unknown_placeholder_rejected` asserts the prefix `router_port, engine_url, engine_port`); keep the existing comment and add `engine_urls` = every replica's URL.
  - Pre-flight evidence (2026-10-16, Tasks 1–5 applied verbatim to a scratch worktree of `origin/vng-benchmark@263ca3ecf`): serving suite 163 passed (151 before + 12 new).
  - `ServingSpec`, right after `gpus: int = Field(gt=0)`:

```python
    replicas: int = Field(default=1, ge=1)
```

  - `load_serving`, right after the `if len(set(spec.conc)) != len(spec.conc):` check:

```python
    if spec.gpus % spec.replicas:
        problems.append(
            f"replicas: gpus {spec.gpus} is not a multiple of replicas {spec.replicas}; "
            "each replica gets gpus/replicas GPUs"
        )
    if spec.replicas > 1:
        if spec.router is None:
            problems.append(
                "replicas: more than 1 replica needs a router: block whose serve lists {{engine_urls}}"
            )
        elif "dynamo.frontend" in spec.router.serve:
            problems.append(
                "replicas: the Dynamo frontend discovers its engines itself; use replicas: 1"
            )
```

  - Module docstring: add one line `V4 adds replicas: N engines behind the router (spec 2026-10-16).`

- [ ] **Step 4: Run, expect pass** — `<PYTEST>` (whole serving suite green).

- [ ] **Step 5: Commit**

```bash
git add infx/serving/schema.py utils/test_serving_schema.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): replicas field and {{engine_urls}} placeholder"
```

---

### Task 2: `to_matrix` + workflow carry `replicas`

**Files:**
- Modify: `infx/serving/to_matrix.py` (`build()` serving dict)
- Modify: `.github/workflows/benchmark-tmpl.yml` (env block, after `ROUTER_METRICS_PORT`)
- Test: `utils/test_serving_to_matrix.py`

**Interfaces:**
- Consumes: `ServingSpec.replicas` (Task 1).
- Produces: `serving["replicas"]: str`; job env `REPLICAS`.

- [ ] **Step 1: Update/write the tests** in `utils/test_serving_to_matrix.py`:
  - In `test_serving_object`, add `"replicas": "1",` to the expected dict (after `"router-metrics-port": "",`).
  - Add `"replicas"` to `NEW_SERVING_FIELDS` (the Dynamo golden JSON predates it).
  - Append:

```python
def test_replicas_travel_in_serving_and_tp_stays_total():
    text = (YAML.replace("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01")
                .replace("gpus: 1", "gpus: 2\nreplicas: 2")
            + "router:\n  image: vllm/vllm-openai:v0.25.1\n"
              "  serve: vllm-router --port {{router_port}} --worker-urls {{engine_urls}}\n")
    result = build(text)
    assert result["serving"]["replicas"] == "2"
    assert {row["tp"] for row in result["matrix"]} == {2}
    assert "{{engine_urls}}" in result["serving"]["router-cmd"]
```

- [ ] **Step 2: Run, expect failure** — `<PYTEST> -k "serving_object or replicas"` → KeyError / dict mismatch.

- [ ] **Step 3: Implement**
  - `infx/serving/to_matrix.py`, in `build()`'s `"serving": {...}` dict, after `"router-metrics-port": rendered["router-metrics-port"],` add `"replicas": str(spec.replicas),`. `_row` is unchanged (`"tp": spec.gpus`).
  - `.github/workflows/benchmark-tmpl.yml`, right after the `ROUTER_METRICS_PORT:` env line add (same indentation):

```yaml
  REPLICAS: ${{ inputs.serving != '' && fromJSON(inputs.serving).replicas || '' }}
```

- [ ] **Step 4: Run, expect pass** — `<PYTEST>`; also `python -c "import yaml; yaml.safe_load(open('.github/workflows/benchmark-tmpl.yml'))"`.

- [ ] **Step 5: Commit**

```bash
git add infx/serving/to_matrix.py .github/workflows/benchmark-tmpl.yml utils/test_serving_to_matrix.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): carry replicas to the job env (tp stays total GPUs)"
```

---

### Task 3: `{{engine_urls}}` expansion

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (`expand_router_placeholders`, ~line 708)
- Test: `utils/test_serving_router.py`

**Interfaces:**
- Consumes: env `ENGINE_PORTS` (space-separated; Task 4 sets it), falls back to `ENGINE_PORT`.

- [ ] **Step 1: Write the failing test** (append to `utils/test_serving_router.py`; `run_generic_router` exists there)

```python
@pytest.mark.parametrize("ports, urls", [
    ("9200 9201", "http://127.0.0.1:9200 http://127.0.0.1:9201"),
    (None, "http://127.0.0.1:9200"),
], ids=["replicas", "single-engine"])
def test_serve_router_expands_engine_urls(tmp_path, ports, urls):
    cmd = "python3 -c 'import sys; print(*sys.argv[1:])' --worker-urls {{engine_urls}}"
    out = run_generic_router(tmp_path, ROUTER_CMD=cmd, ENGINE_PORTS=ports)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == f"--worker-urls {urls}"
```

- [ ] **Step 2: Run, expect failure** — `<PYTEST> -k engine_urls` → output still contains `{{engine_urls}}`.

- [ ] **Step 3: Implement** — in `expand_router_placeholders`, after the `{{engine_url}}` line:

```bash
    local urls="" engine_port
    for engine_port in ${ENGINE_PORTS:-$ENGINE_PORT}; do
        urls+="${urls:+ }http://127.0.0.1:$engine_port"
    done
    text="${text//\{\{engine_urls\}\}/$urls}"
```

(`{{engine_url}}` cannot match inside `{{engine_urls}}`: the `}}` must follow `url`.) Update the comment above `expand_router_placeholders` to list `{{engine_urls}}`.

- [ ] **Step 4: Run, expect pass** — `bash -n benchmarks/benchmark_lib.sh && <PYTEST>`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_lib.sh utils/test_serving_router.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): expand {{engine_urls}} for replica routers"
```

---

### Task 4: Launcher — one port per replica

**Files:**
- Modify: `runners/launch_hardware-hcm.sh` (generic-router branch, `export ENGINE_PORT="$(free_tcp_port)"`, ~line 262)
- Test: `utils/test_serving_launcher_hcm.py`

**Interfaces:**
- Consumes: env `REPLICAS` (Task 2).
- Produces: env `ENGINE_PORTS` (space-separated, first == `ENGINE_PORT`), exported before the router and bench sruns.

- [ ] **Step 1: Write the failing test**
  - In `run_launcher`'s `stubs`, append to the `srun` body (keep the existing echo): `; echo "${{ENGINE_PORTS:-none}}" >> {tmp_path}/ports` — i.e. the `"srun"` value becomes the old f-string + `f'\necho "${{ENGINE_PORTS:-none}}" >> {tmp_path}/ports'`.
  - Append:

```python
def test_generic_router_replicas_get_distinct_engine_ports(tmp_path):
    extra = {"SERVE_CMD": "vllm serve m", "ROUTER_IMAGE": "vllm/vllm-openai:v0.25.1",
             "ROUTER_CMD": "vllm-router --worker-urls {{engine_urls}}", "ROUTER_PROFILE": "generic",
             "REPLICAS": "2"}
    _, envlog = run_launcher(tmp_path, extra)
    seen = {line for line in (tmp_path / "ports").read_text().splitlines() if line != "none"}
    assert len(seen) == 1
    ports = seen.pop().split()
    assert len(ports) == 2 and len(set(ports)) == 2
    assert {line.split()[-1] for line in envlog.splitlines() if line.split()[-1] != "none"} == {ports[0]}
```

- [ ] **Step 2: Run, expect failure** — `<PYTEST> -k distinct_engine_ports` → `ports` has only `none`.

- [ ] **Step 3: Implement** — replace the generic branch's single line `export ENGINE_PORT="$(free_tcp_port)"` with:

```bash
        export ENGINE_PORT="$(free_tcp_port)"
        # Serving-YAML replicas (spec 2026-10-16): one engine port per replica; the first is ENGINE_PORT.
        # ponytail: free ports are picked one by one (bind+close); a clash is possible but has not happened
        # with the single-port path either. Reserve them together if it ever does.
        ENGINE_PORTS="$ENGINE_PORT"
        for ((replica = 1; replica < ${REPLICAS:-1}; replica++)); do
            ENGINE_PORTS+=" $(free_tcp_port)"
        done
        export ENGINE_PORTS
```

- [ ] **Step 4: Run, expect pass** — `bash -n runners/launch_hardware-hcm.sh && <PYTEST>`

- [ ] **Step 5: Commit**

```bash
git add runners/launch_hardware-hcm.sh utils/test_serving_launcher_hcm.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): hardware-hcm launcher allocates one engine port per replica"
```

---

### Task 5: `serve_generic.sh` — start N replicas

**Files:**
- Modify: `benchmarks/single_node/agentic/serve_generic.sh` (metrics URLs in the generic-router branch; engine launch block from `SERVER_LOG="$RESULT_DIR/server.log"` to `SERVER_PID=$!`)
- Test: `utils/test_serving_router.py`

**Interfaces:**
- Consumes: env `REPLICAS`, `ENGINE_PORTS` (Tasks 2, 4), `CUDA_VISIBLE_DEVICES`.
- Produces: `server.log` + `server.<r>.log`, `server_command.txt` + `server_command.<r>.txt`; `AIPERF_SERVER_METRICS_URLS` = router + every engine.

- [ ] **Step 1: Write the failing tests** (append to `utils/test_serving_router.py`; add `import time` at the top; `STUB_LIB` and `GENERIC` exist)

```python
def run_generic_replicas(tmp_path, **extra):
    lib = tmp_path / "benchmarks"
    (lib / "single_node/agentic").mkdir(parents=True)
    (lib / "benchmark_lib.sh").write_text(STUB_LIB)
    script = lib / "single_node/agentic/serve_generic.sh"
    script.write_text(GENERIC.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {"nvidia-smi": "", "engine": f'echo "$CUDA_VISIBLE_DEVICES $*" >> {tmp_path}/engines\nsleep 30\n'}.items():
        (bin_dir / name).write_text("#!/bin/sh\n" + body)
        (bin_dir / name).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "RedHatAI/gemma-4-31B-it-FP8-block", "CONC": "8",
        "PORT": "8888", "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "vllm", "SERVING_SCENARIO": "s",
        "SERVING_LOADER": "l", "AIPERF_HF_CLI": "true", "SERVED_MODEL_NAME": "RedHatAI/gemma-4-31B-it-FP8-block",
        "SERVE_CMD": "engine --tensor-parallel-size 2", "ROUTER_CMD": "vllm-router", "ROUTER_PROFILE": "generic",
        "ENGINE_PORT": "9200", "ENGINE_PORTS": "9200 9201", "REPLICAS": "2", "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    } | extra
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)


def test_serve_generic_replicas_split_gpus_ports_and_logs(tmp_path):
    out = run_generic_replicas(tmp_path)
    assert out.returncode == 0, out.stderr
    engines = tmp_path / "engines"
    for _ in range(100):
        if engines.exists() and len(engines.read_text().splitlines()) == 2:
            break
        time.sleep(0.05)
    assert sorted(engines.read_text().splitlines()) == [
        "4,5 --tensor-parallel-size 2 --host 0.0.0.0 --port 9200",
        "6,7 --tensor-parallel-size 2 --host 0.0.0.0 --port 9201",
    ]
    calls = (tmp_path / "results/calls").read_text()
    assert f"WAIT health --port 9200 --server-log {tmp_path}/results/server.log" in calls
    assert f"WAIT health --port 9201 --server-log {tmp_path}/results/server.1.log" in calls
    assert "WAIT frontend --port 8888 --model RedHatAI/gemma-4-31B-it-FP8-block" in calls
    assert "METRICS http://localhost:8888/metrics,http://localhost:9200/metrics,http://localhost:9201/metrics" in calls
    assert (tmp_path / "results/server_command.1.txt").read_text().strip().endswith("--port 9201")


def test_serve_generic_replicas_need_one_port_each(tmp_path):
    out = run_generic_replicas(tmp_path, ENGINE_PORTS="9200")
    assert out.returncode == 1
    assert "replicas 2 needs 2 ports in ENGINE_PORTS" in out.stderr
```

- [ ] **Step 2: Run, expect failure** — `<PYTEST> -k replicas` → one engine only / no error.

- [ ] **Step 3: Implement** in `benchmarks/single_node/agentic/serve_generic.sh`:

(a) Generic-router metrics: replace

```bash
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics,http://localhost:${ENGINE_PORT}/metrics"
```

with

```bash
        metrics_urls="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics"
        for engine_port in ${ENGINE_PORTS:-$ENGINE_PORT}; do
            metrics_urls+=",http://localhost:${engine_port}/metrics"
        done
        export AIPERF_SERVER_METRICS_URLS="$metrics_urls"
```

(b) Engine launch: wrap the existing block that starts at `SERVER_LOG="$RESULT_DIR/server.log"` and ends at `SERVER_PID=$!` so the old code becomes the `else` branch, unchanged:

```bash
SERVER_LOG="$RESULT_DIR/server.log"
eval "set -- $SERVE_CMD"
if [[ "${REPLICAS:-1}" -gt 1 ]]; then
    # Serving-YAML replicas (spec 2026-10-16): N copies of SERVE_CMD, each on its own GPU slice and
    # ENGINE_PORTS entry. Replica 0 keeps server.log / server_command.txt: the vLLM server-metrics
    # adapter reads "GPU KV cache size" from server.log (replicas are identical).
    read -r -a replica_ports <<< "${ENGINE_PORTS:-}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=, read -r -a replica_gpus <<< "$CUDA_VISIBLE_DEVICES"
    else
        mapfile -t replica_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
    if (( ${#replica_ports[@]} != REPLICAS || ${#replica_gpus[@]} == 0 || ${#replica_gpus[@]} % REPLICAS )); then
        echo "ERROR: replicas $REPLICAS needs $REPLICAS ports in ENGINE_PORTS (got '${ENGINE_PORTS:-}') and a GPU count divisible by $REPLICAS (got ${#replica_gpus[@]})" >&2
        exit 1
    fi
    gpus_per_replica=$(( ${#replica_gpus[@]} / REPLICAS ))
    replica_pids=()
    for ((replica = 0; replica < REPLICAS; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        replica_cmd=("$@" --host 0.0.0.0 --port "${replica_ports[replica]}")
        write_command "$RESULT_DIR/server_command${suffix}.txt" "${replica_cmd[@]}"
        CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${replica_gpus[*]:replica*gpus_per_replica:gpus_per_replica}")" \
            "${replica_cmd[@]}" > "$RESULT_DIR/server${suffix}.log" 2>&1 &
        replica_pids+=("$!")
    done
    for ((replica = 0; replica < REPLICAS; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        wait_for_server_ready --port "${replica_ports[replica]}" --server-log "$RESULT_DIR/server${suffix}.log" \
            --server-pid "${replica_pids[replica]}"
    done
    # ponytail: the server watch follows replica 0 only; another replica dying shows up as failed
    # requests (failed-request threshold 0.1 stops the run). Watch every PID if that proves too slow.
    SERVER_PID="${replica_pids[0]}"
else
    <the existing if/elif/else that builds SERVER_CMD, the write_command line, the launch line and SERVER_PID=$!, unchanged>
fi
```

(The `<…>` line above means: move the existing lines there verbatim; do not retype them.) The readiness block after it (`wait_for_frontend_model` / `wait_for_server_ready`) is unchanged and uses `SERVER_PID`.

- [ ] **Step 4: Run, expect pass** — `bash -n benchmarks/single_node/agentic/serve_generic.sh && <PYTEST>` (the existing `test_serve_generic_branches` cases must stay green).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_router.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): serve_generic starts N engine replicas on GPU slices"
```

---

### Task 6: Flow-test serving YAMLs (tau2 smoke) + docs

**Files:**
- Create: `serving/gemma4-livekit-1gpu-tau2smoke.yaml`, `serving/gemma4-livekit-2rep-cacheaware-tau2smoke.yaml`, `serving/gemma4-livekit-2rep-chash-tau2smoke.yaml`
- Modify: `.agents/skills/serving-bench/SKILL.md`, `docs/serving-yaml-procedures.md`, `docs/serving-yaml-procedures_zh.md`

- [ ] **Step 1: Write the three YAMLs.** `serving/gemma4-livekit-1gpu-tau2smoke.yaml`:

```yaml
# Gemma-4-31B FP8-block, prod flags (HCM gemma4 audit 2026-10-15) minus MTP, 1 GPU, no router.
# Flow test of the LiveKit 111868 sizing setup on the tau2 smoke (spec 2026-10-16).
name: gemma4-livekit-1gpu-tau2smoke
runner: hardware-hcm-8x_01
image: vllm/vllm-openai:v0.25.1
model: RedHatAI/gemma-4-31B-it-FP8-block
gpus: 1
precision: fp8
env:
  VLLM_DISABLE_COMPILE_CACHE: "1"
  NCCL_P2P_LEVEL: NVL
serve: |
  vllm serve RedHatAI/gemma-4-31B-it-FP8-block
  --served-model-name RedHatAI/gemma-4-31B-it-FP8-block
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.92
  --max-model-len 262144
  --max-num-seqs 64
  --max-num-batched-tokens 16384
  --enable-chunked-prefill
  --long-prefill-token-threshold 8192
  --kv-cache-metrics
  --enable-prompt-tokens-details
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
scenario: custom-agentic-tau2-smoke
conc: [8]
```

`serving/gemma4-livekit-2rep-cacheaware-tau2smoke.yaml` = the same file with: comment line 1 `… 2 replicas (TP1 each) behind vllm-router 0.1.14 cache_aware, as prod.`, `name: gemma4-livekit-2rep-cacheaware-tau2smoke`, `gpus: 2`, a new line `replicas: 2` after `gpus`, and before `scenario:`:

```yaml
router:
  image: vllm/vllm-openai:v0.25.1
  metrics_port: 29000
  serve: >-
    bash -c "python3 -m pip install -q vllm-router==0.1.14
    && exec vllm-router --host 0.0.0.0 --port {{router_port}}
    --worker-urls {{engine_urls}} --policy cache_aware
    --prometheus-host 127.0.0.1 --prometheus-port 29000
    --request-timeout-secs 14400 --disable-retries"
```

`serving/gemma4-livekit-2rep-chash-tau2smoke.yaml` = the cache_aware file with `name: gemma4-livekit-2rep-chash-tau2smoke`, comment `… consistent_hash (sticky per session: aiperf's X-Correlation-ID)`, and `--policy consistent_hash`.

Write all three files out in full (no anchors/includes; YAML is the run record).

- [ ] **Step 2: Validate** — for each: `python -P -m infx.serving.to_matrix --check serving/<file>` prints a title; `python -P -m infx.serving.to_matrix serving/gemma4-livekit-2rep-chash-tau2smoke.yaml | python3 -c "import json,sys; s=json.load(sys.stdin)['serving']; print(s['replicas'], s['router-cmd'])"` → `2 bash -c '…--worker-urls {{engine_urls}} --policy consistent_hash…'`. Then `<PYTEST>` (`test_every_checked_in_serving_yaml_builds` picks them up).

- [ ] **Step 3: Docs.**
  - `.agents/skills/serving-bench/SKILL.md`, in the YAML example after `gpus: 1`, add the line `replicas: 1                         # optional: N engine copies, gpus/N GPUs each; N > 1 needs router: with {{engine_urls}}`; and under "Router" add a bullet: `**Replicas** (several copies of one engine, e.g. 2× TP1): set replicas: N and gpus = N × per-engine GPUs; the router command lists the engines as {{engine_urls}} (space-separated URLs). Not with the Dynamo frontend.`
  - `docs/serving-yaml-procedures.md`: a short "Replicas" subsection with the same two sentences and the 2-replica router block from Step 1. `docs/serving-yaml-procedures_zh.md`: the same subsection in Chinese:
    「**副本（replicas）**：同一引擎的多个副本（如 2×TP1）。设置 `replicas: N`，`gpus` = N × 每个引擎的 GPU 数；路由命令用 `{{engine_urls}}`（空格分隔的 URL）列出全部引擎。不支持 Dynamo frontend。」

- [ ] **Step 4: Commit**

```bash
git add serving/gemma4-livekit-1gpu-tau2smoke.yaml serving/gemma4-livekit-2rep-cacheaware-tau2smoke.yaml \
        serving/gemma4-livekit-2rep-chash-tau2smoke.yaml .agents/skills/serving-bench/SKILL.md \
        docs/serving-yaml-procedures.md docs/serving-yaml-procedures_zh.md
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): Gemma-4 LiveKit flow-test YAMLs (1 GPU, 2 replicas cache_aware / consistent_hash) + replicas docs"
```

---

### Task 7: Flow smokes on hardware (user-gated push)

- [ ] **Step 1: Ask the user** to push `feat/serving-replicas-livekit` to `origin` (vngcloud/InferenceX). Push only after an explicit yes: `git push -u origin feat/serving-replicas-livekit`.

- [ ] **Step 2: Pick boxes** — `ssh slurm-client squeue -p hardware-hcm`; the 2-GPU jobs fit next to another user's 2-GPU jobs. Set `runner:` to `hardware-hcm-8x_01` or `_02` per free box (one 8x runner identity = one job at a time per box). Never cancel another user's job.

- [ ] **Step 3: Dispatch the three** — `utils/serving/dispatch.sh serving/gemma4-livekit-1gpu-tau2smoke.yaml` (then the two 2rep files). Record run URLs.

- [ ] **Step 4: Check acceptance 1–2** (spec "Acceptance"), from the artifacts (`gh run download <id> --repo vngcloud/InferenceX -n <artifact>`):
  - all three green; `results/server.log` and (2rep) `results/server.1.log` both show request lines; `router.log` lists both workers; `router_command.txt` shows both URLs.
  - `server.log`/`server.1.log` start lines show one GPU each (vLLM prints the device); if both replicas landed on the same GPU, Slurm's `CUDA_VISIBLE_DEVICES` was not what Task 5 assumed: stop and report.
  - consistent_hash vs cache_aware: `grep -o "Prefix cache hit rate: [0-9.]*%" results/server*.log | tail` per run; consistent_hash should be higher at the same conc.
- [ ] **Step 5: Think-time-off check** (D5): run `gemma4-livekit-1gpu-tau2smoke` once more with a temporary scenario copy whose `traceIdleGapCapSeconds` is `0.0` (do not commit it); compare `benchmark.log` request throughput with Step 3: it must be clearly higher (no idle gaps). If not, D5 falls back to a delay-stripped dataset file (tell the user; the dataset plan adds it).
- [ ] **Step 6:** Write a handoff `InferenceOptimization/docs/handoffs/2026-10-xx-gemma4-livekit-replicas-smoke.md` (+ index row in `docs/handoffs/CLAUDE.md`) with run URLs and results.

---

### Task 8: LiveKit dataset scenarios (after the dataset track publishes)

**Files:**
- Modify: `scenarios/datasets/lock.yaml`
- Create: `scenarios/custom-agentic-livekit111868-v1.yaml`, `-v2.yaml`, `-v1-nothink.yaml`, `-v2-nothink.yaml`, `-smoke.yaml`
- Modify: `utils/serving/check_scenario_golden.py`

**Interfaces:**
- Consumes: from the dataset track (spec 0004 "Published" line): `revision` (40-hex) and the sha256 of both trace files.

- [ ] **Step 1: Lock entries** (append; fill the four values from the dataset track; no placeholder may be committed):

```yaml
livekit-111868-v1:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: <40-hex from the dataset track>
  file: trace.v1-append-ideal.jsonl
  sha256: <64-hex>
livekit-111868-v2:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: <same 40-hex>
  file: trace.v2-observed.jsonl
  sha256: <64-hex>
```

Check: `python utils/serving/fetch_custom_dataset.py --check livekit-111868-v2; echo $?` → 0.

- [ ] **Step 2: Scenarios.** `scenarios/custom-agentic-livekit111868-v2.yaml`:

```yaml
# custom-agentic family: LiveKit Agents voice-agent traces of customer 111868 on Gemma-4-31B
# (Green-Lab/2026-10-16-111868-agentic-traces, synthetic text with logged lengths/cache/timing;
# vmonitor-gateway-datasets spec 0004), 1217 rooms / 12902 requests, V2 = observed cache.
# No request cap: lanes recycle rooms (fresh cache-bust marker per play) for the whole duration.
# Spec: docs/superpowers/specs/2026-10-16-serving-replicas-livekit-design.md
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
  - name: livekit-111868-v2
    type: file
    format: mooncake_trace
    path: ${IX_CUSTOM_DATASET_FILE}
    entries: 1217
    randomSeed: 42
    traceIdleGapCapSeconds: 300.0
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 1200.0
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

The other four, each written out in full from this one:
- `-v1.yaml`: dataset `name: livekit-111868-v1`, header comment says `V1 = append-ideal`.
- `-v1-nothink.yaml` / `-v2-nothink.yaml`: as v1 / v2 with `traceIdleGapCapSeconds: 0.0` and a comment line `# Think time off: a room's pending timers advance as soon as it is idle (mooncake ignores ignoreTraceDelays).`
- `-smoke.yaml`: as v2 with `duration: 180.0`.

- [ ] **Step 3: Golden check.** In `utils/serving/check_scenario_golden.py`:
  - add to `SCENARIOS`: the five names with kind `"livekit"`; `ALLOWED["livekit"] = {".benchmark.datasets[0].name"}`;
  - add `LIVEKIT_IDLE_CAP = {"custom-agentic-livekit111868-v1-nothink": 0.0, "custom-agentic-livekit111868-v2-nothink": 0.0}` (default 300.0);
  - change `tau2_reference_argv` to `def tau2_reference_argv(duration, artifact_dir, dataset_file, entries=1576, request_count=16798, idle_cap=300.0)`: use `--trace-idle-gap-cap-seconds {idle_cap:g}` and append ` --request-count {request_count}` only when `request_count` is not None (tau2 callers unchanged);
  - in `main()`: `elif kind == "livekit": reference = tau2_reference_argv(duration, tmp, str(dataset_file), entries=1217, request_count=None, idle_cap=LIVEKIT_IDLE_CAP.get(scenario, 300.0))`.
  - Run GOLDEN → `OK` for all 9 scenarios + the negative case.

- [ ] **Step 4:** `<PYTEST>` (lock/scenario tests) and commit:

```bash
git add scenarios/datasets/lock.yaml scenarios/custom-agentic-livekit111868-*.yaml utils/serving/check_scenario_golden.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): LiveKit 111868 V1/V2 scenarios (think on/off, smoke) pinned to Green-Lab dataset"
```

- [ ] **Step 5: Sizing YAMLs** — create (full files, engine/router blocks exactly as Task 6):

| File | gpus / replicas | router policy | scenario | conc |
|---|---|---|---|---|
| `gemma4-livekit-1gpu-smoke.yaml` | 1 / 1 | — | `custom-agentic-livekit111868-smoke` | [8] |
| `gemma4-livekit-1gpu-v1.yaml` | 1 / 1 | — | `…-v1` | [4, 6, 8, 16, 32, 48, 64] |
| `gemma4-livekit-1gpu-v2.yaml` | 1 / 1 | — | `…-v2` | [4, 6, 8, 16, 32, 48, 64] |
| `gemma4-livekit-2rep-cacheaware-v1.yaml` | 2 / 2 | cache_aware | `…-v1` | [8, 16, 32, 64, 96, 128] |
| `gemma4-livekit-2rep-cacheaware-v2.yaml` | 2 / 2 | cache_aware | `…-v2` | [8, 16, 32, 64, 96, 128] |
| `gemma4-livekit-2rep-chash-v1.yaml` | 2 / 2 | consistent_hash | `…-v1` | [8, 16, 32, 64, 96, 128] |
| `gemma4-livekit-2rep-chash-v2.yaml` | 2 / 2 | consistent_hash | `…-v2` | [8, 16, 32, 64, 96, 128] |

Validate each with `to_matrix --check`, `<PYTEST>`, commit `feat(serving): Gemma-4 LiveKit 111868 sizing YAMLs`. Push (user already approved the branch in Task 7; still say what is pushed).

- [ ] **Step 6: Acceptance 3** — dispatch `gemma4-livekit-1gpu-smoke.yaml`: green; `benchmark.log` shows `unique conversations: 1217` (or the aiperf line with 1217 sessions); no request-count stop before 180 s.

---

### Task 9: Calibration and ladders (dispatch + report)

- [ ] **Step 1: Calibration (acceptance 4)** — dispatch `gemma4-livekit-1gpu-v2.yaml` and `-v1.yaml` (all conc; c6 is the POC peak). From c6: `grep -o "Prefix cache hit rate: [0-9.]*%" results/server.log | tail -3`. Expected V2 near the prod 77.5%, V1 between 77.5% and 86.4%. A large miss → stop and report (the dataset plan is wrong, not the server).
- [ ] **Step 2: Ladders (acceptance 5)** — dispatch the four 2rep YAMLs (two boxes, one job per box at a time).
- [ ] **Step 3: Report** — per (shape, policy, version): TTFT p95 per conc from `bmk_agentic_*` JSON; max conc with TTFT p95 ≤ 0.5 s per GPU; GPUs for 300 rooms = ⌈300 / (max conc per GPU)⌉ for V1 and V2; routing factor = 2rep max conc / (2 × 1-GPU max conc) per policy. Write it as a handoff (conclusion first, measured vs inferred marked, per `InferenceOptimization/AGENTS.md` report rules; run the `humanizer` skill on it) and update the handoff index.
