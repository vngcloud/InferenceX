# Serving-YAML dispatch V3: generic router sidecar (design)

**Date:** 2026-10-14 · **Branch:** `vng-benchmark` · **Author:** Ly Quang Thang
**Builds on:** `2026-10-13-serving-router-dynamo-design.md` (V2). Everything V2 says holds unless this document overrides it. V2 decision 2 ("Dynamo only") is replaced by this spec.

## Goal

`router:` accepts any router: an `image`, a launch command and env, deployed next to the engine on the runner. The benchmark points aiperf at the router. Adding router N needs a YAML, no code.

Success:
1. The GLM-5.3 W4AFP8 paste (SGLang 0.5.18 engine + sglang-router, `runner: hardware-hcm-8x_*`, `conc: [8, 24, 32, 48]`, `scenario: agentx-cc-062126`) validates and dispatches from one YAML, with no sglang-router-specific code.
2. The three existing Dynamo YAMLs (`serving/glm53-w4afp8-dynamo*.yaml`) render a byte-identical matrix object and need no edit.

## Why V2 is not general

Dynamo is hard-coded in four places:
1. `infx/serving/render.py` accepts only `python3 -m dynamo.frontend`, and requires the engine to be `python3 -m dynamo.sglang`.
2. `runners/launch_hardware-hcm.sh` always injects `DYN_DISCOVERY_BACKEND`, `DYN_FILE_KV`, `DYN_SYSTEM_PORT`.
3. `benchmarks/single_node/agentic/serve_generic.sh` hard-codes two metrics URLs of the Dynamo shape.
4. Readiness is tied to the Dynamo frontend.

An HTTP engine (`sglang.launch_server`, `vllm serve`) behind a router has its own port, and the router needs its address. V2 gives the engine `$PORT` and gives the router nothing to learn it from.

## Decisions (agreed in brainstorming)

1. **Placeholders in the YAML** wire router to engine. Not env vars, not a router `kind` adapter.
2. **Dynamo becomes a profile**, auto-detected, not the only path. Existing Dynamo YAMLs keep working unedited.
3. **One engine behind one router.** Multi-worker and PD-disagg are out of scope.
4. **The router still needs a `hardware-hcm-*` runner**, because only `launch_hardware-hcm.sh` starts the sidecar.
5. **Image and router-command inference live in the skill**, not the validator (see §5).

## §1 Schema (`infx/serving/schema.py`)

```yaml
router:
  image: <any valid image ref>        # required, IMAGE_PATTERN
  env: {}                             # optional, same rules as top-level env
  metrics_port: 18888                 # optional; default: the router port
  serve: |                            # required, any command
    python3 -m sglang_router.launch_router
    --host=0.0.0.0 --port={{router_port}} --prometheus-port=18888
    --worker-urls={{engine_url}} --dp-aware --policy=cache_aware
```

- Allowed placeholders in `router.serve` and `router.env` values: `{{router_port}}` (the port aiperf calls, `$PORT`), `{{engine_url}}` (`http://127.0.0.1:<ENGINE_PORT>`), `{{engine_port}}`. Any other `{{...}}` that is not an inline Jinja template (`{%` / `{{` handled by the V2 `_check_json_values` rule) fails: `router.serve: unknown placeholder {{x}}; allowed: router_port, engine_url, engine_port`.
- `metrics_port`: integer 1-65535. It names a literal port the router serves `/metrics` on.
- The V2 check "only `dynamo.frontend` is supported" and "with a router block the engine must be `dynamo.sglang`" are removed.
- Kept: no `$VAR`, no secrets, no reserved env keys, `router:` requires `hardware-hcm-*`, `--numa-node` rejection, JSON-flag check.
- `RESERVED_ENV` unchanged from V2, plus `ENGINE_PORT`.

## §2 Render and engine port (`infx/serving/render.py`, `serve_generic.sh`)

- Placeholders are **not** resolved in `render`; ports are allocated at runtime. `router-cmd` / `router-env` keep the `{{...}}` text.
- Engine with a router:
  - HTTP engine (`vllm serve`, `sglang.launch_server`, `sglang serve`): the recipe appends `--port $ENGINE_PORT` (plus `--host 0.0.0.0`), not `--port $PORT`. Without a router, unchanged.
  - `dynamo.*` engine: unchanged (no HTTP port).
- New matrix field `router-metrics-port` (`''` when unset). Existing fields keep their names.

## §3 Launch (`launch_hardware-hcm.sh`, `serve_router.sh`)

- The launcher allocates `ENGINE_PORT` (same helper as the other ports) and exports it to the engine step and the router step.
- `serve_router.sh` substitutes `{{router_port}}`, `{{engine_url}}`, `{{engine_port}}` in `ROUTER_CMD` and in each `ROUTER_ENV` value with plain bash string replacement before `eval`. It never evaluates placeholder text.
- **Dynamo profile.** Applies when the engine is `python3 -m dynamo.*` **or** the router argv starts with `python3 -m dynamo.frontend`:
  - the launcher injects `DYN_DISCOVERY_BACKEND=file`, `DYN_FILE_KV`, `DYN_SYSTEM_ENABLED`, `DYN_SYSTEM_PORT` exactly as today;
  - `serve_router.sh` appends `--http-host 0.0.0.0 --http-port $PORT` when the command carries no `{{router_port}}` (current behavior, so old YAMLs need no edit).
- Any other router: no `DYN_*`, nothing appended. The command is the user's.
- **Metrics** (`AIPERF_SERVER_METRICS_URLS`, built in `serve_generic.sh`):
  - router: `http://localhost:<metrics_port or PORT>/metrics`; with `metrics_port` equal to `PORT` or unset, one URL;
  - engine: Dynamo profile `DYN_SYSTEM_PORT/metrics`; HTTP engine `ENGINE_PORT/metrics`.
- **Readiness**: wait for the engine to answer (`/health` on `ENGINE_PORT`, HTTP engine) or the Dynamo condition (as today), then wait for `http://localhost:$PORT/v1/models` to list the served model. The existing router-exited and server-exited guards stay.
- Logs: `router.log`, `server.log`, `router_command.txt` as in V2; `router_command.txt` holds the command after substitution.

## §4 Tests

- `utils/test_serving_*.py`:
  - schema: unknown placeholder fails; `metrics_port` range; non-Dynamo router passes; Dynamo YAMLs still pass.
  - render: placeholders preserved in `router-cmd`; HTTP engine + router sets the engine-port path; `router-metrics-port` filled / empty.
  - **golden**: matrix output of the three existing Dynamo YAMLs is identical before and after (diff the JSON in the test).
  - fixture: `serving/` or test-data YAML built from the GLM-5.3 sglang-router paste validates (engine `lmsysorg/sglang:v0.5.18`, router image `sglang-router-patched:v0.5.18-slim-noargcanon`, `runner: hardware-hcm-8x_02`).
- Shell: a bats-style or plain-bash test of the placeholder substitution in `serve_router.sh` (plain function, sourced).
- Real verification: one dispatch of the sglang-router YAML with `agentx-cc-062126-smoke`, `conc: [8]`. Per `verification-before-completion`, no "done" claim without the run URL and the router/engine metrics visible in the artifacts.

## §5 Skill (`.agents/skills/serving-bench/SKILL.md`)

Replaces three rules:

- **Image, public:** the user gives a version and no image → resolve it, do not ask.
  - SGLang → `lmsysorg/sglang:v<ver>`; vLLM → `vllm/vllm-openai:v<ver>`.
  - Verify the tag exists with the Docker Hub API (`https://hub.docker.com/v2/repositories/<repo>/tags?name=<tag>`) before writing the YAML.
  - Variants (`-cu129`, `-cu130`, `-runtime`) exist → take the bare tag and say which tag you chose in the reply.
  - Tag missing → stop and ask.
  - Neither version nor image given → ask, as before. Never take an image from `serving/*.yaml`, the YAML example or past runs.
- **Router command, inferred:** a pasted router with flags but no command → infer from the flag family and state the assumption in the reply so the user can correct it. `--worker-urls`, `--policy=cache_aware`, `--dp-aware`, `--prometheus-port` → `python3 -m sglang_router.launch_router`. Unknown family → ask.
- **Router translation:** the router's listen host/port flags → `--host=0.0.0.0 --port={{router_port}}`. A worker URL (k8s service, hostname) → `{{engine_url}}`. A literal metrics port flag → `router.metrics_port`. Keep the user's other flags verbatim. The "only the Dynamo pair is supported, ask" rule is removed.
- Ladder and scenario from the user's message stand: `conc: [8, 24, 32, 48]`, `scenario: agentx-cc-062126` ("nghiêm túc" = the full 3600 s scenario, not `-smoke`).

## Risks

- Docker Hub pull rate limits on the HCM runner (`enroot import` of a public image). Mitigation: the squash cache (`/mnt/containers`) already caches by image; first pull is the only cost. Check on the first real run.
- The patched router image's real entrypoint is not known. The inference in §5 is a guess flagged to the user; a wrong guess fails fast in `router.log` before aiperf starts.
- `dp-aware` sglang-router queries the worker for DP info; with `--dp 4` on tp8 the engine must expose it on `ENGINE_PORT`. Verify in the first run.

## Out of scope

Multiple engines behind one router, PD-disaggregated serving, routers on non-HCM runners, per-router metrics adapters.

## Amendments (implementation plan, 2026-10-14)

- **`router-profile`** (`dynamo` | `generic` | `''`) is a second new serving-object field next to `router-metrics-port`. Profile `dynamo` = engine `python3 -m dynamo.sglang` with router `python3 -m dynamo.frontend`; the pairing is enforced (a Dynamo router needs a Dynamo engine and the reverse). Every other router is `generic`. This replaces the spec's "engine is dynamo.* **or** router is dynamo.frontend".
- **Golden:** the existing Dynamo YAMLs keep every existing field byte-identical; the two new fields are added (`utils/testdata/serving_dynamo_build_golden.json`).
- **Placeholder grammar:** exactly `{{` + `[a-z_]+` + `}}`. `{{ x }}` with spaces is inline Jinja and stays literal. `shlex.join` quotes a placeholder token, e.g. `'--port={{router_port}}'`; substitution happens before the launcher's `eval`.
- **Readiness:** generic routers use the existing `wait_for_frontend_model` (router `/v1/models` lists the served model). The separate engine `/health` wait from §3 is dropped: the router listing the model already implies a healthy engine, and the guards for engine-exited / router-exited stay.
- **Metrics:** no dedupe needed; router metrics port (or router port) and `ENGINE_PORT` never coincide.
- `metrics_port` with a Dynamo router is an error (it would be ignored).
