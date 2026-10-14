# Serving-YAML dispatch (V1) — design

**Date:** 2026-10-12 · **Branch:** `feat/serving-config-dispatch` (from `vng-benchmark` @ `6c54ee117`) · **Author:** Ly Quang Thang

## Goal

An agent turns a user's serving config (a launch command, a docker-compose file, or a k8s manifest) into one small YAML file and dispatches an AgentX benchmark with one command. Today an arm needs a `nvidia-master.yaml` key, a hand-written recipe whose filename must match a launcher formula, a `perf-changelog.yaml` entry, a hidden ref push, and a `gh workflow run` with a 40-char SHA and a `generate-cli-command` string. Each of those is output the agent can get wrong.

V1 success = a stateless agent (GLM via `use-greennode-glm`), given only the new skill and a prompt containing the qwen38 serve command, writes the serving YAML, dispatches, and the run finishes **success** on `h200-greennode-slurm-1x_01`.

## Scope

In V1:
- Engine only (one server process). No router.
- Agentic (AgentX) scenarios only.
- Runner `h200-greennode-slurm-1x_01` (node `hoanq3-h200-1x-han-3`, 1×H200, launcher `runners/launch_h200-greennode-slurm-1x.sh`).
- Dispatch through GitHub Actions (`e2e-tests.yml`).
- No ingest. Results stay in run artifacts.

Out of V1, planned next:
- Router as a second container (`router:` block). The schema reserves the key; V1 rejects it with a clear error.
- `hardware-hcm` 8×H200 with the GLM-5.3 W4AFP8 config.
- Dashboard importers. V1 only guarantees stable artifact names and the serving YAML inside the artifacts.

## Overview

```
agent writes serving/<name>.yaml
  → utils/serving/dispatch.sh serving/<name>.yaml [--ref <branch>]
      (validates locally, then: gh workflow run e2e-tests.yml --ref <branch> -f serving="<yaml>" -f test-name=...)
  → e2e-tests.yml get-jobs: input `serving` set → utils/serving/to_matrix.py → agentic-config matrix
  → benchmark-tmpl.yml (unchanged flow: priority queue, runner, artifacts)
  → runners/launch_h200-greennode-slurm.sh → GPU_COUNT=1 → launch_h200-greennode-slurm-1x.sh
      SERVE_CMD set → BENCH_SCRIPT=benchmarks/single_node/agentic/serve_generic.sh
  → serve_generic.sh: start server → wait healthy → aiperf profile --config scenarios/<scenario>.yaml + per-run overrides
```

Why `e2e-tests.yml` and not a new workflow: GitHub only registers `workflow_dispatch` workflows whose file exists on the default branch (`main`); the registered list matches `origin/main/.github/workflows` exactly. A new `serve-bench.yml` on a feature branch would 404. `e2e-tests.yml` is registered, and `gh workflow run e2e-tests.yml --ref <branch>` runs that branch's version of the file, so the feature branch is testable without touching `main` or `vng-benchmark`.

Why not call `benchmark-tmpl.yml` directly from a thin workflow: since the 2026-09-16 upstream sync it takes `config` (JSON), `priority` and `queue-token`; `remote-bench-e2e.yml` still passes the old per-field inputs and has drifted. Producing the same matrix entries the generator produces keeps every downstream step unchanged.

## §1 Serving YAML

Path: `serving/<name>.yaml`. Committing it is optional; a file outside the repo works, because the content travels as a workflow input.

```yaml
name: qwen38-bf16-tp1               # required. [a-z0-9.-]+, no "_" (it becomes model-prefix and the exp-name head)
runner: h200-greennode-slurm-1x_01  # required. A runner name; it must exist in configs/runners.yaml
image: vllm/vllm-openai:v0.29.0     # required. If the user did not give one, the agent asks
model: Qwen/Qwen3.8-27B             # required. HF repo id; never a local path
gpus: 1                             # required. ≤ gpus-per-node of the runner's pool
precision: bf16                     # optional, default bf16. Used for naming only
env:                                # optional. Exported before the serve command
  VLLM_USE_V1: "1"
serve: |                            # required. The launch command as the user gave it
  vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --kv-cache-dtype fp8 --max-model-len 262144
  --enable-prefix-caching --tool-call-parser qwen3_xml --reasoning-parser qwen3
scenario: agentx-cc-062126-smoke    # required. Name of scenarios/<scenario>.yaml; it also fixes the duration
conc: [4, 8]                        # required. One matrix job (own server, own allocation) per value
```

Validation in `utils/serving/schema.py` (pydantic). Each failure prints one line naming the field and the fix:
- unknown keys are errors (catches typos); `router:` → "router is not supported in V1"
- `runner` is listed in `configs/runners.yaml`; `gpus` ≤ that pool's `gpus-per-node`
- `scenarios/<scenario>.yaml` exists
- `model` does not start with `/`
- `framework` is inferred from `serve`: starts with `vllm serve` → `vllm`; contains `sglang.launch_server` or starts with `sglang serve` → `sglang`; anything else is an error
- `serve` must not contain a local model path (`/models/`, `/mnt/`, `/data/`). The agent rewrites those to the HF id

## §2 Scenarios: native aiperf YAML

Path: `scenarios/<name>.yaml`, aiperf `schemaVersion: "2.0"`, holding every flag that defines the workload. Reference command (qwen38 smoke run `37733664587`, the line `build_replay_cmd` produced):

```
aiperf profile --scenario inferencex-agentx-mvp
  --url http://localhost:8888 --endpoint /v1/chat/completions --endpoint-type chat --streaming
  --model Qwen/Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B --tokenizer-trust-remote-code
  --concurrency 8 --benchmark-duration 90
  --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393
  --max-context-length 262144 --random-seed 42 --stats-interval 30 --slice-duration 1.0
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 --trace-idle-gap-cap-seconds 300
  --failed-request-threshold 0.10 --use-server-token-count --unsafe-override
  --gpu-telemetry http://localhost:9400/metrics <csv>
  --server-metrics http://localhost:8888/metrics
  --output-artifact-dir /workspace/results/aiperf_artifacts
```

Split:
- **Scenario file:** `scenario`, `benchmark-duration`, endpoint path/type/streaming, dataset (`public-dataset`, `num-dataset-entries`), `max-context-length`, `random-seed`, `stats-interval`, `slice-duration`, trajectory ratios, warmup settings, `trace-idle-gap-cap-seconds`, `failed-request-threshold`, `use-server-token-count`, `unsafe-override`.
- **Per-run CLI overrides (set by the recipe):** `--url`, `--model`, `--tokenizer`, `--concurrency` (from the serving YAML's `conc`), `--server-metrics`, `--gpu-telemetry`, `--output-artifact-dir`.

The aiperf 2.0 JSON schema (`utils/aiperf/src/aiperf/config/schema/aiperf-config.schema.json`) has keys for `scenario`, `trajectoryStartMinRatio`, `warmupRequestsPerLane`, `traceIdleGapCap`, `failedRequestThreshold`, `useServerTokenCount`, `sliceDuration`, `unsafeOverride`, `randomSeed`. Where `--public-dataset`, `--num-dataset-entries` and `--warmup-grace-period` go in YAML is the first implementation task. A flag with no YAML equivalent stays a fixed CLI flag in `build_replay_cmd_yaml`, with a comment naming it.

V1 ships two scenarios, identical except for the duration:
- `scenarios/agentx-cc-062126.yaml`: `benchmark-duration` 3600 (the generator's `DEFAULT_AGENTIC_DURATION_SECONDS`), the full run.
- `scenarios/agentx-cc-062126-smoke.yaml`: `benchmark-duration` 180.

`inferencex-agentx-mvp` locks `min_benchmark_duration_seconds=900`; the smoke scenario passes only because `unsafeOverride` turns that lock into a warning and stamps `submission_valid=false`, the same way the 90 s smoke run `37733664587` did. Smoke results are for bring-up, not for comparison.

The serving YAML has no `duration` field: the scenario fixes it, and `to_matrix.py` reads it from the scenario file into the matrix entry's `duration` (which `benchmark-tmpl.yml` and the result files expect).

**Correctness gate (golden test).** aiperf writes the resolved run config to `run_config.json`. `utils/serving/test_scenario_golden.py` resolves the config, for each of the two scenarios, from (a) the reference CLI command with `--benchmark-duration` set to that scenario's value and (b) `--config scenarios/<scenario>.yaml` + the per-run overrides, same concurrency, offline with no server, and asserts equality after removing fields that legitimately differ (paths, timestamps). It must pass before any GPU run. If aiperf cannot resolve offline, the fallback is to compare the `CLI Command:` / resolved-config lines aiperf prints at startup, taken from the smoke run.

`build_replay_cmd` (CLI form) stays as it is for the existing recipes. The V1 path calls a new `build_replay_cmd_yaml` in `benchmarks/benchmark_lib.sh`, which reuses the existing helpers (`install_agentic_deps`, telemetry/metrics env, `run_agentic_replay_and_write_outputs`).

## §3 CI

`e2e-tests.yml`:
- New optional input `serving` (string, multi-line YAML). The workflow has 19 dispatch inputs; the limit is 25.
- `get-jobs`: when `serving` is non-empty, write it to a file through `env:` (never `${{ }}` interpolation inside `run:`), run `python utils/serving/to_matrix.py <file>`, and emit its output as `agentic-config`; every other output is `[]`. `generate-cli-command` and `duration-override` are ignored in this mode (the scenario owns the duration).
- `run-name` uses `test-name`, which `dispatch.sh` always sets.

`to_matrix.py` output, one entry per `conc`, with the same keys the generator emits for agentic entries (`image`, `model`, `model-prefix`, `precision`, `framework`, `runner`, `tp`, `pp`, `dcp-size`, `pcp-size`, `ep`, `dp-attn`, `spec-decoding`, `conc`, `kv-offloading: none`, `total-cpu-dram-gb: 0`, `duration`, `exp-name`, `scenario-type: agentic-coding`) plus:
- `serve-cmd`: the serve command after rewriting (below)
- `serve-env`: `KEY=VALUE` lines from `env:`
- `serving-scenario`: scenario name
- `serving-yaml`: the original YAML text, uploaded as an artifact

`tp` is read from `--tensor-parallel-size`/`--tp`/`--tp-size` (default 1). `exp-name` = `<name>_c<conc>`. `runner` is the runner name, which is also a label on that runner, so `runs-on` pins it exactly.

Serve-command rewriting (Python, unit-tested): join continuation lines; drop `--host X`, `--host=X`, `--port N`, `--port=N`; append `--host 0.0.0.0 --port $PORT`; for sglang, append `--enable-metrics` if missing. vLLM exposes metrics by default; if `--disable-log-stats` is present, validation fails.

`benchmark-tmpl.yml`: export `SERVE_CMD`, `SERVE_ENV`, `SERVING_SCENARIO` from `fromJSON(inputs.config)` when present; write `serving-yaml` to `results/serving.yaml` so the existing results artifact carries it. No other change.

No ingest: the fork's `e2e-tests.yml` has no ingest job and this path adds none. A test in `utils/serving/` asserts that no job in `e2e-tests.yml` references `ingest`.

## §4 Launcher and per-runner `.env`

The GitHub runner reads `<runner dir>/.env` (`KEY=VALUE`) at startup and puts it in every job's environment. Workflow-level `env:` overrides it, and `benchmark-tmpl.yml` already sets `HF_HUB_CACHE` and `PORT`, so runner variables use their own `IX_` prefix.

`launch_h200-greennode-slurm-1x.sh`:
- `IX_HF_CACHE_HOST` → host path mounted at `$HF_HUB_CACHE` (default: today's `$NODE_ROOT/hf`)
- `IX_SCRATCH_ROOT` → replaces `/data/inferencex` in `NODE_ROOT` (default: today's value)
- `IX_SLURM_NODELIST` → replaces the hostname map (default: the map)
- `SERVE_CMD` non-empty → `BENCH_SCRIPT=benchmarks/single_node/agentic/serve_generic.sh`; otherwise the existing filename formula
- `SERVE_CMD`, `SERVE_ENV`, `SERVING_SCENARIO` reach the container through the existing `.ix_env` file (no change needed: `export -p` already carries them)

The defaults keep existing recipes and the `_02` runner working without a `.env`.

`GREENLAB_HF_TOKEN` (added by the custom-agentic tau2 work): fine-grained read-only HF token for lock-pinned
Green-Lab datasets, read only by `utils/serving/fetch_custom_dataset.py` (never echoed; unset = anonymous). It lives
in `<runner dir>/.env` (mode 600), not in GitHub secrets, and not as `HF_TOKEN` (the workflow overrides that).
Rotate: replace the line in `.env`, then restart the runner daemon (`scancel` the daemon job, `sbatch
/shared/thanglq5/daemon-1x01.sh`).

`.env` for `h200-greennode-slurm-1x_01` (manual, one time, then restart the runner daemon):
```
IX_HF_CACHE_HOST=/data/hf-cache
IX_SCRATCH_ROOT=/data/inferencex
IX_SLURM_NODELIST=hoanq3-h200-1x-han-3
```
Shared HF cache, independent of the runner's uid: `mkdir -p /data/hf-cache && chmod 2777 /data/hf-cache && setfacl -R -d -m u::rwx,g::rwx,o::rwx /data/hf-cache`. Before that, an sbatch probe on han-3 checks that `/data` supports ACLs; if it does not, the container runs with `umask 000` instead.

## §5 `serve_generic.sh`

`benchmarks/single_node/agentic/serve_generic.sh`, about 40 lines:
1. `check_env_vars MODEL CONC DURATION PORT RESULT_DIR SERVE_CMD SERVING_SCENARIO`; export each `SERVE_ENV` line.
2. `hf download "$MODEL"` (no-op once it is in the shared cache).
3. Metrics env by framework: `AIPERF_SERVER_METRICS_URLS=http://localhost:$PORT/metrics`; `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:` or `sglang:`. GPU telemetry comes from the launcher (DCGM on 9400).
4. Write `SERVE_CMD` to `$RESULT_DIR/server_command.txt`; run it with `bash -c` in the background, log to `$RESULT_DIR/server.log`.
5. `wait_for_server_ready`, then `build_replay_cmd_yaml "$RESULT_DIR"` and `run_agentic_replay_and_write_outputs "$RESULT_DIR"`.

## §6 Skill `serving-bench` (the harness)

New `.agents/skills/serving-bench/SKILL.md` + symlink `.claude/skills/serving-bench`. Self-contained: it does not mention `inferencex-dispatch`, `nvidia-master.yaml`, recipes, hidden refs or SHAs. Content:
1. Trigger: "benchmark agentx with this config…", with an engine command, compose file or k8s manifest.
2. The serving YAML schema and one full example.
3. Translation rules: local model path → HF id (ask if unknown); drop host/port; env vars → `env:`; `gpus` from the parallel flags (tp × pp, or tp with dp-attention); compose/k8s → take `image`, `command`/`args`, `env`.
4. Ask the user when: the image is missing; the scenario is missing (`ls scenarios/`, list the names, ask); `gpus` exceeds every runner; a router is present (not supported in V1).
5. One command: `utils/serving/dispatch.sh serving/<name>.yaml`. It prints the run URL.
6. Reading results: `gh run view <id> --repo vngcloud/InferenceX`, the artifact names (`agentic_<name>_c<conc>…`, `bmk_agentic_…`, `server_logs_…`, results with `serving.yaml`), and where a failure shows (server log vs aiperf log).

`utils/serving/dispatch.sh <yaml> [--ref <branch>]`: runs `to_matrix.py --check`, sets `test-name="serving <name> <scenario> c<conc list>"`, calls `gh workflow run e2e-tests.yml --repo vngcloud/InferenceX --ref <branch, default vng-benchmark> -f serving="$(cat <yaml>)" -f test-name=...`, then prints the new run's URL.

## Testing

1. Unit (`pytest utils/serving/`): schema errors; framework inference; host/port rewrite incl. `=` forms and continuation lines; `tp` extraction; matrix shape equals a generator agentic entry (same key set); no-ingest assertion.
2. Golden: §2.
3. `bash -n` on the changed shell files; `actionlint` on the changed workflows if available.
4. Smoke (manual dispatch from the branch): qwen38, `vllm/vllm-openai:v0.29.0`, `h200-greennode-slurm-1x_01`, scenario `agentx-cc-062126-smoke` (180 s), conc 8. Pass = run success, `serving.yaml` in artifacts, server and aiperf artifacts present. Baseline for comparison: run `37733664587` (25/25 requests).
5. Acceptance: a stateless GLM agent with only `serving-bench` and a prompt holding the qwen38 serve command, without an image at first (it must ask), then with it. Pass = it writes a valid YAML, runs `dispatch.sh`, and the run succeeds.

## Risks

- The YAML key for `--public-dataset` (or another flag) may not exist → handled by the fixed-CLI-flag fallback in §2, caught by the golden test.
- `.env` changes need a runner restart; the `_01` daemon must be resubmitted (7-day sbatch) after the edit.
- A user-supplied serve command can set flags that break the run (wrong parser, OOM). That is a real benchmark failure, shown in `server.log`; V1 does not try to validate engine flags beyond §1.

## Amendments (2026-10-12, found while planning)

These override the sections above where they conflict.

1. **Rendered serving data does not go into matrix entries.** `get-jobs` validates the matrix with `infx.workflows.benchmark_schema`, whose models are `extra="forbid"`; extra keys like `serve-cmd` would fail the job. Instead `get-jobs` emits a second output `serving` (one JSON object: `serve-cmd`, `serve-env`, `served-model-name`, `scenario`, `loader`, `yaml`), `test-sweep-agentic` passes it to a new `benchmark-tmpl.yml` input `serving` (string, default `''`), and `benchmark-tmpl.yml` maps it to env `SERVE_CMD`, `SERVE_ENV`, `SERVED_MODEL_NAME`, `SERVING_SCENARIO`, `SERVING_LOADER`, `SERVING_YAML`. Matrix entries are plain generator-shaped rows and must pass `benchmark_schema.validate_matrix`.
2. **Python lives in the `infx` package**, invoked like the existing steps (`python -P -m infx.serving.to_matrix`): `infx/serving/{__init__,schema,render,to_matrix}.py`. Tests follow the repo pattern `utils/test_serving_*.py` (run from `utils/`, `pythonpath = ..`). `utils/serving/dispatch.sh` stays a shell script.
3. **Scenario YAML content is known.** A spike resolved the reference CLI with aiperf's own `resolve_config` and dumped it: `--public-dataset` → `datasets[0].type: public` + `dataset:`, `--num-dataset-entries` → `entries`, `--warmup-grace-period` → `phases[0].agenticWarmupGracePeriod`. YAML + per-run overrides resolved to a config identical to the CLI one (zero differing fields; a negative check caught changed `entries` and `concurrency`). The golden check is a script run in the aiperf project env: `utils/serving/check_scenario_golden.py`.
4. **Trace dataset pre-download.** `serve_generic.sh` sets `WEKA_LOADER_OVERRIDE=$SERVING_LOADER` and calls the existing `resolve_trace_source` (validates the loader name and downloads the HF dataset); its `TRACE_SOURCE_FLAG` is not used.
5. **Served model name.** If `serve` has `--served-model-name X`, the renderer exports `SERVED_MODEL_NAME=X`; `build_replay_cmd_yaml` passes `--model ${SERVED_MODEL_NAME:-$MODEL} --tokenizer $MODEL`, like `build_replay_cmd`.
6. **`.env` location.** Runner `_01` is sbatch daemon job 2362 (user `thanglq5`, `/shared/thanglq5/daemon-1x01.sh`, `cd /data/thanglq5/runner-1x_01 && exec ./run.sh`) on han-3. The `.env` file goes in `han-3:/data/thanglq5/runner-1x_01/.env`; applying it means `scancel 2362` and resubmitting the daemon. Submit host: `ssh slurm-client`.
7. `IX_SCRATCH_ROOT` keeps the uid suffix: `NODE_ROOT="${IX_SCRATCH_ROOT:-/data/inferencex}-$(id -u)"`.
