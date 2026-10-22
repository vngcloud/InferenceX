# custom-agentic: tau2 scenario on the serving-YAML stack — design

Date: 2026-10-12 · Author: Ly Quang Thang · Status: approved design
Builds on: `docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md` (V1, branch `feat/serving-config-dispatch`)
Branch: `feat/custom-agentic-tau2` (worktree `../InferenceX-tau2`), on top of V1's final tip `05f47bad1`; fast-forwarded into `feat/serving-config-dispatch` when done

## Goal

An agent dispatches a tau2 (customer-service agent traces) benchmark exactly like a Weka one:
`scenario: custom-agentic-tau2` (or `-smoke`) in a `serving/<name>.yaml`, then `utils/serving/dispatch.sh`.
The run is reproducible: pinned seed, pinned dataset revision, verified file hash.

"custom-agentic" is a **scenario family** (file-backed agentic traces), not a new `scenario-type`.
Results keep `scenario_type: agentic-coding`; matrix, results, klaud and ingest are untouched.
A real scenario-type is a later step if the dashboard needs it.

## Facts this design relies on (verified 2026-10-12)

- Local data: `InferenceOptimization/vmonitor-gateway-datasets/out/agent_traces_v2/trace.tau2.jsonl`
  (1,126,364,854 B, md5 `1d6073c485afec5e53495043101852c5`, 16,798 lines, 1,576 sessions;
  identical copy at `InferenceOptimization/docs/aiperf-docs/data/trace.tau2.jsonl`).
  Format: aiperf `mooncake_trace` messages mode (`session_id`, `messages`, `output_length`, optional `tools`, `delay` ms).
  Source: tau2 slice (airline/retail/telecom) of `Exgentic/agent-llm-traces-v2`, not internal gateway traffic.
- HF today: private `thangquang09/customer-service-agent-traces`, snapshot `8deff81a5135f7af8ebea721e4cc8ef30e218fa4`
  (every past tau2 run resolved this snapshot). Token `thangquang09` is admin of org `Green-Lab`.
- Reference command (run `37920508184`, gemma4tau2 vLLM MTP, c16):
  ```
  aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:8888 --endpoint /v1/chat/completions --endpoint-type chat --streaming --model google/gemma-4-31b-it --tokenizer google/gemma-4-31b-it --concurrency 16 --benchmark-duration 1200 --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --gpu-telemetry http://localhost:9400/metrics --tokenizer-trust-remote-code --num-dataset-entries 1576 --slice-duration 1.0 --server-metrics http://localhost:8888/metrics --output-artifact-dir /workspace/results/aiperf_artifacts --unsafe-override --custom-dataset-type mooncake_trace --input-file <snapshot>/trace.tau2.jsonl --request-count 16798
  ```
  Note: no `--max-context-length` (unlike Weka).
- aiperf (`utils/aiperf` @ `faf3f8b3`): YAML supports `${VAR}` / `${VAR:default}` substitution
  (`src/aiperf/config/loader/__init__.py:24`); `--input-file`→`datasets[].path`, `--custom-dataset-type`→`format`
  (`config/flags/_converter_dataset.py`); no HF-revision field in dataset config; scenario
  `inferencex-agentx-mvp` accepts `mooncake_trace` only because of fork commit `faf3f8b3`.
- Submodule fetch: `faf3f8b3` is fetchable from the upstream URL in `.gitmodules`
  (GitHub fork network). Risk only if the fork commit is GC'd. Not changed here.
- Workflow `env:` overrides runner `.env`; `benchmark-tmpl` sets `HF_TOKEN` from a repo secret.
  So a runner-local token must use another name.

## Decisions

| # | Decision |
|---|---|
| D1 | Scenario family, not scenario-type (option B). |
| D2 | Durations: full **1200 s** (comparable with past tau2 runs), smoke **180 s**. |
| D3 | Sampling: server/model default (no `temperature` etc. in YAML), same as past runs. |
| D4 | Dataset published **public** at `Green-Lab/customer-service-agent-traces`, content = snapshot `8deff81a` verbatim; README provenance corrected. |
| D5 | Dataset pinned by HF commit sha **and** sha256 of the file, in a lock file. Mismatch = hard fail. |
| D6 | `GREENLAB_HF_TOKEN` is wired end-to-end **in this phase** (runner `.env` → container → `fetch_custom_dataset`), so a private dataset later is only a lock entry. Value for now: personal token of HF user `thangquang09` (admin of Green-Lab); later swapped for a shared Green-Lab read token by editing `.env` only. Never named `HF_TOKEN` (the workflow overrides that). |

## Components

### 1. Dataset publication (one-off, user-gated: outward-facing)

1. Download `thangquang09/customer-service-agent-traces` @ `8deff81a` (token from `~/.cache/huggingface/token`).
2. Verify `trace.tau2.jsonl` md5 == `1d6073c485afec5e53495043101852c5`; stop on mismatch.
3. Fix `README.md`: source = tau2 slice of `Exgentic/agent-llm-traces-v2` (LiteLLM-proxied agent runs),
   converted to aiperf `mooncake_trace`; license = upstream declares none (v1 was CDLA-Permissive-2.0);
   remove any "production / internal gateway" wording. Keep `plots/distributions.png`.
4. `create_repo("Green-Lab/customer-service-agent-traces", repo_type="dataset", private=False)` + `upload_folder`.
5. Record the resulting commit sha and the file sha256 for the lock file.

The script lives at `utils/serving/publish_tau2_dataset.py` (reproducible, takes `--dry-run`); the push itself
needs the user's OK at execution time.

### 2. Lock file `scenarios/datasets/lock.yaml`

```yaml
# Custom (file-backed) datasets referenced by scenarios/*.yaml via datasets[0].name.
# Each entry pins one file at one HF commit; fetch_custom_dataset verifies sha256.
tau2-cs:
  repo: Green-Lab/customer-service-agent-traces
  revision: <40-hex commit sha from step 1.5>
  file: trace.tau2.jsonl
  sha256: <64-hex>
```

The lock lives in a subdirectory so V1's non-recursive `scenarios/*.yaml` glob (`schema.available_scenarios`, the
`scenarios/<x>.yaml` existence check) never treats it as a scenario.

### 3. Scenarios

`scenarios/custom-agentic-tau2.yaml` (smoke: `custom-agentic-tau2-smoke.yaml`, identical except `duration: 180.0`):

```yaml
schemaVersion: "2.0"
benchmark:
  scenario: inferencex-agentx-mvp
  unsafeOverride: true
  endpoint: {type: chat, path: /v1/chat/completions, streaming: true, useServerTokenCount: true}
  datasets:
  - name: tau2-cs
    type: file
    format: mooncake_trace
    path: ${IX_CUSTOM_DATASET_FILE}
    entries: 1576
    randomSeed: 42
    traceIdleGapCapSeconds: 300.0
  phases:
  - {name: profiling, kind: profiling, type: concurrency, duration: 1200.0, requests: 16798,
     gracePeriod: 30.0, failedRequestThreshold: 0.1, trajectoryStartMinRatio: 0.25,
     trajectoryStartMaxRatio: 0.75, warmupRequestsPerLane: 10, agenticWarmupGracePeriod: 1800.0}
  artifacts: {sliceDuration: 1.0}
  tokenizer: {trustRemoteCode: true}
  runtime: {statsInterval: 30.0}
randomSeed: 42
```

Verified by spike (2026-10-13, aiperf `faf3f8b3`, `resolve_config`): this YAML + V1's per-run overrides resolves to
the reference CLI config with exactly **one** difference, `.benchmark.datasets[0].name` (`main` vs `tau2-cs`), a label.
`${IX_CUSTOM_DATASET_FILE}` is substituted (resolved `path` equals the CLI `--input-file`); `requests`,
`traceIdleGapCapSeconds` on a `file` dataset and `gracePeriod` are the right keys.

### 4. `infx/serving/to_matrix.py`

`scenario_facts` today accepts only one `public` dataset and returns `(duration, loader)`.
Extend: also accept one `type: file` dataset whose `name` is a key of `scenarios/datasets/lock.yaml`; return
`(duration, name)`. Unknown name → `ServingError("scenario: scenarios/<s>.yaml dataset '<n>' is not in scenarios/datasets/lock.yaml")`.
The `serving` object's `loader` field carries that name unchanged; no new fields.

### 5. Recipe side (`benchmarks/serve_generic.sh` + `benchmarks/benchmark_lib.sh`)

Real V1 file: `benchmarks/single_node/agentic/serve_generic.sh` (runs under `set -x`). Today it always does
`WEKA_LOADER_OVERRIDE=$SERVING_LOADER; resolve_trace_source`; that call becomes the `else` branch below.

- New `fetch_custom_dataset <name>` in `benchmark_lib.sh`:
  - reads `$INFMAX_CONTAINER_WORKSPACE/scenarios/datasets/lock.yaml` with `$AIPERF_PYTHON`;
  - `hf_hub_download(repo, file, revision=..., repo_type="dataset", token=$GREENLAB_HF_TOKEN or None)` into the
    mounted HF cache (`HF_HUB_CACHE`, shared `$NODE_ROOT/hf` on the 1×H200 runner). The token is used whenever
    it is set (public repos too), so public and private datasets share one code path;
  - auth failure (401/403, `RepositoryNotFoundError`, `GatedRepoError`) → one line
    `dataset <name>: <repo>@<rev> not readable; GREENLAB_HF_TOKEN is <set|unset> on runner <RUNNER_NAME>` and exit 1;
  - sha256 the file; mismatch → print expected/actual and exit 1;
  - `export IX_CUSTOM_DATASET_FILE=<path>`.
  - **No token in xtrace:** the token is read only inside Python via `os.environ["GREENLAB_HF_TOKEN"]`; bash never
    expands `$GREENLAB_HF_TOKEN` (not even in a test like `[[ -n ... ]]`; use `${GREENLAB_HF_TOKEN:+set}`, whose
    trace shows only `set`). The function wraps its body in `{ set +x; } 2>/dev/null` and restores xtrace after.
- `serve_generic.sh`: if `SERVING_LOADER` is a key in `datasets.yaml` → `fetch_custom_dataset`; else the V1 path
  (`WEKA_LOADER_OVERRIDE` + `resolve_trace_source`).
- `build_replay_cmd_yaml` unchanged (aiperf substitutes `${IX_CUSTOM_DATASET_FILE}` itself).
- `serve_generic.sh` logs one non-secret line before fetching: `GREENLAB_HF_TOKEN: set|unset`
  (never the value, never a prefix).

### 5b. `GREENLAB_HF_TOKEN` plumbing (this phase)

- **Runner `.env`:** add `GREENLAB_HF_TOKEN=<token>` to `han-3:/data/thanglq5/runner-1x_01/.env` (file mode 600),
  then restart the runner daemon (V1 procedure: `scancel 2362`-style + resubmit `/shared/thanglq5/daemon-1x01.sh`).
  Token value is copied on the workstation from `~/.cache/huggingface/token` straight into the remote file
  (pipe over ssh, no echo, never into any repo file, chat or log). User-gated step.
- **Launcher:** `runners/launch_h200-greennode-slurm-1x.sh` (V1) writes every exported var except a deny-list
  (`PATH|HOME|…|SLURM_*|ENROOT_*|…`) to `.ix_env` (mode 600, deleted after `source`) and the container sources it.
  The GitHub runner exports `.env` keys into the job, so `GREENLAB_HF_TOKEN` already flows; no launcher code change.
  A test pins it: the deny-list regex must not match `GREENLAB_HF_TOKEN`. The launcher's own `set -x` never prints
  it (it is only in `export -p` output redirected to the file).
- **Workflows:** no change. `GREENLAB_HF_TOKEN` is not a workflow secret, so the runner `.env` value is not overridden.
- **Docs:** the `serving-bench` skill and the runner-env section of V1 docs list `GREENLAB_HF_TOKEN`
  (purpose, where it lives, how to rotate: edit `.env` + restart daemon).

### 6. Tests

- Golden: extend `utils/serving/check_scenario_golden.py` (V1) with both tau2 scenarios against the reference
  command above (same model/URL/telemetry normalization as the Weka reference, duration = scenario's), with
  `IX_CUSTOM_DATASET_FILE` set to a temp file. The only allowed difference is `.benchmark.datasets[0].name`
  (explicit allow-list in the script, file datasets only). A negative case (entries 1575) must be caught.
- `utils/test_serving_to_matrix.py`: file dataset → `(1200, "tau2-cs")`; unknown name → error text above;
  `available_scenarios()` does not list anything under `scenarios/datasets/`.
- `utils/test_benchmark_lib.py` (or new `utils/test_fetch_custom_dataset.py`): stub `hf_hub_download` via a fake
  `$AIPERF_PYTHON` module path; sha match exports the var; mismatch exits 1 with both hashes.
- Lock sanity test: every scenario's file dataset name exists in the lock; revision is 40-hex; sha256 is 64-hex.
- Token tests (stubbed `hf_hub_download`): token set → passed as `token=`; unset → `token=None`; stub raising an
  auth error → the one-line message with `set|unset`, exit 1; the token value never appears in stdout/stderr.
- Launcher test (style of `utils/test_serving_launcher.py`): the `.ix_env` deny-list regex does not match
  `GREENLAB_HF_TOKEN`.
- xtrace test: run `fetch_custom_dataset` under `set -x` with `GREENLAB_HF_TOKEN=hf_dummySECRET` and a stub;
  `hf_dummySECRET` appears nowhere in captured stdout+stderr.

### 7. Skill

`serving-bench` skill (V1) gets one line: scenarios starting with `custom-agentic-` are file-backed agentic
traces; `-smoke` = 180 s. Listing `scenarios/` already surfaces them.

## Acceptance

1. `Green-Lab/customer-service-agent-traces` is public; its pinned file sha256 equals the lock.
2. Golden check OK for all four scenarios.
3. `dispatch.sh serving/qwen38-bf16-tp1.yaml` with `scenario: custom-agentic-tau2-smoke`, `conc: 8`, runner
   `h200-greennode-slurm-1x_01`: GitHub run succeeds; aiperf log shows `unique conversations: 1576` and the
   `--config` path; second run hits the HF cache (no re-download).
4. Same run log shows `GREENLAB_HF_TOKEN: set`, and a one-off container check on runner `_01`
   (`hf_hub_download` of `README.md` from private `Green-Lab/speed-bench-prepared` using only
   `GREENLAB_HF_TOKEN`) succeeds: proves the private path works before any private dataset exists.

## Out of scope

- New `scenario-type` / dashboard / ingest changes.
- Pinning sampling params.
- Moving `.gitmodules` to the fork URL.
- Other custom datasets (the lock format allows them later).
- Creating the shared Green-Lab read-only token (later swap: `.env` edit only).
- `GREENLAB_HF_TOKEN` on runners other than `h200-greennode-slurm-1x_01`.

## Amendments from the V1 review (2026-10-13, opus review of `172509b55`; V1 then fixed part of it up to `3ba14d43e`)

This branch is rebased onto V1 `3ba14d43e`. Already fixed in V1 (do not redo): runner allow-list
(`SERVING_RUNNER_POOL`), reserved `env:` denylist, `SERVE_ENV`/`SERVING_YAML` handled with xtrace off,
`dispatch.sh` 60 s clock skew, `agentx-fast`+`serving` rejected, runner label verified present on `_01`.

Owned by this branch (V1 session will not touch them):
- A1 `serve_generic.sh`: call `install_agentic_deps` **before** the dataset branch (`fetch_custom_dataset` needs
  `$AIPERF_PYTHON`; today `resolve_trace_source` triggers it implicitly via `ensure_hf_cli`). Write
  `RESULT_DIR/dataset.txt` = `<name> <repo>@<revision> sha256=<...> path=<...>` for provenance (file datasets).
- A4 `serve_generic.sh` writes `serving.yaml` with `printf '%s'` (no extra trailing newline).
- A5 `dispatch.sh`: the run-not-found error adds "do not redispatch; check `gh run list`".
- A7 `scenario_facts`: malformed scenario -> `ServingError` one-liner, never a traceback.
- A8 Ship `benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv` (copy of the qwen38 recipe's CSV) so the
  DCGM fieldset matches baseline run 37733664587.
- A9 Test command: run `utils/test_serving_*.py` (+ new tau2 tests) explicitly; whole-suite collection breaks once
  `utils/aiperf` is initialised.
- Infra steps (node `.env`, daemon restart, push, smoke runs) are done by a person with node access from the
  handoff checklist; the code work is complete without them.
- Not taken: secret-like `env:` key rejection (an engine may legitimately need a token; values are kept out of
  xtrace by V1), per-run `--max-context-length`, `model`/`serve` consistency check.

## Amendments at implementation end (2026-10-13)

- Published: `Green-Lab/customer-service-agent-traces` (public, ungated) at revision
  `2c793380d8ba45bfde79f9a5c7b5991ed40ab2ea`, `trace.tau2.jsonl` sha256
  `598ab95c793326182175d56599c743dc84d8c6d46623359327813099df05c47e` (1,126,364,854 B, byte-identical to
  `thangquang09/customer-service-agent-traces@8deff81a`). Lock entry `tau2-cs` in `scenarios/datasets/lock.yaml`.
- D6 runner token: a fine-grained **read-only** HF token with access to the Green-Lab repos, not the workstation's
  personal admin token (handoff step 1). Local copy `~/.secrets/greenlab_hf_read_token`.
- §6 unset token: `fetch_custom_dataset` passes `token=False` (truly anonymous), not `None`, so it can never pick up
  the workflow's `HF_TOKEN`.
- A1 `dataset.txt` also carries `file=<file>`.
- §6 lock sanity: AGENTS.md forbids tests that read checked-in config, so the checks live in code instead:
  `to_matrix.load_lock` rejects an entry without exactly `repo, revision, file, sha256`, a non-40-hex revision or a
  non-64-hex sha256 (one `ServingError` line); tested with fake locks. `available_scenarios` is tested on a temp repo
  with `scenarios/datasets/lock.yaml`.
- `fetch_custom_dataset.py --check` exits 0 (locked), 1 (not locked), 2 (lock unreadable); `serve_generic.sh` stops
  on 2 instead of falling into the Weka loader path.
