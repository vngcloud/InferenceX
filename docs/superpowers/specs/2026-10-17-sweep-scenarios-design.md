# sweep: synthetic ISL/OSL/cache scenarios with in-job SLO search — design

Date: 2026-10-17 · Author: Ly Quang Thang · Status: draft, awaiting review
Builds on: `docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md`, `docs/superpowers/specs/2026-10-12-custom-agentic-tau2-design.md`
Branch: `vng-benchmark`

## Goal

Benchmark a serving config on a fixed grid of synthetic workloads and report, for each grid point, the **highest
concurrency (CCU) that still meets a stated SLO**. One CI job per (serving YAML, grid point). The CCU is found by
an adaptive search inside the job, not by a fixed ladder of jobs.

Reference method: the "GLM-5.2 test example" PDF (27 points: ISL 32k/64k/128k × OSL 512/1k/2k × cache 30/60/90%,
SLO TTFT < 8s and TPOT < 50ms, coarse sweep + bisection + fine sweep on request rate). This design keeps the idea
(max load under SLO per grid point) and changes the grid, the load knob and the tool.

## Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | Grid: ISL {8k, 32k, 64k, 128k} × OSL {512, 1k} × cache {0%, 50%, 90%} = **24 scenarios** | Chosen by the user. Wider ISL low end, OSL pruned to two, cache includes the 0% worst case. |
| 2 | Tool: **AIPerf** (`utils/aiperf`, pin `faf3f8b3`) | Same tool as AgentX/tau2; one metric vocabulary. |
| 3 | Load knob: **concurrency**, not request rate | AIPerf/NIM docs: with request rate, outstanding requests grow without bound past capacity; concurrency is self-limiting. Rate and TPM are derived from the result. |
| 4 | One job per (serving YAML, scenario); search runs **inside** the job | Today one `conc` = one job = one model load (~20 min). A fixed ladder over 24 scenarios would load the model ~190 times. |
| 5 | Search: AIPerf `max-concurrency-under-sla`, style `smooth_isotonic` | Exponential bracket + PAVA/PCHIP root-find. Handles the non-monotone points visible in the PDF better than plain bisection. Reports `cliff` instead of faking a smooth boundary. No custom search code. |
| 6 | Search params: range **[1, 512]**, precision **5%**, **max 16 iterations**, 1 run per probe | Chosen by the user. Worst case about 16 × 5–6 min per job. |
| 7 | Probe size: **300 s with a floor of 100 completed requests**; first probe has 60 s warmup | Keeps p90 meaningful at 128k where few requests finish in 300 s. |
| 8 | SLO is data: list of `metric:stat:op:threshold` | User needs mixed statistics, e.g. `ttft avg < 8000` with `itl p90 < 50`. AIPerf `--search-sla` takes exactly this 4-tuple (stat in avg/p50/p90/p95/p99, op in lt/le/gt/ge, repeatable). |

## Scenario files

New family `scenarios/sweep-isl<ISL>-osl<OSL>-cache<C>.yaml` (24 files, for example `sweep-isl64k-osl1k-cache90.yaml`).
They differ only in three values; a small generator script writes them so the grid stays in one place.

- Dataset: AIPerf synthetic text, fixed seed 42.
- ISL: 8192 / 32768 / 65536 / 131072 tokens (binary k, so "128k" = 131072, matching the PDF's 131072 table).
- OSL: min = max = 512 or 1024, `ignore_eos: true` (via extra args), so every request generates the full length.
- Cache hit: `shared_system_prompt_length = round(ISL × cache%)` is the prefix shared by all requests; the rest of the
  prompt is unique per request. Cache 0% uses AIPerf cache-bust so no prefix is shared.
  (Same construction as the PDF's `--prefix-length`.)
- Endpoint: streaming chat or completions (decide in the spike; the PDF uses `/v1/completions`).
- Phase: `type: concurrency` under `sweep: adaptive_search`, recipe `max-concurrency-under-sla`.
- SLO defaults in the file: `time_to_first_token:avg:lt:8000`, `inter_token_latency:p90:lt:50`,
  plus `--error-rate-sla` below 1%.

`serving/<name>.yaml` selects a scenario by name exactly as today (`scenario: sweep-isl64k-osl1k-cache90`);
`available_scenarios()` in `infx/serving/schema.py` already lists any file in `scenarios/`.

### Metric naming caveat

AIPerf measures `inter_token_latency` (ITL), not the PDF's TPOT. The two are close but not defined identically.
The report states which one the SLO used. Numbers are comparable to the PDF in shape, not digit for digit.

## SLO override

Default SLO lives in the scenario file. A serving YAML may override it through one `env` key
(`IX_SLO`, semicolon-separated tuples, for example `time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50`).
The receiving script validates it with `check_env_vars` and rejects a malformed tuple before launching the engine
(per AGENTS.md: no silent fallback defaults for caller-supplied config).

## Search flow in the job

1. Engine and router start once (existing `serve_generic.sh` path).
2. `build_replay_cmd_yaml` gains an adaptive branch: no `--concurrency`; the scenario's `sweep` block drives AIPerf.
3. AIPerf probes CCU 1, 2, 4, 8, … until the first SLO breach, then refines inside the bracket with
   PAVA + PCHIP until `(infeasible_min − feasible_max) / infeasible_min < 5%` or 16 iterations.
4. Outputs in `aiperf_artifacts/`: `search_history.json` with `boundary_summary`
   (`feasible_max`, `infeasible_min`, `binding_constraint`, `boundary_type`, `convergence_reason`).
5. The result writer reads `boundary_summary`, then reports for the winning CCU: CCU, RPS, output tok/s, TPM
   (tok/s × 60), TTFT and ITL (avg and p90), plus the binding constraint.

Terminal states are explicit result fields, never an empty row: `no_pass_in_range` (even CCU 1 fails SLO),
`no_failure_in_range` (still passing at 512, search ceiling hit), `cliff`, `max_iterations`.

## Aggregation across the 24 points

A post-processing script (`utils/…`, name decided in the plan) collects the 24 results into one table shaped like the
PDF's summary: rows = (cache, ISL, OSL), columns = max CCU, RPS, TPM, TTFT, ITL, state. Same script emits the
markdown/CSV used for the report. This is a collector over finished runs, not part of the job.

## Verification of the cache-hit axis

Design intent is not measured hit rate. Each job records the server's prefix-cache hit counters (server metrics URL is
already passed to AIPerf) at the winning probe and reports measured hit rate next to the nominal one. A gap larger
than 10 points marks the row `cache_mismatch`.

## Known risks and spikes (resolve before the plan is final)

1. **Wrapper support.** `build_replay_cmd_yaml` passes `--config <scenario> --concurrency $CONC`. Confirm that a
   scenario with a `sweep: adaptive_search` block plus `type: concurrency` loads, and that `--search-recipe` flags work
   alongside `--config`. If not, the recipe flags are added by the wrapper instead of the YAML.
2. **Result writer.** `infx.results.agentic.process_agentic_result` assumes one concurrency per run. It needs a
   `boundary_summary` path, or a separate writer for sweep scenarios. Check what `ingest` and the dashboard do with a
   result that has no single `conc`; likely `scenario_type` stays `agentic-coding` as tau2 did, or a new type is needed.
3. **128k at CCU 1.** The PDF shows 128k/30% failing TTFT at about 0.12 RPS. With cache 0% the very first probe may
   already breach SLO; the planner must return `no_pass_in_range`, and the job must treat it as a valid result
   (non-zero exit would fail the run and hide the data).
4. **Timeouts.** Longest job estimate 1.5–2 h. Check workflow and Slurm time limits for the serving path.
5. **Synthetic prompt cost at 128k.** Tokenizing and generating 131072-token prompts has its own startup cost
   (`AIPERF_DATASET_CONFIGURATION_TIMEOUT` is already 1800 s). Measure in the smoke run.
6. **Cache 0% construction.** Confirm AIPerf cache-bust really removes prefix sharing for synthetic prompts, by
   checking the server's hit counter on a smoke run (see verification above).

## Rollout

1. Generator script and 24 scenario files.
2. Smoke: one scenario (`sweep-isl8k-osl512-cache50`) end to end on a 1-GPU runner with a small model, to settle
   spikes 1, 2, 5, 6.
3. One 128k scenario to settle spikes 3, 4.
4. Dispatch all 24 for one real config (GLM-5.3 W4AFP8 via the sglang-router serving YAML).
5. `perf-changelog.yaml` entry (append-only) and `docs/serving-yaml-procedures.md` scenario list update, with the
   `_zh.md` counterpart.

## Out of scope

- Request-rate (open-loop) mode and comparison digit-for-digit with the PDF.
- Multi-objective Pareto search and Bayesian styles (`bo`, `optuna`); only `smooth_isotonic`.
- Automatic tuning of engine flags; the search varies load only.
- A new dashboard `scenario_type` unless spike 2 shows it is required.

## Testing

Per `AGENTS.md` test quality rules, tests run real code on controlled inputs:

- Generator: given (ISL, OSL, cache) produce a file whose parsed values equal hand-computed numbers
  (for example ISL 65536 at 90% gives prefix 58982). No assertions on file text.
- SLO parser: valid tuples round-trip to the expected structure; malformed stat, op, threshold and empty list are
  rejected with a clear error.
- Result writer: feed a canned `search_history.json` per terminal state (`smooth`, `cliff`, `no_pass_in_range`,
  `no_failure_in_range`) and assert the written result fields.
- Aggregator: feed several canned results and assert the table rows and ordering.
- Search behavior itself is AIPerf's; it is exercised by the smoke runs, not re-tested here.
