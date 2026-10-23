# result-report: one report format for benchmark runs — design

Date: 2026-10-23 · Author: Ly Quang Thang · Status: draft, awaiting review
Branch: `vng-benchmark`

## Goal

Turn one or more benchmark runs into a standard markdown report (optional HTML with charts), so every report the
team posts to Redmine or an MR has the same tables, the same metrics and the same units. An agent invokes the skill
with a GitHub Actions run link or a local artifacts folder. The script downloads only the bytes it needs.

Readers: the team deciding prod configs, plus Redmine/MR posts. Lead with the conclusion, then the detail tables.

## Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | Skill `result-report` at `.agents/skills/result-report/`, symlink in `.claude/skills/` | Tied to this repo's artifact layout; ships with it. |
| 2 | Script is **self-contained** under `scripts/`, stdlib only, imports nothing from `infx/` | User choice. Logic copied from the repo carries a `# source: file:line` comment for re-sync. |
| 3 | Inputs: GH run URL or ID, a downloaded artifacts folder, or a raw aiperf folder. **N inputs** in one report | 1 run is N=1. A grid sweep is one run per scenario, so N inputs can be grid pieces, A/B arms, or both. |
| 4 | Inputs are grouped by **arm** (serving YAML name + image) and **scenario**. Same arm + different scenarios = one grid; different arms + same scenario = comparison | One mechanism covers A/B and grids; no mode flag. |
| 4b | Baseline = first arm; `--baseline <arm>` overrides. Other arms show delta %, signed by metric direction | Throughput up = good, latency up = bad. |
| 5 | Download = **HTTP Range reads** inside the artifact zip (stdlib `zipfile` over a ranged file object) | Verified: artifact blob returns 206 + `Accept-Ranges: bytes`. Fetches ~50 KB from a 21–266 MB zip. Works on old runs, no workflow change. |
| 6 | Fallback: full zip via `gh run download -n` only if Range fails; ask the user when the zip is > 100 MB | Never pull 1 GB silently. |
| 7 | Cache downloaded files under `<out>/.cache/` | Re-runs and re-renders cost nothing. |
| 8 | Latency + throughput from aiperf `profile_export_aiperf.json` (ms, has p99) | Aggregate JSON has no p99 and is in seconds. |
| 9 | Fallback when no aiperf file (aggregate-only run): aggregate latency × 1000, p99 = `—`, footnote "p99 unavailable: aggregate-only". Throughput source is named in the report | e.g. run 39558375380. |
| 10 | Identity + engine cache tiers from the aggregate `bmk_agentic_<RF>/<RF>.json`; never parse the artifact name | Name is truncated to 120 bytes + hash; spec field cut, no hw. |
| 11 | Cache tier presence is decided by measured values, not identity flags | `kv_offloading`/`router` flags are often unset even when host hits are 43%. |
| 12 | Search mode: headline = `boundary_summary.feasible_max` (max CCU), binding SLA + observed breach; table of **all probes**; mark `best_trials` probe if different | Shows where and how the SLA breaks. |
| 13 | Search-mode cache tiers: parse `server_metrics_export.json` with copied sglang/vllm tier logic **only for the winning probe**, and only if its compressed zip member is ≤ 40 MB (`--max-member-mb`). Above the cap: skip, add a Note, fall back to the search aggregate's `server_metrics.cache` if present. Per-probe `Hit overall` comes from the aiperf export (client-side) | Real probes are 350-400 MB uncompressed (25-35 MB compressed) each, not ~130 KB; reading all of them downloaded 100+ MB and took minutes. |
| 14 | Router hit rate: optional. Shown when computable (search probes: `smg_cache_aware_hit` hit/(hit+miss); dynamo: aggregate `router_kv_hit_rate`), else `—` | Fixed mode would need the 1.26 GB export. |
| 15 | Never read `server_metrics_export.json` in fixed mode or for non-winner/oversized search probes, nor `profile_export.jsonl`, `inputs.json`, or timeslice files by default | These are the heavy files (member sizes come from the zip directory, no payload download). Per-request reads are deferred (no `--per-request` flag yet). |
| 16 | No server metrics at all (raw aiperf folder) → cache section shows "no server metrics logged"; client-side `overall_usage_prompt_cache_read_pct` shown as "client-side, untiered" | |
| 17 | Script writes tables deterministically; the agent writes the "Key findings" section from `data.json` | Tables identical across reports; only commentary varies. |
| 18 | Output dir default `~/bench-reports/<YYYY-MM-DD>-<slug>/` (`--out` overrides): `report.md`, `data.json`, `report.html` (with `--html`) | Outside the repo; nothing committed. |
| 19 | Report language: English | Metric names are English; posts go to MR/Redmine. |
| 20 | Synthetic grid, search mode (`sweep-isl*-osl*-cache*`): summary matrix, rows = ISL × OSL, cols = cache %, cell = `max CCU · output tok/s`; `✗` for `no_pass_in_range`, `*` for `cliff`; 2+ arms → `A / B (Δ%)` | One glance over the 24-point grid. |
| 21 | Synthetic grid, fixed CCU (`fixed-isl*-osl*-cache*`): three matrices (output tok/s, TTFT p99, ITL p99), rows = scenario, cols = CCU; per-scenario detail inside `<details>` | Several CCUs per scenario do not fit one cell. |
| 22 | Grid tables carry `cache nominal / measured overall` and flag ⚠ when they differ by > 10 points | Compare against **overall**, not GPU tier (repo `sweep.py` uses GPU only, wrong under HiCache). |
| 23 | ISL/OSL/cache parsed from the scenario name (`(sweep\|fixed)-isl(\d+k)-osl(\d+k?)-cache(\d+)`); non-matching names fall back to the raw name | Agentic scenarios (AgentX, tau2, livekit) have no grid axes. |

## Inputs and detection

| Input | Detection | What is fetched |
|---|---|---|
| `https://github.com/<o>/<r>/actions/runs/<id>[/...]` or bare `<id>` (repo defaults to `vngcloud/InferenceX`) | URL regex / digits | `gh api .../runs/<id>/artifacts` list, then per config: `bmk_agentic_<RF>` (whole, ~2 KB) and Range reads from `agentic_<RF>` |
| Downloaded folder | contains `bmk_agentic_*` / `agentic_*` dirs | local files only |
| Raw aiperf folder | `profile_export_aiperf.json` found by glob, no aggregate | local files only |

Configs are paired across artifacts by stripping the prefixes (`bmk_agentic_`, `agentic_`) to `<RF>`.
Mode is decided per config: `search_history.json` present → search, else fixed.

Files read per config (Range-fetched from `agentic_<RF>` when remote):

| File | Mode | Use |
|---|---|---|
| `<RF>.json` (aggregate) | fixed | identity, engine cache tiers, request accounting |
| `aiperf_artifacts/profile_export_aiperf.json` | fixed | latency/throughput |
| `serving.yaml`, `server_command.txt`, `router_command.txt`, `benchmark_command.txt` | both, optional | collapsed config section |
| `aiperf_artifacts/search_history.json` | search | iterations, boundary, SLA filters |
| `search_iter_NNNN/profile_runs/run_0001/profile_export_aiperf.json` | search | per-probe metrics |
| `search_iter_NNNN/profile_runs/run_0001/server_metrics_export.json` | search, winner probe only, ≤ 40 MB compressed (350-400 MB raw) | winner cache tiers, router hit |

Only `run_0001` per probe is read (the search uses one run per probe).

## Report layout (`report.md`)

```
# <title>                      (slug, date, run links)
## Key findings                (agent-written; script leaves a placeholder)
## Runs                        (table: label, run link, model, hw, framework, precision, parallelism, image, mode)
## <config/scenario section>   (one per config; rows = CCU; N runs joined on (scenario, CCU))
### Throughput & cache         (one wide table)
### TTFT (ms) / ITL (ms) / E2E latency (ms)   (three tables)
### Search                     (search mode only: headline + probe table)
<details> Config </details>    (serving.yaml + commands, when present)
## Notes                       (footnotes: fallbacks, warnings)
```

**Throughput & cache** columns: CCU · RPS · input tok/s · output tok/s · total tok/s · output tok/s/GPU ·
hit overall · hit GPU (HBM) · hit CPU (DRAM) · hit external (L3) · hit router · cached tokens by tier.
Columns that are null in every row are hidden.

**TTFT / ITL / E2E** tables: CCU · avg · p50 · p90 · p95 · p99. The E2E table adds output tok/s/user (avg) and
error rate.

**Search** headline: `max CCU = <feasible_max>` · binding SLA as text (e.g. `ITL p90 < 50 ms`, breached at
50.85 ms at c72) · state (`ok`, `cliff`, `no_pass_in_range`, …). Probe table: iter · CCU · pass/fail ·
each SLA metric observed vs threshold · output tok/s · cache hit overall; `best_trials` probe marked.

**Grid summary** (when parsed scenarios exist) goes right after `## Runs`: the search matrix (decision 20) or the
three fixed-CCU matrices (decision 21), with the nominal/measured cache column (decision 22).

**Comparison** (2+ arms): each metric cell becomes `value (Δ%)` against the baseline; rows that exist in only some
runs show `—` in the others.

`data.json` holds every extracted number (with units and source per field) so a report can be re-rendered or
re-compared without downloading again.

## HTML (`--html`)

Same tables, plus inline SVG line charts built from summary numbers only: output tok/s vs CCU, TTFT p50/p99 vs CCU,
ITL p50/p99 vs CCU (one line per run). Search mode: probe CCU vs binding metric with the threshold line.
No matplotlib, no per-request series (per-request is deferred).

## Data rules to respect

- aiperf latency is ms with `avg/p50/p90/p95/p99`; throughputs carry `avg` only. Aggregate latency is seconds,
  `mean/p50/p75/p90/p95`, no p99.
- Pick one throughput source per table (aiperf preferred); they differ slightly (78369 vs 78832 input tok/s).
- `per_gpu` divides by aggregate `num_gpus`.
- SGLang tiers: `sglang:cached_tokens{cache_source=device|host|storage_*}` over `sglang:prompt_tokens`. Exclude a
  `cache_source=total` label from the overall sum (latent double-count in the repo adapter).
- vLLM cannot split L2 from L3: show its offload hit as external and footnote "vLLM: CPU/L3 not separable".
- A missing tier is `—`, never 0 (missing host counters do not mean zero host hits).
- Do not present `prompt_by_source.computed` as a partition of prompt tokens (SGLang sums exceed the total).
- Error rate from aiperf. If the aggregate `records_error_dropped` differs a lot (e.g. 18 vs 397), add a warning
  under the E2E table.
- `qps` histogram p50 = 0 is normal; do not headline it.

## Logic copied from the repo (self-contained, with source comments)

| Copied | From | Why not import |
|---|---|---|
| sglang/vllm tier population + `sum_stat`/`sum_by_label`/`rate` helpers (~60 lines) | `infx/results/agentic/backends/{sglang,vllm,common}.py` | Decision 2; also adds the `total`-label fix |
| `summarize_search` state logic (~25 lines) | `infx/results/sweep.py:32` | Module pulls env-based `identity_fields` |
| Scenario regex + sort key | `utils/serving/collect_sweep_table.py:11` | Raises on non-matching names; we fall back to raw name |

Not reused: `process_agentic_result.main`/`build_result` (env + jsonl), `generate_aiperf_plots.py` (matplotlib +
1.2 GB file), `analyze_benchmark_distributions.py` (needs jsonl), `result_filename.py` (one-way).

## Fixed-CCU synthetic scenarios (prerequisite, in scope)

Today every `scenarios/sweep-*.yaml` has a top-level `sweep:` block, so `infx/serving/to_matrix.py:100` always
runs the adaptive search and there is no synthetic ISL/OSL/cache workload at fixed CCU. Agentic scenarios (AgentX,
tau2, livekit) already run fixed ladders; only the synthetic grid lacks it.

- `utils/serving/gen_sweep_scenarios.py` gains an option to also write `scenarios/fixed-isl*-osl*-cache*.yaml`:
  same dataset and prefix settings as the `sweep-` twin, no `sweep:` block, profiling phase 300 s. All 24 are
  generated (same code path, data only). `to_matrix.py` is unchanged: no `sweep:` block already means fixed CCU.
- Test run (feeds acceptance case 7): new serving YAML copied from `serving/gemma4-sweep-smoke-1xh200.yaml`
  (Gemma-4-31B FP8-block, SGLang v0.5.22, 1×H200) with `--enable-hierarchical-cache` added so the CPU (DRAM) tier
  has real numbers. Runner `h200-greennode-slurm-1x_02` (HAN node `hoanq3-h200-1x-han-3`, the online one).
  Scenarios `fixed-isl8k-osl1k-cache0` and `fixed-isl8k-osl1k-cache50`, `conc: [8, 32]` → 4 jobs.
  cache0 vs cache50 gives the nominal/measured check and a baseline-style comparison inside one grid.

## Skill files

```
.agents/skills/result-report/
  SKILL.md                  when to use, the one command, how to write Key findings, flags
  scripts/fetch.py          sources, local discovery, Range zip reader, remote discovery
  scripts/extract.py        point building, cache tiers, search summary, arms, build_data
  scripts/render.py         blocks, markdown, HTML, SVG charts
  scripts/result_report.py  CLI: fetch -> extract -> data.json -> report.md [-> report.html]
  scripts/test_fetch.py, test_extract.py, test_render.py, test_cli.py   stdlib unittest per module
  scripts/fixtures/         trimmed real files: aggregate, aiperf export, search_history, probe export
.claude/skills/result-report -> ../../.agents/skills/result-report
```

CLI: `result_report.py <input>... [--repo owner/name] [--baseline ARM] [--out DIR] [--title T] [--html] [--findings FILE] [--max-full-zip-mb 100]`.
Needs `gh` (authenticated) for remote inputs only. `--findings` re-renders with the agent's Key findings; `--per-request` is deferred until someone needs per-request charts.

## Testing

1. Unit tests (stdlib unittest files per module) on trimmed real fixtures: unit conversion, tier logic incl. `total` label,
   aggregate-only fallback, search summary, delta signs, null-column hiding.
2. Acceptance: one fresh subagent per case, given only `SKILL.md` and the input, no session context:

| # | Input | Checks beyond layout + 3 spot-checked numbers |
|---|---|---|
| 1 | Fixed run link 39558375380 | aggregate-only fallback footnote, p99 `—` |
| 2 | Search run link 39642191354 (266 MB zip) | bytes downloaded ≪ zip size (aiperf per probe + history + winner server metrics ≤ cap: a few MB to ~35 MB); probe table; feasible_max headline |
| 2b | Several `sweep-*` run links of one grid (from `/tmp/h200sweep-chk` runs) | search summary matrix, `✗`/`*` markers |
| 3 | Local folder `/tmp/fill-1015/38843679057` | p99 present, CPU tier present |
| 4 | Raw aiperf folder `/mnt/poc_tencent/artifacts/*` | "no server metrics logged", client-side hit shown |
| 5 | Two runs A/B | delta column, baseline marked, signed colours in HTML |
| 6 | Run with failed jobs or expired artifacts | clear message, partial report, no crash |
| 7 | The fixed-CCU synthetic test run (4 jobs, HAN 1×H200) | three fixed-CCU matrices, nominal 0/50 vs measured overall, CPU tier present |

## Out of scope

- Writing router hit rate into the aggregate pipeline (`smg_*` mapping): separate PR if needed.
- Uploading a small `summary_<RF>` artifact from workflows.
- Non-aiperf harnesses: upstream fixed-seq `bmk_*` / `agg_<RF>.json` (`benchmark_serving`, no tiered cache) and
  evals. The layout leaves room to add them later.
