---
name: write-bench-report
description: >
  Write, commit, and push a structured benchmark report for a completed InferenceX run.
  Invoked automatically from inspect-run after a successful run when the user agrees to
  persist the report. Saves a rich Markdown file to reports/ in the dev-lmcache branch
  with full performance tables, cache analysis, scheduler health, anomaly list,
  root-cause analysis, and actionable recommendations. Also use this skill whenever the
  user asks to "save the report", "write up the results", "commit the findings", or
  "add a report for this run".
---

# Write Bench Report

You have already inspected the run (via `inspect-run`) and have the parser JSON, run
metadata, and git context in hand. Your job now is to write a polished Markdown report,
save it to `reports/` in the repo, and push it to `dev-lmcache`.

## Step 1 — Assemble inputs

Gather everything you need from the inspect-run context (all of this should already be
in your conversation):

| Input | Source |
|---|---|
| Run ID, display title, branch, conclusion | `gh run view` output |
| Commit SHA + message | `git show --stat` output |
| Parser JSON | output of `parse_metrics.py` |
| Dataset / scenario | config key in `nvidia-master.yaml` |
| LMCache config (chunk_size, DRAM budget) | launch script or server.log init messages |
| Date | today's date (YYYY-MM-DD) |

If any field is missing, derive it from what you have — don't ask the user for data
that's already in the logs.

## Step 2 — Determine the report filename

Use this naming convention:

```
reports/YYYY-MM-DD_<display-title-slug>_<RUN_ID>.md
```

- Slug = display title lowercased, spaces replaced with `-`, `e2e-test-` prefix stripped
- Example: `reports/2026-06-25_gemma4-lmcache-minimax-smoke_28162657488.md`

## Step 3 — Write the report

Use this exact template. Fill every field — never leave a placeholder unfilled.
If a metric is absent from the parser output (e.g., `block_size_align` is null for
pure-attention models), note "N/A — pure-attention model" rather than leaving it blank.

---

```markdown
# Benchmark Report: <displayTitle>

> **Run:** [<RUN_ID>](https://github.com/vngcloud/InferenceX/actions/runs/<RUN_ID>) | **Branch:** `<branch>` | **Status:** ✅ <conclusion>
> **Commit:** `<sha>` — <commit message one-liner>
> **Date:** <YYYY-MM-DD>

## Executive Summary

<2–3 sentences. State what was benchmarked (model, stack, dataset), the dominant
finding (e.g., "TTFT of 56.5s reveals KV cache saturation at conc=16"), and the
verdict: healthy / degraded / config issue / expected for smoke.>

## Configuration

| Field | Value |
|---|---|
| Model | `<perf.model>` |
| Framework / Image | `<perf.framework>` / `<image from config>` |
| Precision | `<perf.precision>` TP`<perf.tp>` |
| Hardware | `<perf.hw>` |
| Concurrency | `<perf.conc>` |
| Dataset / Scenario | `<input_file basename>` (`<scenario_type>`) |
| Duration override | `<duration>s` |
| LMCache | `yes (v<version>, chunk_size=<N>, DRAM <Xg>)` — or `no` |

## Performance

| Metric | Value |
|---|---|
| Requests completed | `<profile.request_count>` |
| Benchmark window | `<profile.benchmark_duration>s` |
| Mean TTFT | **`<mean_ttft>s`** (p50 `<p50_ttft>s` · p99 `<p99_ttft>s`) |
| Mean TPOT | `<mean_tpot_ms>ms` (~`<mean_intvty>` tok/s decode) |
| Total throughput/GPU | `<tput_per_gpu>` tok/s |
| Input / Output tput | `<input_tput_per_gpu>` / `<output_tput_per_gpu>` tok/s |
| Mean E2E latency | **`<mean_e2el>s`** (p99 `<p99_e2el>s`) |
| Avg ISL / OSL | `<isl>` / `<osl>` tokens |
| GPU power | `<mean_power_w>W` (`<tok_per_watt>` tok/W) |

## Cache

| Metric | Value |
|---|---|
| GPU prefix hit rate | `<gpu_hit_rate_pct>%` (`<gpu_hits>` / `<gpu_queries>` tokens) |
| External (LMCache) hit rate | `<ext_hit_rate_pct>%` (`<ext_hits>` / `<ext_queries>` tokens) |
| GPU KV usage (avg / max) | `<kv_usage_avg_pct>%` / `<kv_usage_max_pct>%` |
| Prompt tokens cached | `<prompt_tokens_cached>` |

## Stack Initialization

- [✓/✗] LMCache version: `<lmcache_version>` _(need 0.5.0 for hybrid-attention models)_
- [✓/✗] Attention block size aligned: `<block_size_align>` tokens _(N/A if pure-attention)_
- [✓/✗] LMCacheMPConnector loaded
- [✓/✗] Heartbeat thread running
- [✓/✗] Hybrid KV manager ON _(✗ if hybrid_kv_turned_off=true)_
- [✓/✗] No connector crash at startup

## Scheduler Health

Running avg / max: **`<running.avg>`** / `<running.max>`  
Waiting avg / max: **`<waiting.avg>`** / `<waiting.max>`  
_(from `<sample_count>` × 10s samples)_

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `<warmup_elapsed_s>s` elapsed |
| Profiling | `<profiling_sent>` | `<profiling_completed>` | `<profiling_in_flight_at_end>` | `<timeout_triggered>` |
| Grace period | — | `<profile.request_count - profiling_completed>` added | — | `<grace_period_timeout>` |

_Errors: `<errors>`_

## Anomalies

<!-- Use specific numbers; never write "high TTFT" — write "TTFT 56.5s (p50 67s), 56× above
     the < 1s target for this concurrency" or "10/17 profiling requests cancelled because
     p99 E2E 201s exceeds the 120s grace period". -->

<bullet list — one line per anomaly, each with the exact metric value and why it matters.
 Write "None detected — all metrics within expected ranges." if the run is clean.>

## Root Cause Analysis

<4–6 sentences. Explain WHY each anomaly exists. Connect numbers causally — trace the
chain from root cause to observed metric. Example pattern:
"X is caused by Y (value Z), which leads to W (value V). This explains the high Q
because [mechanism]. For this to improve, [condition]."
Don't restate what the tables already show — explain the mechanism.>

## Recommendations

<Numbered, actionable list. Each item: what to change, what flag/value, what result to
expect. At least one item for immediate next experiment. Flag any items that would be
breaking changes or require scheduler/config coordination.>

1. ...
2. ...
3. ...

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download <RUN_ID> --repo vngcloud/InferenceX`_
```

---

## Step 4 — Save to the reports/ folder

Ensure the `reports/` directory exists and write the file:

```bash
cd d:/projects/InferenceX
git status  # confirm we're on dev-lmcache or can switch safely
git switch dev-lmcache 2>/dev/null || true
git pull origin dev-lmcache  # bring in latest before appending
mkdir -p reports
```

Write the full report Markdown to `reports/<filename>.md` using the Write tool.

## Step 5 — Commit and push

Stage only the new report file (never stage unrelated working-tree changes):

```bash
cd d:/projects/InferenceX
git add reports/<filename>.md
git commit -m "report(<run-title-slug>): add benchmark report for run <RUN_ID>

<model> <precision> TP<tp> on <hw> — <scenario_type> <dataset_basename>
Status: <conclusion> | Concurrency: <conc> | TTFT p50: <p50_ttft>s"
git push origin dev-lmcache
```

Confirm the push succeeded and tell the user the file path in the repo.

## What makes a good report

**Executive Summary** — the user reads this first. Answer: what was tested, was it
healthy, what's the #1 thing to know? Don't just list what you'll say in later sections.

**Anomaly bullets** — every bullet must have a number and a "so what". Bad: "TTFT is
high." Good: "TTFT 56.5s (p50 67s) — 10× the healthy target for conc=16 on H100; root
cause is KV saturation (see §Root Cause)."

**Root Cause Analysis** — this is the paragraph the user will share with their team.
Make every sentence load-bearing. Connect metrics causally, don't echo them.

**Recommendations** — be specific. "Reduce concurrency" is weak. "Reduce concurrency
from 16 → 4 to keep KV usage below 40% (16 × 42k > GPU KV budget; 4 × 42k fits
comfortably)" is useful.
