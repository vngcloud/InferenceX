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
<!-- [Agent] The rows above are the minimum set. Add any additional performance metrics
     present in agg.json or profile.json that are not listed above (e.g. p90_ttft,
     p50_e2el, output_tput_per_gpu breakdown, per-phase latencies). Label them plainly. -->

## Cache

### GPU / External tier

| Metric | Value |
|---|---|
| GPU prefix hit rate | `<gpu_hit_rate_pct>%` (`<gpu_hits>` / `<gpu_queries>` tokens) |
| External (LMCache) hit rate | `<ext_hit_rate_pct>%` (`<ext_hits>` / `<ext_queries>` tokens) |
| GPU KV usage (avg / max / min) | `<kv_usage_avg_pct>%` / `<kv_usage_max_pct>%` / `<kv_usage_min_pct>%` |
| Prompt tokens cached | `<prompt_tokens_cached>` |
<!-- [Agent] Add any extra vllm:* counter rows you find in server_metrics.json that
     are relevant to understanding cache behavior (e.g. eviction counts, swap counts). -->

### LMCache MP internal _(MP connector path only — omit entire subsection for V1 connector or SGLang)_

<!-- [Agent] Include this subsection only when lmcache_server_metrics.json was present
     and lmcache_mp_* fields are non-null in the parser output. If the subsection is
     absent or all fields are null, replace it with a single line:
     "_LMCache MP scrape not available for this run._"
     See docs/lmcache-metrics.md for the full field→metric→connector→impact table.
     V1 fields are null when running MP connector. MP L1 bandwidth throughput fields (⚠️)
     are null until metric names are confirmed. MP L2 fields are null for L1-only runs. -->

| Metric | Value |
|---|---|
| MP lookup hit rate (L1 + L2) | `<mp_hit_rate_pct>%` (`<mp_hit_tokens>` / `<mp_query_tokens>` tokens) |
| L2 prefetch hit rate | `<mp_l2_hit_rate_pct>%` |
| L2 prefetch failures | `<mp_l2_prefetch_fails>` |
| L1 (CPU DRAM) usage | `<mp_l1_usage_ratio × 100>%` (`<mp_l1_memory_gb> GB`) |
| Active prefetch jobs at scrape | `<mp_active_prefetch_jobs>` |
| MP L1 write / read chunks | `<lmcache_mp_l1_write_chunks>` / `<lmcache_mp_l1_read_chunks>` |
| MP L1 evicted chunks | `<lmcache_mp_l1_evicted_chunks>` |
| MP L1 eviction pressure ratio | `<lmcache_mp_l1_eviction_loop_triggered>`/`<lmcache_mp_l1_eviction_loop_ticks>` |
| MP L1 read throughput p50/p95 | `<lmcache_mp_l1_read_throughput_GBps_p50>` / `<lmcache_mp_l1_read_throughput_GBps_p95>` GB/s ⚠️ |
| MP L1 write throughput p50/p95 | `<lmcache_mp_l1_write_throughput_GBps_p50>` / `<lmcache_mp_l1_write_throughput_GBps_p95>` GB/s ⚠️ |
| MP L2 load throughput p50/p95 | `<lmcache_mp_l2_load_throughput_GBps_p50>` / `<lmcache_mp_l2_load_throughput_GBps_p95>` GB/s |
| Stored tokens (V1) | `<lmcache_stored_tokens>` |
| Retrieve latency p50/p95 (V1) | `<lmcache_retrieve_latency_ms_p50>`ms / `<lmcache_retrieve_latency_ms_p95>`ms |
| Retrieve speed p50/p95 (V1) | `<lmcache_retrieve_speed_GBps_p50>` / `<lmcache_retrieve_speed_GBps_p95>` GB/s |
<!-- [Agent] If lmcache_server_metrics.json contains additional lmcache_mp_* counters
     beyond the above (check the raw JSON), add them as extra rows with a plain-English
     label. Never silently drop a non-null metric. -->

## Stack Initialization

- [✓/✗] LMCache version: `<lmcache_version>` _(need 0.5.0 for hybrid-attention models)_
- [✓/✗] Attention block size aligned: `<block_size_align>` tokens _(N/A if pure-attention)_
- [✓/✗] LMCacheMPConnector loaded
- [✓/✗] Heartbeat thread running
- [✓/✗] Hybrid KV manager ON _(✗ if hybrid_kv_turned_off=true)_
- [✓/✗] No connector crash at startup
<!-- [Agent] Add any other initialization signals you found in server.log that are
     relevant to correctness — e.g. vLLM engine version, CUDA version mismatch warnings,
     chunked-prefill enabled/disabled, max-model-len value, tensor-parallel init lines.
     Use ✓/✗ bullets for binary checks; plain bullets for informational lines. -->

## Scheduler Health

Running avg / max: **`<running.avg>`** / `<running.max>`  
Waiting avg / max: **`<waiting.avg>`** / `<waiting.max>`  
_(from `<sample_count>` × 10s samples)_
<!-- [Agent] If server.log shows meaningful trends (e.g. waiting queue grew over time,
     running count collapsed mid-run, external hit rate ramped up as cache warmed),
     add a short "Trend" note beneath the stats rather than flattening them to avg/max. -->

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `<warmup_elapsed_s>s` elapsed |
| Profiling | `<profiling_sent>` | `<profiling_completed>` | `<profiling_in_flight_at_end>` | `<timeout_triggered>` |
| Grace period | — | `<profile.request_count - profiling_completed>` added | — | — |

_Errors: `<errors>`_
<!-- [Agent] If aiperf.log contains additional phase information (e.g. ramp-up details,
     cancellation counts, retry counts, connection errors), add it as extra rows or a
     note beneath the table. -->

## Observed log patterns _(optional — include only if noteworthy)_

<!-- [Agent] Include this section if server.log or aiperf.log contain patterns worth
     preserving for future debugging: recurring warnings, memory pressure events,
     connector reconnects, LMCache eviction messages, CUDA OOM recoveries, etc.
     Format as a short bullet list with the exact log line and count if repeated.
     Omit the section entirely if there is nothing unusual. -->

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

**The template is a minimum, not a ceiling.** Every fixed row must be filled. Beyond
that, add every non-null metric you can find — extra agg.json fields, extra
`lmcache_mp_*` counters, extra server.log signals, aiperf phase details. A metric that
exists in the data but is absent from the report is a silent omission; don't drop it
just because the template doesn't list it. If you're unsure whether a metric is worth
including, include it with a plain label.

**Executive Summary** — the user reads this first. Answer: what was tested, was it
healthy, what's the #1 thing to know? Don't just list what you'll say in later sections.

**Cache section** — always include both subsections when the MP connector was used.
A non-null `mp_hit_rate_pct` with a zero `ext_hit_rate_pct` is informative: it means
LMCache is serving tokens but they never reached the external-counter path. Explain that
in §Root Cause rather than treating the two hit rates as duplicates.

**Anomaly bullets** — every bullet must have a number and a "so what". Bad: "TTFT is
high." Good: "TTFT 56.5s (p50 67s) — 10× the healthy target for conc=16 on H100; root
cause is KV saturation (see §Root Cause)."

**Root Cause Analysis** — this is the paragraph the user will share with their team.
Make every sentence load-bearing. Connect metrics causally, don't echo them.

**Recommendations** — be specific. "Reduce concurrency" is weak. "Reduce concurrency
from 16 → 4 to keep KV usage below 40% (16 × 42k > GPU KV budget; 4 × 42k fits
comfortably)" is useful.
