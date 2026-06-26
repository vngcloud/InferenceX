# Benchmark Report: e2e Test - gemma4-lmcache-minimax-smoke-metrics

> **Run:** [28213679019](https://github.com/vngcloud/InferenceX/actions/runs/28213679019) | **Branch:** `exp/gemma4-lmcache-minimax-gpu0.65-dram20g` | **Status:** ✅ success
> **Commit:** `9558776` — fix(lmcache-metrics): add scrape_lmcache_server_metrics to gemma4-lmcache-minimax script
> **Date:** 2026-06-26

## Executive Summary

First smoke run of `RedHatAI/gemma-4-31B-it-FP8-dynamic` with the `scrape_lmcache_server_metrics` call wired in, on the MiniMax Claude-Code trace at conc=2. The stack initialized correctly and ran cleanly (0 errors). However, `lmcache_server_metrics.json` was not uploaded because the artifact upload path covered only the agentic-coding path (`results/`) while this agentic-replay script writes the file to the host CWD root — a bug in `benchmark-tmpl.yml` fixed in the next run (28214899672). All `mp_*` fields are null. Performance at conc=2 with ISL=4096 MiniMax sessions is dominated by long per-session context; mean TTFT 31.8s is skewed by two heavy in-flight sessions that extended the benchmark window to 205s.

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| Framework / Image | `vllm` / `vllm/vllm-openai:v0.23.0` |
| Precision | `fp8` TP`1` |
| Hardware | `h100-greennode_00` |
| Concurrency | `2` |
| Dataset / Scenario | `minimax_claude_code_prod_v3.jsonl` (`agentic-replay`) |
| Duration override | `120s` |
| LMCache | `yes (v0.5.0, chunk_size=256, DRAM 20 GB, MP connector ZMQ :5555)` |

## Performance

| Metric | Value |
|---|---|
| Requests completed | `7` |
| Benchmark window | `205.0s` (120s profiling + ~85s grace for 2 in-flight) |
| Mean TTFT | **`31.8s`** (p50 `24.0s` · p90 `57.9s` · p99 `71.0s`) |
| Mean TPOT | `66.7ms` (~`15.0` tok/s decode) |
| Total throughput/GPU | `2,514.8` tok/s |
| Input / Output tput | `2,506.3` / `8.4` tok/s |
| Mean E2E latency | **`51.7s`** (p50 `51.2s` · p99 `102.3s`) |
| Avg ISL / OSL | `4,096` / `512` tokens (nominal first-turn; actual avg ~73k tokens total ISL across all turns) |
| GPU power | `607.3W` (`4.14` tok/W) |

## Cache

### GPU / External tier

| Metric | Value |
|---|---|
| GPU prefix hit rate | `20.04%` (`103,008` / `513,885` tokens) |
| External (LMCache) hit rate | `10.90%` (`44,800` / `410,877` tokens) |
| GPU KV usage (avg / max / min) | `68.1%` / `98.8%` / `0.0%` |
| Prompt tokens cached | `221,408` |

### LMCache MP internal _(MP connector path only)_

_LMCache MP scrape not available for this run — `lmcache_server_metrics.json` was produced by the script but not uploaded due to path mismatch in `benchmark-tmpl.yml` (fixed in run [28214899672](https://github.com/vngcloud/InferenceX/actions/runs/28214899672))._

## Stack Initialization

- [✓] LMCache version: `0.5.0`
- [✓] Attention block size aligned: N/A — pure-attention model (Gemma 4, no Mamba layers)
- [✓] LMCacheMPConnector loaded
- [✓] Heartbeat thread running
- [✓] Hybrid KV manager ON _(hybrid_kv_turned_off=false)_
- [✓] No connector crash at startup

## Scheduler Health

Running avg / max: **`1.5`** / `2`
Waiting avg / max: **`0.2`** / `1`
_(from `63` × 10s samples)_

Cache warming trend: GPU hit rate 0.5% → 33.6%, ext hit rate 0% → 8.9% over the 205s window.

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `527.0s` elapsed |
| Profiling | `7` | `5` | `2` | `true` (120s duration) |
| Grace period | — | `2` added | — | ~85s extended |

_Errors: `0`_

## Anomalies

- **`mp_*` fields all null**: `lmcache_server_metrics.json` was produced by `scrape_lmcache_server_metrics /workspace/ 8080` but the artifact upload step in `benchmark-tmpl.yml` only listed `results/lmcache_server_metrics.json` (agentic-coding path), not the bare `lmcache_server_metrics.json` (agentic-replay path). Fixed in commit `e1a4d91`.
- **Mean TTFT 31.8s vs p50 24.0s**: 2 in-flight requests at profiling end were the longest sessions in the batch. Their TTFT counted but their request completion skews the mean upward while p50 (24.0s) is more representative.
- **Benchmark window extended to 205s**: the 2 heavy in-flight requests required ~85s of grace period to complete beyond the 120s profiling window.

## Root Cause Analysis

The upload bug is the primary artifact of this run: the script correctly called `scrape_lmcache_server_metrics /workspace/ 8080` and wrote `lmcache_server_metrics.json` to `/workspace/` (host CWD root in mooncake-trace execution), but `benchmark-tmpl.yml`'s upload spec only matched `results/lmcache_server_metrics.json` — the agentic-coding path where `RESULT_DIR=results/`. Since GitHub Actions `if-no-files-found: ignore` silently skips missing paths, the file was dropped from the artifact with no CI error.

Performance is within expected range for this workload: conc=2 on a 31B FP8 model with MiniMax multi-turn sessions (ISL growing to 60-80k tokens across turns) will see TTFT of 20-35s. At KV max 98.8%, eviction is occurring and the ext hit rate (10.9%) confirms LMCache is serving evicted blocks back. The mean/p50 gap on TTFT (31.8s vs 24.0s) is explained by 2 outlier sessions that ran longer than average.

## Recommendations

1. **See run 28214899672** for verified `mp_*` metrics — the upload fix is confirmed there.
2. **Do not use this run's TTFT mean (31.8s) as a reference** — it is skewed by 2 heavy outlier sessions in a n=7 smoke sample. The p50 (24.0s) and the next run's p50 (22.3s) are more comparable.
3. **Extend grace period to 180s** (`BENCHMARK_GRACE_PERIOD=180`) if running at conc=2 with MiniMax sessions: p99 E2E 102s exceeds the default 120s grace ceiling and could cancel long sessions at higher concurrency.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28213679019 --repo vngcloud/InferenceX`_
