# Benchmark Report: deepseek-p2p-router 900s conc2-4-8 n949 (baseline, NO-P2P)

> **Run:** [29089117091](https://github.com/vngcloud/InferenceX/actions/runs/29089117091) | **Branch:** `exp/20260710-deepseek-p2p-router-600s` | **Status:** ✅ success
> **Commit:** `0181613` — test: extend P2P router smoke to 900s, 949 dataset entries
> **Date:** 2026-07-12
> **Topology:** `deploy/p2p-router/compose.nop2p.yaml` — identical 2× vLLM + 2× LMCache + split router, but **no coordinator** and the four P2P flags stripped. Each LMCache keeps only its own local L1; peers cannot discover each other.

## Executive Summary

The fair **no-P2P baseline** for the DeepSeek-Coder-V2-Lite-FP8 dual-instance stack: same hardware, dataset, router, and per-instance L1 CPU cache as the P2P arm — the only difference is the coordinator/NIXL path is removed. The defining result is a **hard 0.00% external (LMCache) hit rate at both conc2 and conc4**: with no peer discovery and no GPU eviction (KV usage ≤ 33%), LMCache is never populated in a way any request can retrieve, so every cross-instance prefix miss must recompute. The cost surfaces at conc4, where **mean TTFT nearly doubles vs conc2 (2.01s vs 1.07s, p90 5.25s)**. Verdict: **healthy baseline, behaving exactly as expected for P2P-off.** Small-sample smoke run — directional.

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8` (served as `deepseek-coder-v2-lite-fp8`) |
| Framework / Image | `vllm` / `lmcache/vllm-openai:latest` |
| Precision | `fp8` TP`1` (per instance) |
| Hardware | dual-GPU node (GPU0 + GPU1); benchmark client remote (`benchmark-client`) |
| Concurrency | 2 and 4 (conc8 arm produced 0 completed requests — excluded) |
| Dataset / Scenario | 949-entry Weka agentic-replay trace (`agentic-coding`) |
| Duration override | `900s` |
| LMCache | `yes (local L1 only)` — 2 servers, `chunk-size=528`, `l1-size-gb=3`, LRU, blake3; **no coordinator, no P2P flags** |
| max-model-len / max-num-seqs / mnbt | 80000 / 4 / 4096 per instance |

## Performance

| Metric | conc 2 | conc 4 |
|---|---|---|
| Requests completed | 15 / 15 | 43 / 43 |
| Benchmark window | 745.45s | 924.67s |
| Mean TTFT | **1.065s** (p90 1.803 · p95 2.774) | **2.008s** (p90 5.247 · p95 8.411) |
| Mean TPOT (ITL) | 100.6ms (~9.9 tok/s decode) | 105.2ms (~9.5 tok/s decode) |
| Mean E2E latency | **63.69s** (p90 115.6 · p95 167.1) | **49.37s** (p90 121.8 · p95 145.6) |
| System input throughput | ~1095 tok/s (~547/GPU) | ~2058 tok/s (~1029/GPU) |
| System output throughput | ~20.8 tok/s (~10.4/GPU) | ~39.7 tok/s (~19.9/GPU) |
| Avg ISL / OSL | 34,259 / 618 tokens | 27,462 / 451 tokens |
| Total prompt / gen tokens | 815,940 / 15,526 | 1,902,535 / 36,740 |
| Mean QPS | 0.0220 | 0.0475 |

_Throughput computed as total tokens ÷ benchmark window; "/GPU" = system ÷ 2 backends. GPU power/tok-per-watt: N/A — GPU telemetry not exported for the remote router path._

## Cache

### GPU / External tier _(from scraped vLLM `/metrics`, `server_metrics_export.json`)_

| Metric | conc 2 | conc 4 |
|---|---|---|
| GPU prefix hit rate | 77.02% (628,400 / 815,940 tok) | 63.80% (1,213,776 / 1,902,535 tok) |
| **External (LMCache) hit rate** | **0.00%** (0 / 187,540 tok) | **0.00%** (0 / 688,759 tok) |
| GPU KV usage (avg / max / min) | 2.8% / 18.2% / 0% | 10.3% / 32.5% / 0% |
| Prompt tokens cached | 628,400 | 1,213,776 |

### LMCache MP internal

_Full MP scrape not available for the remote router path — only embedded `lmcache_mp_*` gauges present; hit/eviction/throughput counters null._

| Metric | conc 2 | conc 4 |
|---|---|---|
| L1 (CPU DRAM) usage ratio | 71.5% (15.364 GB) | 43.8% (9.409 GB) |
| Active prefetch jobs at scrape | **0** | 1 |
| MP lookup / L2 hit rate, eviction, throughput | N/A (counters null on remote path) | N/A |

_Note: conc2 shows **0 active prefetch jobs** — consistent with the P2P prefetch path being off in this arm._

## Stack Initialization

- [–] LMCache version / block-size-align / connector / heartbeat / hybrid-KV: **N/A — remote replay, server.log not in artifacts.**
- [✓] Connector wired and healthy — metrics scraped, decode steady at ~9.5–9.9 tok/s.
- [✓] External hit rate exactly 0% at both concurrencies — consistent with a correctly-wired connector that has no peer to fetch from and no local eviction to populate CPU cache (not a bug).

## Scheduler Health

N/A — remote replay; queue samples require server.log. Indirect signal: conc4 mean TTFT 2.01s with p90 5.25s (vs conc2's 1.07s / 1.80s) points to concurrent long-prefix recomputes queueing against each other under load.

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup (conc2) | — | — | — | 12.57s elapsed |
| Profiling (conc2) | 17 | 16 | 1 | true (normal for duration run) |
| Warmup (conc4) | — | — | — | **53.87s elapsed** |
| Profiling (conc4) | 58 | 55 | 3 | true |

_Errors: 1 (conc2), 12 (conc4) — all end-of-window cancellations (`grace_period_timeout=true`), `error_records=0`. conc4 warmup of 53.9s is notably longer than the P2P arm's 31.9s — an early sign of higher first-token cost without cross-instance reuse._

## Anomalies

- **External hit rate is a hard 0.00%** at both concurrencies (0 / 187,540 and 0 / 688,759). Structural, not a fault — see §Root Cause.
- **conc4 mean TTFT 2.008s (p90 5.247s)** — ~1.9× the conc2 mean and ~2.7× the P2P arm's conc4 p90 (1.933s).
- conc4 warmup 53.87s vs P2P's 31.92s (+69%).
- Small samples (15 / 43 completed) → directional only. conc8 arm: 0 completed requests (excluded).

## Root Cause Analysis

With the coordinator removed, the split router still alternates turns across vllm-a/vllm-b, but a turn landing on the instance that lacks the prefix has **no peer to pull from and must recompute** — there is no cross-instance visibility. Local LMCache CPU cache cannot fill the gap either: GPU KV usage peaks at only 18–33%, so the GPU pool never evicts blocks down to CPU, leaving the local LMCache store empty of anything a later request would query — hence external hits are exactly 0 (the 0-active-prefetch-jobs gauge at conc2 corroborates the dormant P2P/prefetch path). The recompute cost is invisible at conc2 (light load; a missed prefix recomputes cheaply, TTFT ~1.07s) but compounds at conc4, where concurrent long-prefix (mean 27k-token) recomputes queue against each other, pushing mean TTFT to 2.01s and p90 to 5.25s and stretching warmup to 53.9s. Decode (ITL ~100–105ms) and E2E latency are unaffected because P2P only touches prefill/first-token.

## Recommendations

1. **Use this as the control against the P2P arm (run 29109445314)** — the 0% external hit rate is the reference that makes the P2P arm's 44.6% interpretable. See the companion comparison report.
2. **Raise sample size** — 15 requests at conc2 is too few for stable tails; extend duration or dataset reuse to ≥ 100 requests/concurrency before capacity claims.
3. **Export server.log + `lmcache_server_metrics.json`** from the router path so scheduler queue depth and MP counters are available to confirm the recompute-queueing hypothesis at conc4 directly.
4. **If baseline TTFT at higher concurrency matters,** either raise `--max-num-seqs`/`mnbt` per instance or add sticky-session routing so a conversation's turns stay on one instance and reuse its GPU prefix cache — the only reuse channel available when P2P is off.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 29089117091 --repo vngcloud/InferenceX`_
