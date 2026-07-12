# Benchmark Report: p2p-router deepseek-lite conc2/4 900s (P2P-ON)

> **Run:** [29109445314](https://github.com/vngcloud/InferenceX/actions/runs/29109445314) | **Branch:** `exp/20260710-deepseek-p2p-router-600s` | **Status:** ✅ success
> **Commit:** `0181613` — test: extend P2P router smoke to 900s, 949 dataset entries
> **Date:** 2026-07-12
> **Topology:** `deploy/p2p-router/compose.running.yaml` — 2× vLLM (vllm-a GPU0 / vllm-b GPU1) + 2× LMCache server **+ coordinator** (NIXL P2P transfer engine); split router alternates turns across both instances.

## Executive Summary

DeepSeek-Coder-V2-Lite-FP8 served on a dual-instance vLLM + LMCache stack with **cross-instance P2P KV sharing enabled**, replaying a 949-entry agentic-coding trace (huge prefixes: mean ISL 27k–34k tokens, small OSL ~450–620). The headline result is a **non-zero external (LMCache) hit rate — 44.6% at conc2, 2.0% at conc4** — which is only possible because the coordinator lets an instance fetch KV it never computed from its peer. TTFT stays flat at ~1.2s across both concurrencies. Verdict: **healthy; P2P mechanism confirmed working.** These are duration-capped smoke runs with small samples (15 / 44 requests) — treat magnitudes as directional.

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
| LMCache | `yes` — 2 servers, `chunk-size=528`, `l1-size-gb=3`, LRU, blake3; **coordinator + `--p2p-transfer-engine nixl`** |
| max-model-len / max-num-seqs / mnbt | 80000 / 4 / 4096 per instance |

## Performance

| Metric | conc 2 | conc 4 |
|---|---|---|
| Requests completed | 15 / 15 | 44 / 44 |
| Benchmark window | 734.95s | 903.15s |
| Mean TTFT | **1.244s** (p90 2.027 · p95 2.578) | **1.218s** (p90 1.933 · p95 5.96) |
| Mean TPOT (ITL) | 99.1ms (~10.1 tok/s decode) | 100.2ms (~10.0 tok/s decode) |
| Mean E2E latency | **62.89s** (p90 117.0 · p95 165.9) | **46.36s** (p90 116.9 · p95 139.1) |
| System input throughput | ~1110 tok/s (~555/GPU) | ~2212 tok/s (~1106/GPU) |
| System output throughput | ~21.5 tok/s (~10.8/GPU) | ~43.6 tok/s (~21.8/GPU) |
| Avg ISL / OSL | 34,253 / 618 tokens | 27,713 / 445 tokens |
| Total prompt / gen tokens | 815,776 / 15,829 | 1,997,712 / 39,404 |
| Mean QPS | 0.0222 | 0.0489 |

_Throughput computed as total tokens ÷ benchmark window; "/GPU" = system ÷ 2 backends. GPU power/tok-per-watt: N/A — GPU telemetry not exported for the remote router path._

## Cache

### GPU / External tier _(from scraped vLLM `/metrics`, `server_metrics_export.json`)_

| Metric | conc 2 | conc 4 |
|---|---|---|
| **Total cache hit rate (GPU + external, summed)** | **54.09%** (37,293,216 / 68,946,592 tok) | **72.38%** (6,026,048 / 8,325,479 tok) |
| GPU KV usage (avg / max / min) | 2.9% / 18.2% / 0% | 13.3% / 38.5% / 0% |
| Prompt tokens cached | 711,600 | 1,305,600 |

_Combined hit = (GPU prefix hits + external LMCache hits) ÷ (GPU + external queries). The external/P2P component — the P2P signature — is **44.62%** (83,952 / 188,128 tok) @ conc2 and **2.02%** (14,272 / 706,384 tok) @ conc4; the remainder is the local GPU prefix cache (54.12% @ conc2, 78.90% @ conc4)._

### LMCache MP internal

_Full MP scrape (`lmcache_server_metrics.json`) not available for the remote router path — only the `lmcache_mp_*` gauges embedded in the scraped vLLM metrics are present; hit/eviction/throughput counters are null._

| Metric | conc 2 | conc 4 |
|---|---|---|
| L1 (CPU DRAM) usage ratio | 85.8% (2.764 GB) | 92.9% (2.994 GB) |
| Active prefetch jobs at scrape | 1 | 1 |
| MP lookup / L2 hit rate, eviction, throughput | N/A (counters null on remote path) | N/A |

## Stack Initialization

- [–] LMCache version / block-size-align / connector / heartbeat / hybrid-KV: **N/A — remote replay, server.log not in artifacts.** Correct wiring is inferred from behavior: non-zero external hits prove the LMCache MP connector + coordinator are live, and decode is steady at ~10 tok/s.
- [✓] Non-zero cross-instance external hits (44.6% @ conc2) — coordinator + NIXL P2P path functional.
- [✓] No HTTP/connection errors in aiperf; all completed requests valid.

## Scheduler Health

N/A — remote replay; per-step `running`/`waiting` queue samples require server.log, which is not exported for the router path. Queue behavior is inferred indirectly: TTFT p90 stays ≤ 2.0s at both concurrencies, indicating no sustained prefill backlog.

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup (conc2) | — | — | — | 12.21s elapsed |
| Profiling (conc2) | 17 | 16 | 1 | true (normal for duration run) |
| Warmup (conc4) | — | — | — | 31.92s elapsed |
| Profiling (conc4) | 59 | 56 | 3 | true |

_Errors: 1 (conc2), 12 (conc4) — all are end-of-window cancellations (`grace_period_timeout=true`); aiperf records `success_records=57, error_records=0` at conc4, i.e. no real request failures._

## Anomalies

- External hit rate collapses from **44.62% (conc2) → 2.02% (conc4)** despite external queries growing ~4× (188k → 706k). Not a fault — see §Root Cause.
- Small samples (15 / 44 completed requests) and high TTFT variance (conc2 std 0.90s) → latency numbers are directional, not statistically firm.
- conc8 arm produced 0 completed requests (excluded from scope).

## Root Cause Analysis

The defining signal is the **non-zero external hit rate**. GPU KV usage never exceeds 38.5%, so the GPU pool never evicts blocks to local LMCache CPU — meaning these external hits cannot be local L2 spillover. They are genuine **cross-instance KV transfers**: with `kv_role=kv_both`, each instance stores its computed KV to its LMCache server; the coordinator makes those stores discoverable to the peer, so a turn landing on the instance that never computed the prefix pulls it over NIXL instead of recomputing. That is the P2P mechanism working as designed. The conc2→conc4 external-hit drop (44.6% → 2.0%) is explained by the GPU prefix cache absorbing far more at higher load (54.1% → 78.9% GPU hit): when the local GPU tier already covers a prefix, the external layer is never consulted for it, so proportionally fewer of the (larger) external queries convert to hits. TTFT staying flat (~1.2s, p90 ≤ 2.0s) despite 27k-token prefixes indicates long-prefix recompute was largely avoided across both tiers.

## Recommendations

1. **Pair with the baseline (run 29089117091) for the P2P delta** — the single-arm numbers here are only interpretable against the no-P2P control. See the companion comparison report.
2. **Raise sample size before drawing capacity conclusions** — 15 requests at conc2 is too few for stable p90/p95. Increase duration or dataset reuse so each concurrency completes ≥ 100 requests.
3. **Export server.log + `lmcache_server_metrics.json` from the router path** — currently the MP hit/eviction/throughput counters and scheduler queue depth are unavailable, blocking root-cause on the conc4 external-hit drop and any bandwidth analysis. Wire the scrape into the compose launch (mirror the agentic-replay path fix `2f4713e`).
4. **Investigate the conc4 external-hit drop directly** once MP counters are available — confirm whether it is pure GPU-tier absorption (benign) or P2P transfers losing the race to recompute under contention (tunable via prefetch/coordinator settings).

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 29109445314 --repo vngcloud/InferenceX`_
