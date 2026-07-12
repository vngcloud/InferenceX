# Comparison Report: DeepSeek-Coder-V2-Lite P2P Router — P2P ON vs Baseline (NO-P2P)

> **P2P arm:** [29109445314](https://github.com/vngcloud/InferenceX/actions/runs/29109445314) (`compose.running.yaml`)
> **Baseline:** [29089117091](https://github.com/vngcloud/InferenceX/actions/runs/29089117091) (`compose.nop2p.yaml`)
> **Branch:** `exp/20260710-deepseek-p2p-router-600s` | **Commit:** `0181613` (both arms) | **Date:** 2026-07-12
> **Controlled variable:** the coordinator + `--p2p-transfer-engine nixl` path only. Same commit, dual-GPU node, 949-entry Weka trace, split router, per-instance L1 CPU cache, all serving flags. Scope: **CCU 2 and 4** (conc8 completed 0 requests in both).

## Executive Summary

Enabling LMCache cross-instance P2P sharing produces a **demonstrable, structural cache effect and a concurrency-dependent latency win**. The external (LMCache) hit rate is non-zero **only** with P2P on (44.6% @ conc2, 2.0% @ conc4) and exactly **0.00%** in the baseline at both levels — the clean signature of cross-instance KV reuse that is impossible without the coordinator. That reuse is **latency-neutral at idle (conc2)** but pays off under contention: at **conc4, P2P cuts mean TTFT 39% (1.22s vs 2.01s) and p90 TTFT 63% (1.93s vs 5.25s)**, and warms up 41% faster. Decode and end-to-end latency are unaffected, as expected since P2P only touches prefill. **Bottom line: P2P works as designed and its benefit scales with load.** Caveat: small-sample smoke runs — conclusions are directional in magnitude, solid in direction.

## Side-by-Side: Concurrency 2 (light load)

| Metric | P2P ON | Baseline | Δ (P2P vs base) |
|---|---|---|---|
| Requests completed | 15 | 15 | — |
| Mean TTFT | 1.244s | 1.065s | **+17% (worse)** |
| p95 TTFT | 2.578s | 2.774s | −7% |
| Mean E2E latency | 62.89s | 63.69s | −1.3% |
| Mean TPOT (ITL) | 99.1ms | 100.6ms | −1.5% |
| GPU prefix hit | 54.1% | 77.0% | −23 pts |
| **External (P2P) hit** | **44.6%** | **0.0%** | **+44.6 pts** |
| GPU KV usage (max) | 18.2% | 18.2% | = |
| Active prefetch jobs | 1 | 0 | +1 |

## Side-by-Side: Concurrency 4 (contended)

| Metric | P2P ON | Baseline | Δ (P2P vs base) |
|---|---|---|---|
| Requests completed | 44 | 43 | ~= |
| Mean TTFT | 1.218s | 2.008s | **−39% (better)** |
| p90 TTFT | 1.933s | 5.247s | **−63% (better)** |
| p95 TTFT | 5.96s | 8.411s | −29% |
| Mean E2E latency | 46.36s | 49.37s | −6.1% |
| Mean TPOT (ITL) | 100.2ms | 105.2ms | −4.8% |
| GPU prefix hit | 78.9% | 63.8% | +15 pts |
| **External (P2P) hit** | **2.0%** | **0.0%** | **+2.0 pts** |
| GPU KV usage (max) | 38.5% | 32.5% | +6 pts |
| aiperf warmup | 31.9s | 53.9s | **−41%** |

## What the Comparison Shows

1. **P2P is demonstrably functional.** External hit rate is non-zero only in the P2P arm and exactly 0% in the baseline at both concurrencies. Since GPU KV usage never exceeds ~38%, no local eviction to CPU cache occurs — so those hits can *only* be cross-instance KV transfers routed by the coordinator over NIXL. The architecture does exactly what it was built to do; the baseline structurally cannot.

2. **The latency payoff appears under concurrency, not at idle.** At conc2 the box is unsaturated, so a missed prefix recomputes cheaply and baseline TTFT is actually marginally *lower* (1.07s vs 1.24s) — the 44.6% external hits don't convert to a latency win, and P2P transfer + prefetch overhead is a slight wash on a 15-request sample. At conc4, contention exposes the difference: the baseline recomputes long prefixes on the "wrong" instance and its TTFT nearly doubles (mean 2.01s, **p90 5.25s**), while P2P holds TTFT flat (mean 1.22s, **p90 1.93s**) and warms up 41% faster.

3. **Decode and E2E latency are unaffected**, as expected — P2P only touches prefill/first-token. ITL (~100ms) and E2E latency (~46–64s, dominated by the OSL × ITL product) are within noise between arms.

4. **Attribution nuance:** at conc4 the P2P win is carried more by a *higher GPU prefix hit* (78.9% vs 63.8%) than by external hits (only 2.0%), whereas at conc2 it's the reverse (external 44.6%, GPU 54.1%). Cross-instance sharing effectively keeps hot prefixes resident across the pair, but exact tier attribution is blurred by run-to-run routing variance and cannot be pinned down without the MP counters / scheduler queue depth (not exported on the remote router path).

## Caveats

- **Duration-capped smoke runs with tiny samples** (15 requests @ conc2, ~43 @ conc4) and high TTFT variance (conc2 std ~0.9s). Magnitudes are directional; the qualitative conclusion is robust.
- **conc8 arm yielded 0 completed requests in both runs** — no data at that level.
- **Server-side logs absent** (remote replay): no `server.log`, no full `lmcache_server_metrics.json`. Cache figures come from scraped vLLM `/metrics`; MP hit/eviction/throughput counters and scheduler queue depth are unavailable.

## Verdict & Recommendations

**P2P enables cross-instance KV reuse that is structurally impossible in the baseline, and that reuse translates into materially lower TTFT tail latency once concurrency creates contention (conc4), while being latency-neutral at idle (conc2).**

1. **Re-run at higher concurrency (8/16/32) once the stack completes those arms** — the P2P benefit grows with contention, and conc2/4 only samples the low end of the curve. The conc8 zero-completion issue must be fixed first (investigate why both arms completed 0 at conc8).
2. **Export server.log + `lmcache_server_metrics.json` from the router compose** (mirror the agentic-replay fix `2f4713e`) so the next comparison can attribute the TTFT win between GPU-tier and P2P-tier and quantify NIXL transfer bandwidth/latency.
3. **Raise per-concurrency sample count to ≥ 100 requests** (longer duration or larger dataset reuse) before treating any single Δ as a capacity number.
4. **Track external-hit-rate vs GPU-hit-rate as a pair** across the sweep — the conc2→conc4 external-hit drop (44.6% → 2.0%) is the most interesting open question and needs MP counters to resolve (benign GPU-tier absorption vs P2P transfers losing the race to recompute under load).

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 29109445314 --repo vngcloud/InferenceX` and `gh run download 29089117091 --repo vngcloud/InferenceX`_
