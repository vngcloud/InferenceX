# Benchmark Report: gemma4-fp8-minimax-baseline-capacity

> **Run:** [28166668983](https://github.com/vngcloud/InferenceX/actions/runs/28166668983) | **Branch:** `exp/gemma4-fp8-lmcache-capacity` | **Status:** ✅ success
> **Commit:** `214809a` — feat(gemma4-lmcache): capacity sweep conc [2,4,8] for MiniMax-prod trace
> **Date:** 2026-06-25

## Executive Summary

Capacity sweep for Gemma4-31B-FP8 (vLLM TP1, H100 Greennode) against the MiniMax production agentic-replay trace at concurrencies 2, 4, and 8, with no LMCache (baseline control arm). The dominant finding is a hard capacity ceiling at conc=8: GPU KV usage reaches 99.6% max, GPU prefix cache hit rate collapses from 65.9% (conc=2) to 0.45% (conc=8), and throughput drops 30% (3,426 → 2,404 tok/s/GPU) while TPOT degrades 6.7×. Conc=2–4 runs are healthy; conc=8 marks the saturation breakpoint for single-GPU deployment and is the primary concurrency to compare against the paired LMCache run.

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| Framework / Image | `vllm` / `vllm/vllm-openai:v0.23.0` |
| Precision | `fp8` TP`1` |
| Hardware | `h100-greennode_00` |
| Concurrency | `2 / 4 / 8` (capacity sweep) |
| Dataset / Scenario | MiniMax-prod trace (`agentic-replay`) |
| Duration override | `900s` |
| LMCache | `no` (baseline) |

## Performance

| Metric | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Requests completed | 44 | 45 | 53 |
| Benchmark window | 865.8s | 913.5s | 968.7s |
| Mean TTFT | **11.80s** (p50 0.74s · p99 77.6s) | **20.85s** (p50 5.98s · p99 106.1s) | **40.82s** (p50 33.1s · p99 163.9s) |
| Mean TPOT | 48.4ms (~20.7 tok/s decode) | 171.9ms (~5.8 tok/s decode) | 322.7ms (~3.1 tok/s decode) |
| Total throughput/GPU | 3,426 tok/s | 3,318 tok/s | 2,404 tok/s |
| Input / Output tput/GPU | 3,417 / 9.3 tok/s | 3,309 / 9.6 tok/s | 2,388 / 16.7 tok/s |
| Mean E2E latency | **21.3s** (p99 111.1s) | **58.0s** (p99 170.3s) | **126.7s** (p99 416.2s) |
| Avg ISL / OSL | 4096 / 512 tokens (config) | 4096 / 512 tokens (config) | 4096 / 512 tokens (config) |
| GPU power | 418.9 W (8.18 tok/W) | 595.0 W (5.58 tok/W) | 622.3 W (3.86 tok/W) |

## Cache

| Metric | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| GPU prefix hit rate | **65.91%** (1,951,488 / 2,960,855 tok) | **40.85%** (1,235,520 / 3,024,896 tok) | **0.45%** (206,272 / 46,229,827 tok) |
| External (LMCache) hit rate | 0% — baseline, n/a | 0% — baseline, n/a | 0% — baseline, n/a |
| GPU KV usage (avg / max) | 20.7% / 62.6% | 49.6% / 85.7% | 71.4% / **99.6%** |
| Prompt tokens cached | 1,951,488 | 1,235,520 | 202,112 |

## Stack Initialization

- [N/A] LMCache version — baseline run, no LMCache wired
- [N/A] Attention block size aligned — baseline run
- [N/A] LMCacheMPConnector loaded — baseline run
- [N/A] Heartbeat thread running — baseline run
- [N/A] Hybrid KV manager ON — baseline run
- [✓] No connector crash at startup

All N/A init markers are expected; this is the no-LMCache control arm.

## Scheduler Health

| | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Running avg / max | **1.5** / 2 | **3.1** / 4 | **5.5** / 8 |
| Waiting avg / max | **0.0** / 0 | **0.1** / 2 | **1.2** / 6 |
| GPU hit rate (first → last sample) | 0.5% → 60.9% | 8.5% → 34.6% | 3.8% → 0.4% |
| Log sample count | 105 | 137 | 153 |

## aiperf Phase Summary

| Phase | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Warmup elapsed | 487.4s | 479.3s | 507.2s |
| Profiling sent | 63 | 64 | 64 |
| Profiling completed | 63 | 60 | 59 |
| In-flight at end | 0 | 4 | 5 |
| Timeout triggered | yes (normal) | yes (normal) | yes (normal) |
| Errors | 0 | 0 | 0 |

## Anomalies

- **GPU prefix cache collapses at conc=8: 0.45%** (vs 65.9% at conc=2, 40.9% at conc=4). KV max usage hits 99.6%, meaning every newly prefilled block is immediately evicted before the next request can reuse it. Prompt tokens cached drops from 1.95M (conc=2) to 202k (conc=8) despite more total requests — the cache is thrashing, not warming.

- **TPOT degrades 6.7× from conc=2 → conc=8**: 48.4ms → 322.7ms per output token (~20.7 → 3.1 tok/s decode). The GPU is simultaneously re-prefilling evicted long-context blocks and decoding for 8 concurrent sessions, which starves the decode step.

- **Total throughput drops 30% at conc=8**: 3,426 → 2,404 tok/s/GPU — the system has passed its throughput peak. Compute is being wasted on redundant prefill that would be cache hits at lower concurrency.

- **Warmup consumes 53–56% of the 900s run** across all concurrency levels (479–507s warmup), leaving only ~360–430s for the profiling window. Actual average ISL on the MiniMax trace is ~67k tokens per session (derived from total_isl / request_count), making first-session prefill slow at low concurrency regardless of the configured 4096-token target.

- **GPU hit rate trend reverses at conc=8**: first log sample is 3.8%, last is 0.4% — the cache is not building up over time (as at conc=2 where it climbs 0.5% → 60.9%), but actively losing useful blocks as more sessions compete for the fixed GPU KV pool.

## Root Cause Analysis

The H100 Greennode's GPU KV cache budget is physically bounded; at TP1 with Gemma4-31B-FP8, 8 concurrent agentic sessions each carrying ~67k-token contexts exceed that budget, pushing max KV usage to 99.6%. When the pool is full, every new block evicts a reusable prefix block — this is why the GPU hit rate collapses from 65.9% (conc=2, max KV 62.6%) to 0.45% (conc=8, max KV 99.6%): there are no stable blocks left to hit. The consequence cascades into decode: with no prefix cache reuse, each new request turn must re-prefill the entire conversation history, and those prefill operations compete directly with ongoing decode steps for compute, inflating TPOT from 48ms to 323ms. The 30% throughput drop follows directly — the GPU is doing more total compute per useful output token due to redundant prefill work. The long warmup (~487s) is structural: real MiniMax sessions average ~67k input tokens, and at conc=2 only 2 sessions are active, so early requests have no cache to hit and must fully prefill before the warmup phase ends.

## Recommendations

1. **Compare directly with the paired LMCache run at conc=8.** That run (also on `exp/gemma4-fp8-lmcache-capacity`) should show non-zero external hit rate if the CPU KV offload is working — the baseline here establishes exactly what throughput/TPOT looks like without it. Conc=8 is the critical point where LMCache benefit should be visible.

2. **Consider extending run duration to 1800s for future capacity sweeps.** With 480–507s warmup, the 900s budget leaves only ~400s profiling. Doubling to 1800s would give >1200s of profiling data and make percentile latencies (p99 E2E = 416s at conc=8) more stable.

3. **Establish conc=6 as an additional sweep point.** Conc=4 (KV avg 49.6%) and conc=8 (KV avg 71.4%, saturated) bracket the inflection point but don't pinpoint it. A conc=6 run would show whether the throughput peak is at 4 or 6 and help set the LMCache design target for CPU DRAM offload budget.

4. **If higher concurrency is required without LMCache, consider TP2 (2×H100).** At TP1 the cache ceiling is around conc=4–6 for this model+trace combination. TP2 doubles the KV pool and should support conc=8–12 with healthy hit rates.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28166668983 --repo vngcloud/InferenceX`_
