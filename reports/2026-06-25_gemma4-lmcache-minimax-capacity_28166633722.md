# Benchmark Report: gemma4-lmcache-minimax-capacity

> **Run:** [28166633722](https://github.com/vngcloud/InferenceX/actions/runs/28166633722) | **Branch:** `exp/gemma4-fp8-lmcache-capacity` | **Status:** ✅ success
> **Commit:** `214809a` — feat(gemma4-lmcache): capacity sweep conc [2,4,8] for MiniMax-prod trace
> **Date:** 2026-06-25

## Executive Summary

Capacity sweep for Gemma4-31B-FP8 with LMCache enabled (vLLM TP1, H100 Greennode, LMCache v0.5.0 CPU DRAM offload) against the MiniMax production agentic-replay trace at concurrencies 2, 4, and 8. LMCache is correctly wired and fully operational, but delivers negligible benefit for this workload: external hit rates are 0.53–1.88% across all concurrency levels because the MiniMax trace consists of unique long-context sessions that do not share evictable prefixes across requests. Performance is essentially identical to the paired no-LMCache baseline — the capacity ceiling and KV saturation point (conc=8, 99.6% KV usage, GPU hit rate collapse to 0.86%) are unchanged.

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
| LMCache | `yes (v0.5.0, chunk_size=256, DRAM 5 GB, ZMQ :5555)` |

## Performance

| Metric | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Requests completed | 44 | 49 | 53 |
| Benchmark window | 863.5s | 919.8s | 975.7s |
| Mean TTFT | **12.19s** (p50 0.83s · p99 76.84s) | **24.34s** (p50 6.24s · p99 117.5s) | **42.41s** (p50 35.00s · p99 168.4s) |
| Mean TPOT | 48.0ms (~20.8 tok/s decode) | 145.7ms (~6.9 tok/s decode) | 313.7ms (~3.2 tok/s decode) |
| Total throughput/GPU | 3,435 tok/s | 3,381 tok/s | 2,387 tok/s |
| Input / Output tput/GPU | 3,426 / 9.0 tok/s | 3,370 / 11.1 tok/s | 2,371 / 16.3 tok/s |
| Mean E2E latency | **21.1s** (p99 110.7s) | **53.3s** (p99 179.3s) | **127.6s** (p99 386.3s) |
| Avg ISL / OSL | 67,238 / 177 tokens | 63,251 / 208 tokens | 43,645 / 300 tokens |
| GPU power | 416.1 W (8.25 tok/W) | 593.3 W (5.70 tok/W) | 624.6 W (3.82 tok/W) |

## Cache

| Metric | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| GPU prefix hit rate | **65.91%** (1,951,488 / 2,960,855 tok) | **41.60%** (1,290,272 / 3,101,770 tok) | **0.86%** (416,640 / 48,401,139 tok) |
| External (LMCache) hit rate | 0.53% (5,376 / 1,009,367 tok) | 1.88% (34,048 / 1,811,498 tok) | 0.55% (12,288 / 2,252,884 tok) |
| GPU KV usage (avg / max) | 20.6% / 63.0% | 45.2% / 88.3% | 72.2% / **99.6%** |
| Prompt tokens cached | 1,956,864 | 1,324,320 | 171,488 |

## Stack Initialization

- [✓] LMCache version: `0.5.0` _(need 0.5.0 for hybrid-attention models)_
- [N/A] Attention block size aligned: N/A — pure-attention model (block_size_align not printed by vLLM v0.23.0); `max_num_batched_tokens=8192` confirmed in init log
- [✓] LMCacheMPConnector loaded
- [✓] Heartbeat thread running
- [✓] Hybrid KV manager ON _(hybrid_kv_turned_off not triggered)_
- [✓] No connector crash at startup

## Scheduler Health

| | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Running avg / max | **1.5** / 2 | **2.9** / 4 | **5.6** / 8 |
| Waiting avg / max | **0.0** / 1 | **0.3** / 3 | **1.2** / 7 |
| GPU hit rate (first → last sample) | 0.5% → 60.9% | 1.9% → 36.1% | 14.1% → **0.8%** |
| Log sample count | 104 | 137 | 152 |

The declining GPU hit rate trend at conc=8 (14.1% → 0.8%) mirrors the baseline exactly, confirming LMCache does not alter the KV eviction/preemption dynamics at this saturation level.

## aiperf Phase Summary

| Phase | conc=2 | conc=4 | conc=8 |
|---|---|---|---|
| Warmup elapsed | 485.5s | 478.3s | 506.3s |
| Profiling sent | 63 | 68 | 64 |
| Profiling completed | 63 | 64 | 59 |
| In-flight at end | 0 | 4 | 5 |
| Timeout triggered | yes (normal) | yes (normal) | yes (normal) |
| Errors | 0 | 0 | 0 |

## Anomalies

- **External (LMCache) hit rate near-zero across all concurrencies: 0.53% / 1.88% / 0.55%.** Peak at conc=4 (34k hits out of 1.81M queries). For a 5 GB CPU DRAM budget with 256-token chunk_size, this means fewer than 35k token-chunks were successfully restored from CPU in the entire 920s run at conc=4 — the LMCache layer is wired and counting, but has no useful matches to serve.

- **GPU prefix hit rate collapses at conc=8: 0.86%** — identical to baseline (0.45%). KV max usage hits 99.6%, causing continuous eviction-driven cache thrashing. Prompt tokens cached plummets from 1.96M (conc=2) to 171k (conc=8) despite 20% more requests.

- **TPOT degrades 6.5× from conc=2 → conc=8**: 48ms → 314ms per output token (~20.8 → 3.2 tok/s decode). Identical degradation curve to baseline; LMCache does not improve decode throughput at any concurrency level for this trace.

- **Throughput drops 30% at conc=8**: 3,435 → 2,387 tok/s/GPU. Capacity ceiling is unchanged vs baseline (3,426 → 2,404 tok/s/GPU).

- **Warmup consumes 50–56% of the 900s budget** across all concurrency levels (478–506s), leaving only ~395–422s of profiling time. Structural: real MiniMax sessions average 43–67k input tokens; at low concurrency, early sessions must fully prefill before warmup clears.

- **p99 E2E latency at conc=8: 386s** — the slowest 1% of requests take over 6 minutes end-to-end, making conc=8 unsuitable for interactive agentic workloads.

## Root Cause Analysis

LMCache is fully operational but structurally mismatched to the MiniMax-prod trace at this concurrency range. The CPU DRAM cache can only benefit a session if its KV blocks were previously evicted from GPU *and* are re-requested by the same (or a sharing) session. The MiniMax trace sessions are unique long-context conversations — each session's context is distinct, so an evicted block from session A will not match session B's incoming tokens. This is why external hit rates peak at only 1.88% even at conc=4 where the GPU KV pool reaches 88% max utilization: evictions are happening, but the evicted blocks simply aren't being re-requested. At conc=8, the problem compounds: the KV pool is continuously churning at 99.6% usage, but the 5 GB CPU DRAM budget (≈ chunks for roughly 1–2 full 67k-token sessions at FP8) means that even within-session resume is only possible for a small fraction of evicted context. The identical throughput and TPOT curves between this run and the baseline (within 1–2%) confirm that LMCache is not adding overhead, but also not adding benefit — the connector is healthy and the infrastructure is sound for traces that would actually yield cache sharing.

## Recommendations

1. **Try a shared-prefix workload to validate LMCache benefit.** Switch the dataset from the unique-session MiniMax trace to a synthetic trace with a long shared system prompt (≥16k tokens) and varied completions. This is the canonical LMCache use case and would show external hit rates of 40–80%+ rather than <2%.

2. **Increase LMCACHE_CPU_DRAM_GB from 5 GB to 40–80 GB for session-resume experiments.** At 5 GB and 256-token chunks, the CPU cache can hold context for only ~1 full Gemma4 67k-token session. To enable within-session KV restore across turns, the DRAM budget needs to cover at least `conc × avg_context_size_GB`.

3. **Add `LMCACHE_EVICTION_TYPE=approximate_lru` to the launch script** if not already set. The default eviction policy may not be optimal for long-context sessions; LRU ensures the most recently active session context stays in CPU DRAM.

4. **Run a back-to-back session repeat experiment** to isolate whether LMCache resume works at all. Replay the same session twice in sequence; if LMCache is correctly offloading and restoring, the second run should show ~100% external hit rate and near-zero TTFT. This validates the connector before committing to a larger trace campaign.

5. **Consider conc=6 as the next sweep point** to pinpoint the throughput peak between the healthy conc=4 (KV avg 45%, GPU hit 41.6%) and the saturated conc=8 (KV avg 72%, GPU hit 0.86%). The inflection point likely lies around conc=5–6, and measuring it would set the LMCache design target for how much CPU DRAM offload needs to absorb.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28166633722 --repo vngcloud/InferenceX`_
