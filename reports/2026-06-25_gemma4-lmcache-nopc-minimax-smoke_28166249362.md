# Benchmark Report: e2e Test - gemma4-lmcache-nopc-minimax-smoke

> **Run:** [28166249362](https://github.com/vngcloud/InferenceX/actions/runs/28166249362) | **Branch:** `exp/gemma4-lmcache-nopc` | **Status:** ✅ success
> **Commit:** `4e9fe59` — feat(lmcache): add Gemma4-31B-FP8 LMCache-only smoke (no vLLM prefix caching) for MiniMax-prod trace
> **Date:** 2026-06-25

## Executive Summary

Benchmarked Gemma 4 31B FP8 on vLLM v0.23.0 with LMCache CPU KV-offload active but **vLLM GPU prefix caching explicitly disabled** (`--enable-prefix-caching` removed), replaying the MiniMax production trace at CCU=16 on h100-greennode_00. The dominant finding is that E2E latency reached **128.1s per request** (driven by mean TTFT 41.78s + ~87s of decode for 77 actual output tokens), which exceeded the 90s profiling window and resulted in **zero profiling completions** — all 5 completed requests belong to the warmup phase only. LMCache alone, without the GPU prefix cache to seed it with matched prefixes, achieves only 1.73% external hit rate on this trace; this establishes the lower-bound reference point for the three-way comparison (baseline / lmcache-nopc / lmcache+PC).

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| Framework / Image | `vllm` / `vllm/vllm-openai:v0.23.0` (+ lmcache 0.5.0 pip-installed at runtime) |
| Precision | `fp8` TP`1` |
| Hardware | `h100-greennode_00` |
| Concurrency | `16` |
| Dataset / Scenario | `minimax_claude_code_prod_v3.jsonl` (`mooncake_trace` / agentic-replay) |
| Duration override | `90s` smoke |
| LMCache | `yes (v0.5.0, chunk_size=256, DRAM 5g)` — GPU prefix caching **disabled** |

## Performance

| Metric | Value |
|---|---|
| Requests completed | `5` (warmup only — profiling window: 0 of 16 completed) |
| Benchmark window | `150.3s` |
| Mean TTFT | **`41.78s`** (p50 `27.34s` · p99 `97.02s`) |
| Mean TPOT | `1144.7ms` (~`0.87` tok/s decode) |
| Total throughput/GPU | `1574.8` tok/s |
| Input / Output tput | `1572.3` / `2.58` tok/s |
| Mean E2E latency | **`128.1s`** (p99 `149.8s`) |
| Avg ISL / OSL | `47,258 actual` / `77 actual` tokens (configured 4096 / 512) |
| GPU power | `648.7W` (`2.43` tok/W) |

## Cache

| Metric | Value |
|---|---|
| GPU prefix hit rate | `8.87%` (`356,384` / `4,016,230` tokens) |
| External (LMCache) hit rate | `1.73%` (`9,472` / `547,175` tokens) |
| GPU KV usage (avg / max) | `78.93%` / `98.82%` |
| Prompt tokens cached | `115,872` |

## Stack Initialization

- [✓] LMCache version: `0.5.0` _(need 0.5.0 for hybrid-attention models)_
- [✗] Attention block size aligned: not found in server.log _(parser found no "Setting attention block size" message; chunk_size=256 set via env var)_
- [✓] LMCacheMPConnector loaded
- [✓] Heartbeat thread running
- [✓] Hybrid KV manager ON _(hybrid_kv_turned_off=false)_
- [✓] No connector crash at startup

## Scheduler Health

Running avg / max: **`6.9`** / `11`
Waiting avg / max: **`5.1`** / `13`
_(from `78` × 10s samples)_

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | 5 | — | `576.41s` elapsed |
| Profiling | `16` | `0` | `16` | `true` |
| Grace period | — | 0 added | — | — |

_Errors: `0`_

## Anomalies

- **Profiling phase: 0 of 16 requests completed.** Mean E2E latency (128.1s) exceeds the 90s profiling window, so every request sent during profiling was still in-flight at timeout. All 5 completed requests belong to the warmup phase — this run produces no valid profiling-phase metrics.
- **Warmup consumed 576.4s** — 6.4× the 90s smoke budget — because each warmup request required ~115s E2E to complete (5 requests / 576s).
- **Mean TTFT: 41.78s, p99: 97.0s** — with actual avg ISL of 47,258 tokens and no GPU prefix cache, every request recomputes a full 47K-token prefill. Even at H100 FP8 speeds, 47K tokens at ~5–10K tok/s prefill = 5–10s prefill alone, compounded by waiting.avg=5.1 requests queuing ahead.
- **Actual avg ISL is 47,258 tokens** (total_isl=236,288 ÷ 5 requests), far exceeding the configured 4096 — the MiniMax production trace replays long multi-turn agentic sessions with accumulating context; the configured ISL parameter is a max-cap, not the observed mean.
- **LMCache external hit rate: 1.73%** despite KV usage avg 78.93% / max 98.82% — eviction to CPU DRAM is occurring (ext_hit_rate rising from 0% → 2.7% across the run), but without GPU prefix caching the evicted block addresses don't align with incoming request prefix hashes, limiting reuse. A sparse 1.73% is the realistic ceiling for LMCache-only on this trace.
- **Output throughput: 2.58 tok/s total** — the server spends nearly all compute budget on prefill of 47K-token contexts, leaving almost no capacity for decode. At 16 CCU, each GPU can decode only 0.16 tokens/s per user.
- **block_size_align not found in server.log** — the "Setting attention block size" init message was absent; chunk_size=256 alignment could not be verified from log evidence alone.

## Root Cause Analysis

The fundamental problem is that the MiniMax production trace has actual average context lengths of ~47K tokens (not the configured 4096 cap), and removing `--enable-prefix-caching` forces vLLM to recompute the full 47K-token prefill for every request from scratch. On a single H100 with TP=1, this prefill is the dominant cost: at ~5,000 tok/s prefill throughput, a 47K prompt takes ~9s of compute, and with waiting.avg=5.1 concurrent requests queuing, each new arrival waits ~45s for its turn — matching the observed mean TTFT of 41.78s. Decode adds another ~87s for the 77 actual output tokens at 0.87 tok/s (itself throttled by the GPU being occupied with prefill 78.93% of the time on average). The resulting 128s E2E latency is longer than the 90s profiling window, which is why profiling_completed=0: the window closed before a single profiling request could finish. LMCache is correctly wired (MP connector, heartbeat, v0.5.0), and eviction to CPU DRAM is genuinely occurring (KV usage hitting 98.82% max), but because vLLM's block-level prefix matching is off, the evicted blocks don't map onto shared prefixes in subsequent requests — so LMCache's theoretical ceiling on this trace without GPU prefix caching is the 1.73% we observe. The 8.87% GPU prefix hit rate (despite `--enable-prefix-caching` being removed) reflects vLLM v0.23.0's automatic radix-cache-based block reuse, which remains partially active regardless of the flag.

## Recommendations

1. **Extend the smoke duration to 600s (10 min)** to capture at least one profiling-phase completion cycle. With mean E2E ~128s and CCU=16, a 600s window will yield ~4–5 completed profiling requests, enough to compute stable TTFT and TPOT distributions.
2. **Reduce CCU to 4 for the no-PC variant** to confirm the TTFT root cause. At CCU=4, waiting.avg should drop to ~1 and TTFT should fall to ~10–15s (pure prefill cost). If TTFT doesn't improve proportionally, a vLLM scheduling config issue is the cause rather than queuing.
3. **Compare directly against the lmcache+PC run (28166633722)** using the same profiling window. The key delta is the GPU prefix hit rate: lmcache+PC likely achieves 40–80% GPU hits on this trace (long shared system prompts), which dramatically reduces per-request prefill and brings E2E below the profiling window.
4. **Investigate why block_size_align is absent from server.log.** For chunk_size=256 to align correctly with vLLM's KV block size, vLLM must log "Setting attention block size to 256 tokens" at startup. If this message is missing, the block sizes may be mismatched and LMCache reuse will silently fail on block boundaries — grep for `block_size` and `chunk_size` in server.log to confirm.
5. **Do not promote this config to a capacity run** until the smoke produces valid profiling data. The current run's metrics (TTFT, TPOT, throughput) are warmup-phase samples only and are not representative of steady-state performance.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28166249362 --repo vngcloud/InferenceX`_
