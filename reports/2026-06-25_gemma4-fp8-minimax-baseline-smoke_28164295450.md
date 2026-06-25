# Benchmark Report: e2e Test - gemma4-fp8-minimax-baseline-smoke

> **Run:** [28164295450](https://github.com/vngcloud/InferenceX/actions/runs/28164295450) | **Branch:** `exp/gemma4-fp8-minimax-baseline` | **Status:** ✅ success
> **Commit:** `1dbaac3` — feat(agentic-replay): add Gemma4-31B-FP8 baseline smoke (no LMCache) for MiniMax-prod trace
> **Date:** 2026-06-25

## Executive Summary

Gemma4-31B-FP8 on vLLM v0.23.0 was benchmarked against the MiniMax Claude-Code prod trace (mooncake_trace, conc=16, TP=1, H100) with prefix caching enabled but **no LMCache**. The dominant finding is a mean TTFT of 67.2s driven by very long average input sequences (~42k tokens/request) and a near-zero GPU prefix cache hit rate (0.71%), meaning almost every request requires a near-full chunked prefill. KV cache peaked at 96% — the pressure required for LMCache eviction to occur — making this a valid and well-conditioned baseline for the paired LMCache run.

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| Framework / Image | `vllm` / `vllm/vllm-openai:v0.23.0` |
| Precision | `fp8` TP`1` |
| Hardware | `h100-greennode_00` |
| Concurrency | `16` |
| Dataset / Scenario | `minimax_claude_code_prod_v3.jsonl` (`mooncake_trace` / `agentic-replay`) |
| Duration override | `90s` profiling window (207.4s total incl. warmup + grace) |
| LMCache | `no` — baseline; `--enable-prefix-caching` only |

## Performance

| Metric | Value |
|---|---|
| Requests completed | `7` |
| Benchmark window | `207.4s` |
| Mean TTFT | **`67.24s`** (p50 `54.0s` · p99 `121.9s`) |
| Mean TPOT | `706ms` (~`1.4` tok/s decode) |
| Total throughput/GPU | `1,428` tok/s |
| Input / Output tput | `1,424` / `3.6` tok/s per GPU |
| Mean E2E latency | **`147.1s`** (p99 `207.3s`) |
| Avg ISL / OSL | `~42,199` / `~106` tokens per request (measured from 7 completed) |
| GPU power | `623.7W` (`2.29` tok/W) |

## Cache

| Metric | Value |
|---|---|
| GPU prefix hit rate | `0.71%` (`168,960` / `23,670,182` tokens) |
| External (LMCache) hit rate | `N/A` — baseline run, no LMCache wired |
| GPU KV usage (avg / max) | `77.45%` / `96.13%` |
| Prompt tokens cached | `51,712` |

## Stack Initialization

- ✓ vLLM server started cleanly — no crash, no OOM
- ✓ `--enable-prefix-caching` active (GPU prefix cache building throughout run)
- ✓ No connector crash at startup
- N/A LMCache version — baseline run, no `kv-transfer-config`
- N/A Attention block size aligned — no LMCache chunk alignment required
- N/A LMCacheMPConnector — not loaded (expected for baseline)
- N/A Heartbeat thread — not applicable
- N/A Hybrid KV manager — not applicable

## Scheduler Health

Running avg / max: **`6.8`** / `11`
Waiting avg / max: **`6.1`** / `14`
_(from `72` × 10s samples)_

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `506.6s` elapsed |
| Profiling | `18` | `2` | `16` | `true` (duration-based) |
| Grace period | — | `5` added | — | — |

_Errors: `0`_

## Anomalies

- **Mean TTFT 67.2s** (p50 54.0s, p99 121.9s) — ~67× the healthy < 1s target for conc=16 on H100; driven by ~42k-token avg ISL and 0.71% GPU prefix cache hit rate forcing near-full chunked prefill on every request.
- **GPU prefix cache hit rate 0.71%** — effectively zero despite `--enable-prefix-caching`; the MiniMax Claude-Code trace sessions carry unique long-form contexts with negligible cross-session prefix overlap in this cold-start run.
- **Warmup elapsed 506.6s** — 8.4 minutes; 20 warmup requests × ~50–120s TTFT each is arithmetically consistent; warmup did not fail but confirms the server is in a sustained high-TTFT regime.
- **Only 2 profiling requests completed** in the 90s profiling window (16 in-flight at end); 5 more completed in the 120s grace period, giving 7 total. The 90s window is shorter than a single request's E2E latency (mean 147s), so low completion count is expected.
- **KV usage peaked at 96.13%** — near saturation with 16 × ~42k-token concurrent sessions approaching the GPU KV pool limit at TP=1, gpu-memory-utilization=0.90.
- **Output throughput 3.6 tok/s/GPU** — not a decoding bottleneck; avg OSL is only ~106 tokens/request; the benchmark is ~99.8% prefill by token count.

## Root Cause Analysis

The MiniMax Claude-Code prod trace sessions average ~42,199 input tokens each, making this an extreme prefill-bound workload. With `--max-num-batched-tokens=8192` (default in the launcher), vLLM processes each request in ≥6 chunked-prefill steps before generating a single output token, accumulating 67s of TTFT. The GPU prefix cache hit rate of 0.71% confirms that these agentic sessions carry largely unique contexts — no large shared system-prompt or conversation history prefix is being reused across the 7 completed requests — so `--enable-prefix-caching` provides minimal benefit in this cold-start scenario. KV saturation at 96% peak is a direct consequence of 16 × 42k tokens ≈ 672k tokens held simultaneously in the GPU KV pool; this is at the boundary of what TP=1 can hold at 0.90 gpu-memory-utilization. The combination of saturated KV cache and high TTFT creates the ideal pressure condition for LMCache evaluation: blocks are being evicted, and those evictions are the supply-side for the CPU KV offload cache in the paired LMCache run.

## Recommendations

1. **Compare immediately with the LMCache counterpart** (run [28162657488](https://github.com/vngcloud/InferenceX/actions/runs/28162657488)): the key delta metrics are `ext_hit_rate_pct` (should be > 0% if eviction occurs), mean TTFT (should decrease if LMCache re-serves evicted blocks for repeated prefixes), and `kv_usage_max_pct` (should stay below 96% if CPU offload relieves GPU pressure).
2. **Increase `VLLM_MAX_NUM_BATCHED_TOKENS`** from 8192 to 32768 or 65536 to reduce chunked-prefill steps per request from ≥6 to ≤2 — this could cut mean TTFT by ~3× for the same trace without changing the model or hardware. Note: for the LMCache-enabled variant, `mnbt` must stay within `[chunk_size, 2*chunk_size)`.
3. **Reduce concurrency to 8** in a follow-up sweep: at 8 × 42k tokens, KV usage drops to ~50%, reducing eviction pressure and giving the GPU prefix cache more room to retain warm blocks across requests. This isolates whether 0.71% hit rate is a cold-cache artifact or structural to the trace.
4. **Inspect trace prefix diversity** with a subset replay (`#20` suffix to limit to 20 records): if hit rate rises above 10% when records repeat, the near-zero hit rate here is a cold-start artifact; if it stays near 0%, the trace is fundamentally non-repetitive and prefix caching (GPU or LMCache) will not help TTFT regardless of concurrency.
5. **Extend the profiling window** from 90s to 300s+ for this trace: with mean E2E latency of 147s, a 90s profiling window captures < 2 full request lifetimes, making throughput metrics statistically unreliable. A 300s window would yield ~15 completed requests and representative averages.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28164295450 --repo vngcloud/InferenceX`_
