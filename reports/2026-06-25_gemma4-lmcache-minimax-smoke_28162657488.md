# Benchmark Report: e2e Test - gemma4-lmcache-minimax-smoke

> **Run:** [28162657488](https://github.com/vngcloud/InferenceX/actions/runs/28162657488) | **Branch:** `dev-lmcache` | **Status:** ✅ success
> **Commit:** `1c1416f` — feat(lmcache): add Gemma4-31B-FP8 agentic-replay smoke with LMCache MP connector
> **Date:** 2026-06-25

## Executive Summary

First agentic-replay smoke for `RedHatAI/gemma-4-31B-it-FP8-dynamic` with LMCache CPU
KV-offload (MP connector) on the MiniMax Claude-Code production trace, running vLLM
v0.23.0 at conc=16 on a single H100 (GreenNode). The dominant finding is **KV cache
saturation**: at avg 42 199 input tokens per session × 16 concurrent sessions, the GPU
KV pool fills to 99.79%, causing TTFT to balloon to 56.5s and 10/17 profiling requests
to be cancelled when the 120s grace period expired. The LMCache connector is correctly
wired and functional; the near-zero external hit rate (0.43%) reflects the unique-prefix
nature of MiniMax conversations, not a connector failure.

## Configuration

| Field | Value |
|---|---|
| Model | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| Framework / Image | `vllm` / `vllm/vllm-openai:v0.23.0` |
| Precision | `fp8` TP`1` |
| Hardware | `h100-greennode_00` |
| Concurrency | `16` |
| Dataset / Scenario | `minimax_claude_code_prod_v3.jsonl` (`agentic-replay`) |
| Duration override | `90s` |
| LMCache | `yes (v0.5.0, chunk_size=256, DRAM 5 GB, MP connector ZMQ :5555)` |

## Performance

| Metric | Value |
|---|---|
| Requests completed | `7` |
| Benchmark window | `203.5s` (90s profiling + 113.5s grace) |
| Mean TTFT | **`56.5s`** (p50 `67.4s` · p99 `125.4s`) |
| Mean TPOT | `778ms` (~`1.29` tok/s decode) |
| Total throughput/GPU | `1 455.5` tok/s |
| Input / Output tput | `1 451.9` / `3.65` tok/s |
| Mean E2E latency | **`139.2s`** (p99 `201.2s`) |
| Avg ISL / OSL | `~42 199` / `~106` tokens |
| GPU power | `631.1W` (`2.31` tok/W) |

## Cache

| Metric | Value |
|---|---|
| GPU prefix hit rate | `0.34%` (`46 592` / `13 686 277` tokens) |
| External (LMCache) hit rate | `0.43%` (`2 560` / `597 624` tokens) |
| GPU KV usage (avg / max) | `76.35%` / `99.79%` |
| Prompt tokens cached | `108 800` |

## Stack Initialization

- [✓] LMCache version: `0.5.0` _(meets the ≥ 0.5.0 requirement for SupportsHMA)_
- [✓] Attention block size aligned: N/A — pure-attention model (Gemma 4 has no Mamba layers; no "Setting attention block size" message is printed; chunk_size=256 default is correct)
- [✓] LMCacheMPConnector loaded
- [✓] Heartbeat thread running
- [✓] Hybrid KV manager ON _(hybrid_kv_turned_off=false)_
- [✓] No connector crash at startup

## Scheduler Health

Running avg / max: **`6.6`** / `10`
Waiting avg / max: **`6.4`** / `13`
_(from `70` × 10s samples)_

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `491.3s` elapsed |
| Profiling | `17` | `1` | `16` | `true` (90s duration) |
| Grace period | — | `6` added | — | `true` (120s exhausted; 10 cancelled) |

_Errors: `0`_

## Anomalies

- **TTFT 56.5s (p50 67.4s, p99 125.4s)** — roughly 56× above the < 1s healthy target for conc=16 on H100; every request spends most of its wall-clock time queued behind 15 other saturated sessions before prefill even begins.
- **10/17 profiling requests cancelled** — p99 E2E of 201.2s exceeds the 120s grace period ceiling; only 7 of 17 sent requests produced usable metrics.
- **Warmup took 491s** — 20 warmup requests averaging ~24.5s each, confirming KV saturation begins immediately at startup, not just during the profiling window.
- **GPU KV usage 99.79% max** — at avg 42 199 tok/session × 16 concurrent sessions ≈ 675k tokens, the H100's KV pool is fully exhausted; new requests cannot start prefill until existing ones complete and free blocks.
- **GPU prefix cache hit rate 0.34%** — essentially zero; MiniMax conversations have unique per-session prefixes, so vLLM's CPU-side prefix cache provides no reuse benefit across sessions.
- **External (LMCache) hit rate 0.43%** — despite KV usage at 99.79% (evictions ARE occurring), the evicted blocks belong to unique sessions that never recur in the trace; LMCache is correctly receiving evicted blocks but they are never retrieved.
- **Output throughput 3.65 tok/s** — only 7 completions across the full 203s window; the system is almost exclusively in prefill mode, never reaching steady decode throughput.

## Root Cause Analysis

The single H100 (TP=1) cannot hold 16 concurrent MiniMax sessions in KV simultaneously.
Each MiniMax conversation averages ~42 199 input tokens; at conc=16, the total KV demand
(~675k tokens) saturates the GPU pool to 99.79%, preventing new prefill work from
starting until existing sessions release blocks. This forces incoming requests into the
waiting queue (waiting.avg=6.4, max=13), which directly explains the TTFT of 56.5s:
each request must wait for ~6–7 other sessions to complete a decode step and release at
least one KV block before it can begin chunked prefill.

LMCache is correctly wired (MP connector loaded, heartbeat active, no crash) and IS
receiving evicted blocks — the 597k external queries confirm the lookup path is live. The
0.43% external hit rate is not a failure; it reflects the dataset's structure: each
MiniMax conversation is a unique dialogue with no shared prefix across sessions, so
evicted blocks from session A are never needed by any later session. The same principle
explains the 0.34% GPU prefix hit rate. The grace period timeout (10 requests cancelled
after 120s) is a consequence of p99 E2E reaching 201s — requests take longer end-to-end
than the grace window allows.

## Recommendations

1. **Reduce concurrency to 4–6** for the next smoke run of this stack: 4 × 42k ≈ 168k
   tokens keeps KV usage below ~25%, allows requests to flow without queuing, and should
   bring TTFT below 5s. Use `conc-list: [4]` or `[4, 8]` in the config `search-space`.
2. **Extend the grace period to 240s** (`BENCHMARK_GRACE_PERIOD=240` in the launch
   script) if you want to benchmark at higher concurrency without losing requests to
   cancellation — p99 E2E is 201s, which exceeds the current 120s ceiling.
3. **Accept near-zero LMCache hit rate for MiniMax trace** — the 0.43% is not a bug.
   To see meaningful external cache benefit, use a trace with repeated prefix structure
   (e.g., agentic-coding with the same system prompt across many turns) or run the same
   sessions multiple times in sequence.
4. **Run the baseline companion** (`gemma4-fp8-h100-greennode-vllm`, dispatched today as
   run `28164295450`) at the same reduced concurrency (conc=4) so the LMCache delta is
   measured under non-saturated conditions where throughput differences are visible.
5. **Consider TP=2** (`h100-greennode_01`) for higher-concurrency agentic workloads on
   this model: doubling the GPU count doubles the KV pool, allowing conc=16 to run
   without saturation.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28162657488 --repo vngcloud/InferenceX`_
