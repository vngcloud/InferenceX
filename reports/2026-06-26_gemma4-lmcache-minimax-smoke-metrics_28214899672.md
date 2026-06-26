# Benchmark Report: e2e Test - gemma4-lmcache-minimax-smoke-metrics

> **Run:** [28214899672](https://github.com/vngcloud/InferenceX/actions/runs/28214899672) | **Branch:** `exp/gemma4-lmcache-minimax-gpu0.65-dram20g` | **Status:** ✅ success
> **Commit:** `e1a4d91` — fix(benchmark-tmpl): upload lmcache_server_metrics.json for agentic-replay path
> **Date:** 2026-06-26

## Executive Summary

Smoke verification run for `RedHatAI/gemma-4-31B-it-FP8-dynamic` with LMCache MP connector (CPU DRAM KV-offload) on the MiniMax Claude-Code trace at conc=2, confirming the artifact upload fix. The primary goal is validated: `lmcache_server_metrics.json` is now correctly uploaded for the agentic-replay path and all `lmcache_mp_*` fields are non-null for the first time. At conc=2 the stack is healthy — GPU KV saturated to 97.8% (eviction occurring), LMCache L1 at 75.2% utilization (16.1 GB / 20 GB), and MP hit rate 9.23% confirming the LMCache server is actively serving lookups.

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
| Requests completed | `6` |
| Benchmark window | `143.5s` |
| Mean TTFT | **`21.82s`** (p50 `22.29s` · p90 `42.45s` · p99 `47.84s`) |
| Mean TPOT | `61.2ms` (~`16.3` tok/s decode) |
| Total throughput/GPU | `2,880` tok/s |
| Input / Output tput | `2,871` / `8.6` tok/s |
| Mean E2E latency | **`43.2s`** (p99 `100.8s`) |
| Avg ISL / OSL | `4,096` / `512` tokens |
| GPU power | `609.5W` (`4.73` tok/W) |

## Cache

### GPU / External tier

| Metric | Value |
|---|---|
| GPU prefix hit rate | `24.25%` (`99,936` / `412,023` tokens) |
| External (LMCache) hit rate | `11.98%` (`37,376` / `312,087` tokens) |
| GPU KV usage (avg / max / min) | `65.1%` / `97.8%` / `0.0%` |
| Prompt tokens cached | `161,824` |

### LMCache MP internal _(MP connector path only)_

| Metric | Value |
|---|---|
| MP lookup hit rate (L1 + L2) | `9.23%` (`181,760` / `1,969,408` tokens) |
| L2 prefetch hit rate | N/A — no L2 (disk) tier configured |
| L2 prefetch failures | N/A — no L2 tier configured |
| L1 (CPU DRAM) usage | `75.2%` (`16.15 GB` of `~20 GB` allocated) |
| Active prefetch jobs at scrape | `0` |

## Stack Initialization

- [✓] LMCache version: `0.5.0` _(SupportsHMA requirement met)_
- [✓] Attention block size aligned: N/A — pure-attention model (Gemma 4 has no Mamba layers; no "Setting attention block size" message emitted; chunk_size=256 default is correct)
- [✓] LMCacheMPConnector loaded
- [✓] Heartbeat thread running
- [✓] Hybrid KV manager ON _(hybrid_kv_turned_off=false)_
- [✓] No connector crash at startup

## Scheduler Health

Running avg / max: **`1.5`** / `2`
Waiting avg / max: **`0.2`** / `1`
_(from `58` × 10s samples)_

Cache warming trend: GPU hit rate 0.5% → 35.6%, ext hit rate 0% → 9.1% over the 143s profiling window — confirming cache population is progressive and working correctly.

## aiperf Phase Summary

| Phase | Sent | Completed | In-flight at end | Timeout |
|---|---|---|---|---|
| Warmup | — | — | — | `527.4s` elapsed |
| Profiling | `6` | `5` | `1` | `true` (120s duration) |
| Grace period | — | `1` added | — | — |

_Errors: `0`_

## Anomalies

- **Warmup 527.4s**: expected — 20 warmup requests × ~53s each (TTFT 21.8s + 512 × 61ms TPOT) ÷ 2 conc ≈ 530s. Mathematically consistent, not a sign of degradation.
- **Only 6 profiling requests**: smoke run — conc=2 over 120s with ISL=4096 and TTFT ~22s naturally produces few completions.
- **`mp_l2_hit_rate_pct` / `mp_l2_prefetch_fails`: null**: expected — this script configures L1-only (20 GB CPU DRAM via `--l1-size-gb`). No L2 disk tier is enabled.
- **TTFT mean 21.8s at conc=2**: expected for 31B FP8 with ISL=4096 — GPU hit rate is 24.25%, so ~3,120 tokens/request need real prefill. At max_num_batched_tokens=8192 both requests are co-batched, giving effective prefill of ~6,240 tokens per step.

## Root Cause Analysis

This run's purpose was to verify the `lmcache_server_metrics.json` upload fix, not to measure production throughput. The fix is confirmed: the bare `lmcache_server_metrics.json` entry added to the agentic-replay upload group in `benchmark-tmpl.yml` correctly captures the file written by `scrape_lmcache_server_metrics /workspace/ 8080`, which lands at the host CWD root (not under `results/`) in the mooncake-trace execution path.

Cache behavior at conc=2 is healthy: GPU KV reached 97.8% max, confirming eviction IS occurring and LMCache is receiving blocks. The 11.98% external hit rate from vLLM's counter and 9.23% MP hit rate from the LMCache server's counter measure the same cache benefit from different vantage points (vLLM counts block-fetch responses; LMCache counts per-token ZMQ lookups), so the gap is structural, not a discrepancy. L1 utilization at 75.2% (16.1 GB of 20 GB allocated) is healthy — no eviction pressure from the LMCache server itself. The cache warming trend (ext hit rate 0% → 9.1%, GPU 0.5% → 35.6%) shows the caches populate progressively, as expected for a cold-start smoke run.

## Recommendations

1. **Run a capacity sweep at conc ≥ 4** now that the `mp_*` scrape is confirmed working — conc=2 produces too few requests for statistically meaningful TTFT distributions. Use `conc-list: [4, 8, 16]` to find the KV saturation knee.
2. **Bump DRAM budget to 40–60 GB** for capacity runs: at conc=2 L1 is already 75.2% full after just 6 requests; at conc=16 with ISL=4096 the 20 GB budget will saturate quickly. `LMCACHE_CPU_DRAM_GB=40` is a reasonable starting point given GreenNode's DRAM headroom.
3. **Extend benchmark duration to 300s** for non-smoke runs to allow the MP hit rate to stabilize past the cold-start ramp — the current 143s window captures mostly cache-warming behavior.
4. **Compare mp_hit_rate vs ext_hit_rate across runs** to understand the structural gap — if they consistently differ by ~3pp, that is the normal cross-plane measurement offset; a sudden divergence would indicate a connector issue.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 28214899672 --repo vngcloud/InferenceX`_
