# Agentic Replay Datasets

Datasets in this directory are used by the `agentic-replay` scenario with the
AIPerf benchmark client. The dataset format determines which AIPerf source tree
the launch script must install from.

## Active Datasets

| Path | Type | AIPerf source | Shape | Notes |
|---|---|---|---:|---|
| `agentic_coding_1variant_64k_150s.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | 64k tier | Integrated agentic-coding trace. Other tiers must be added here before use. |
| `gemma_blend_prod.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | blend_prod | Back-to-back replay; use `strip-trace-delays: true`. |
| `minimax_cc_v4_weka/` | `weka_trace` | `utils/aiperf-mooncake` | 223 files, 17,672 requests | MiniMax Claude Code v4 Weka traces. Use a directory `input-file`, `no-fixed-schedule: true`, and capped inter-turn delays. |

## Archived Datasets

| Path | Type | Status | Notes |
|---|---|---|---|
| `minimax_claude_code_prod_v3.jsonl` | `mooncake_trace` | outdated | Kept for reference only. Do not use for new MiniMax Claude Code benchmarking unless the old v3 trace is explicitly requested. |

## MiniMax Claude Code v4 Weka

The v4 Weka corpus is the active MiniMax Claude Code replay dataset. It was
filtered to remove no-op rows where `in=0,out=0`; the remaining corpus has:

| Metric | Value |
|---|---:|
| Trace files | 223 |
| Requests | 17,672 |
| Input tokens | 1,358,286,289 |
| Output tokens | 5,462,757 |

Smoke status: validated on `h200-greennode_01` with
`Qwen/Qwen3-4B-Instruct-2507`, vLLM bf16, `TP=1`, `conc=2`, and `duration=90`.
Use `benchmarks/single_node/qwen3-4b-v4-weka_bf16_h200_vllm.sh` and the matching
`qwen3-4b-v4-weka` config entry as the smoke template.

## Replay Notes

Use duration-bounded smoke runs for first validation. For a 90s smoke, keep
warmup at 2 requests so the warmup phase does not consume the profiling window.

For both `mooncake_trace` and `weka_trace` datasets, set `AIPERF_SOURCE_DIR` to
`utils/aiperf-mooncake` (thangquang09 fork, branch `benchtool/agentx-weka`). It
carries both loaders and the SGLang NaN fix; the old `utils/aiperf` (vngcloud
fork) lacks the NaN fix for `weka_trace` and is no longer used here.
