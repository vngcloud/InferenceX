# Agentic Replay Datasets

These JSONL files are Mooncake-compatible traces used by the `agentic-replay`
scenario. Use them with `custom-dataset-type: mooncake_trace` and the AIPerf
benchmark client.

## Files

| File | Records | Sessions | Notes | Recommended `max-model-len` |
|---|---:|---:|---|---:|
| `qwen3.5-4b-smoke.jsonl` | 12 | 5 | Tiny plumbing smoke trace | 8192 |
| `agentic-coding-64k.jsonl` | 18,595 | 1,000 | Single L1 prefix baseline | 73728 |
| `agentic-coding-128k.jsonl` | 16,957 | 1,000 | Single L1 prefix baseline | 147456 |
| `agentic-coding-64k-5variants.jsonl` | 18,554 | 1,000 | Five L1 prefix variants, Zipf alpha 1.2 | 73728 |
| `agentic-coding-128k-5variants.jsonl` | 16,902 | 1,000 | Five L1 prefix variants, Zipf alpha 1.2 | 147456 |
| `agentic-coding-64k-5variants-top150-long-context.jsonl` | 4,238 | 150 | Complete-session top-150 subset by max session-cumulative context | 73728 |
| `agentic-coding-128k-5variants-top150-long-context.jsonl` | 3,897 | 150 | Complete-session top-150 subset by max session-cumulative context | 147456 |
| `agentic-coding-128k-5variants-config300s-seed42-20260605-120047.jsonl` | 5,167 | 300 | Full 300-session 128k config from `config_300s_seed42_20260605-120047` | 147456 |

## `agentic-coding-64k-5variants.jsonl`

Generated from the AIPerf agentic-code synthesizer with seed 42. Compared with
`agentic-coding-64k.jsonl`, this trace assigns each session to one of five L1
prefix variants so cache reuse only happens within the same tenant-style prefix.

Turn-0 variant distribution:

| Variant | Sessions |
|---:|---:|
| 0 | 471 |
| 1 | 230 |
| 2 | 135 |
| 3 | 91 |
| 4 | 73 |

Context sizing must use the session-cumulative prompt, not the per-row
`input_length`. For this dataset, the max per-row `input_length + output_length`
is 38,264 tokens, but the max session-cumulative `input + output` is 66,680
tokens. Use `max-model-len: 73728` for 64k replay jobs.

For Mode 1 capacity smokes, keep the trace delay stripping enabled and set an
explicit `request-count`; the `#N` suffix on `input-file` can be used to limit
the loaded trace rows while AIPerf resamples sessions up to `request-count`.

## `agentic-coding-128k-5variants.jsonl`

Generated from the AIPerf agentic-code synthesizer with seed 42 using the
same five L1 variant layout as the 64k 5-variant trace. This dataset targets
long-context coding-agent sessions with `max_prompt_tokens: 131072`.

Turn-0 variant distribution:

| Variant | Sessions |
|---:|---:|
| 0 | 488 |
| 1 | 209 |
| 2 | 144 |
| 3 | 86 |
| 4 | 73 |

For this dataset, the max per-row `input_length + output_length` is 82,784
tokens and the max session-cumulative `input + output` is 133,413 tokens. Use
`max-model-len: 147456` for 128k replay jobs.

## Top-150 long-context subsets

The `*-top150-long-context.jsonl` files preserve complete sessions from the
corresponding 5-variant traces, selecting the 150 sessions with the highest max
session-cumulative request context. Use `request-count` equal to the record count
to replay each subset once without resampling.

The 64k subset has 4,238 records. Its max session-cumulative request context is
65,532 tokens, and 72 records cross 65,536 after adding output tokens.

The 128k subset has 3,897 records. Its max session-cumulative request context is
131,068 tokens, and 63 records cross 131,072 after adding output tokens.

## 300-session 128k config

`agentic-coding-128k-5variants-config300s-seed42-20260605-120047.jsonl` is a
full replay of the generated 300-session 128k workload from
`aiperf-service-docs/workloads/agentic-coding-5-variants/128k/config_300s_seed42_20260605-120047`.
Do not set `request-count` when replaying this file once; the launcher counts
all 5,167 records and passes that to AIPerf.
