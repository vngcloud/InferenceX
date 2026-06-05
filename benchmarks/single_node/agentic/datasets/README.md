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
