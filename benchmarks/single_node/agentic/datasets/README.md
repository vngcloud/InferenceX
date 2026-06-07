# Agentic Replay Datasets

These JSONL files are Mooncake-compatible traces used by the `agentic-replay`
scenario. Use them with `custom-dataset-type: mooncake_trace` and the AIPerf
benchmark client.

## Files

| File | Records | Sessions | Notes | Recommended `max-model-len` |
|---|---:|---:|---|---:|
| `qwen3.5-4b-smoke.jsonl` | 12 | 5 | Tiny plumbing smoke trace | 8192 |
| `agentic-coding-64k-5variants-config150s-seed42-20260605-131906.jsonl` | 2,821 | 150 | 150-session 64k five-L1-variant config150s trace | 73728 |
| `agentic-coding-128k-5variants-config150s-seed42-20260605-131909.jsonl` | 2,716 | 150 | 150-session 128k five-L1-variant config150s trace | 147456 |
| `agentic-coding-128k-5variants-config300s-seed42-20260605-120047.jsonl` | 5,167 | 300 | Full 300-session 128k config from `config_300s_seed42_20260605-120047` | 147456 |
| `agentic-coding-64k-1l1variant-config150s-seed42-20260605-155033.jsonl` | 2,913 | 150 | Final agentic-coding 64k config with one L1 prefix variant | 147456 |
| `agentic-coding-128k-1l1variant-config150s-seed42-20260605-155045.jsonl` | 2,732 | 150 | Final agentic-coding 128k config with one L1 prefix variant | 147456 |
| `agentic-coding-167k-1l1variant-config150s-seed42-20260606-131503.jsonl` | 2,256 | 150 | Final agentic-coding 167k config with one L1 prefix variant (max session-cumulative 167,899) | 184320 |
| `agentic-coding-167k-1l1variant-config150s-seed42-20260607-040447.jsonl` | 1,603 | 155 | Regenerated 167k one-L1-variant config, saner length/context distribution (seed42 20260607, max session-cumulative 168,376) | 184320 |
| `agentic-coding-167k-5variants-config150s-seed42-20260607-040451.jsonl` | 1,604 | 167 | Regenerated 167k five-L1-variant config (seed42 20260607, max session-cumulative 174,163) | 184320 |

## Five-L1-variant config traces

The `*-5variants-config*` files assign each session to one of five L1 prefix
variants (Zipf alpha 1.2) so cache reuse only happens within the same
tenant-style prefix — a more conservative prefix/KV-reuse case than the
single-L1-variant (`1l1`) traces. Context sizing must use the session-cumulative
prompt, not the per-row `input_length`: use `max-model-len: 73728` for 64k
replay jobs and `147456` for 128k.

For Mode 1 capacity smokes, keep the trace delay stripping enabled and set an
explicit `request-count`; the `#N` suffix on `input-file` can be used to limit
the loaded trace rows while AIPerf resamples sessions up to `request-count`.

## 300-session 128k config

`agentic-coding-128k-5variants-config300s-seed42-20260605-120047.jsonl` is a
full replay of the generated 300-session 128k workload from
`aiperf-service-docs/workloads/agentic-coding-5-variants/128k/config_300s_seed42_20260605-120047`.
Do not set `request-count` when replaying this file once; the launcher counts
all 5,167 records and passes that to AIPerf.

## 150-session one-L1-variant configs

The `*-1l1variant-config150s-*` files are generated from
`aiperf-service-docs/workloads/final_agentic_coding` with `num_sessions: 150`
and `layer1_variants.num_variants: 1`. They are intended for CCU <= 32 replay
jobs where the 5-variant traces are broader than needed.

Do not set `request-count` when replaying these files once; the launcher counts
all records and passes that value to AIPerf. The 64k file has 2,913 records and
the 128k file has 2,732 records.
