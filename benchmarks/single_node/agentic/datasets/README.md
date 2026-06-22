# Agentic Replay Datasets

These JSONL files are Mooncake-compatible traces used by the `agentic-replay`
scenario. Use them with `custom-dataset-type: mooncake_trace` and the AIPerf
benchmark client.

## Files

| File | Records | Sessions | Notes | Recommended `max-model-len` |
|---|---:|---:|---|---:|
| `minimax_claude_code_prod_v3.jsonl` | 19,662 | 5,626 | Production-derived MiniMax claude-code trace (turn-granular, interleaved). Per-turn input up to ~181k tok; per-session cumulative up to ~186k. | 131072 (smoke); higher for full fidelity |

## Replay modes

**Duration-based smoke (current Qwen3-4B gate).** Point `input-file` at the full
trace, set `max-model-len: 131072`, and bound the run with a dispatch
`duration-override` (e.g. 90s) — no filtering, no `#N`, no `request-count`. The
launcher passes `--benchmark-duration` to the adapter, which then **skips** exact
request-count validation. Turns whose context exceeds `max-model-len` are rejected
by the engine and recorded as errored requests; this is expected under duration
mode and does not fail the run. This validates the mooncake → adapter →
raw-artifact-upload plumbing, not real capacity numbers.

**Single-replay (no duration).** Omit the duration override and the launcher
counts all records and passes that as `--request-count`; every request must then
succeed or the adapter refuses to aggregate. Only safe when every turn fits the
context window.

> The trace is **turn-granular and interleaved** — sessions do not appear
> contiguously and long sessions sort first, so a `head -n N` (`#N`) prefix does
> **not** yield a short-session subset. Use duration bounding (above) rather than
> `#N` to cap a smoke against this trace.
