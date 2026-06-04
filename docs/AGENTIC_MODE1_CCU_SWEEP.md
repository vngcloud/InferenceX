# Agentic-Replay Mode 1 — Concurrency (CCU) Capacity Sweep

This is the operating guide for the **Mode 1 capacity sweep** of the
`agentic-replay` path: replaying a Mooncake coding-agent trace under pure
`--concurrency` back-pressure (zero think-time) and stepping concurrency up to
find the throughput/latency knee. It is the **main report result** for agentic
benchmarks.

For the broader AIPerf client integration and the fixed-schedule single-replay
behavior, see [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md); for the decision
to ride official AIPerf, see
[`adr/0001-agentic-on-official-aiperf.md`](adr/0001-agentic-on-official-aiperf.md).

## Status — validated end-to-end (2026-06-04)

The smoke configuration `qwen3-4b-2507-bf16-h100-vllm-mode1-smoke` passed a full
CCU sweep on a GitHub Actions dispatch:

- Run [**26933973732**](https://github.com/vngcloud/InferenceX/actions/runs/26933973732),
  `conclusion: success`. All three concurrency legs (8, 16, 32) plus
  `collect-results` and `calc-success-rate` succeeded; every other scenario type
  was correctly skipped (config-key isolation worked — no `conc=2` single-replay
  or 64k legs leaked in).
- Each leg completed **50/50 requests, `error_request_count == 0`** — the adapter's
  fail-closed gate (`request_count == expected && errors == 0`) passed on all three.
- The matrix expanded to exactly **3 jobs** (one per concurrency); Mode 1 needs
  **no new generator code** — the existing `conc-list` already produces one job
  per concurrency, and each job runs AIPerf at a single fixed concurrency.

This proves the Mode 1 plumbing (config fields → matrix → workflow env → launcher
→ `benchmark_lib.sh` → `aiperf_adapter.py` → `aiperf profile`) works end-to-end.

## What Mode 1 is (vs the default single-replay)

| | Default agentic-replay | **Mode 1 capacity sweep** |
|---|---|---|
| Timing | Fixed-schedule; honors trace `timestamp`/`delay` | **`--no-fixed-schedule`** — pure concurrency back-pressure |
| Think-time | Per-turn `delay` replayed | **Stripped** (`strip-trace-delays`) → zero think-time |
| `--request-count` | = dataset record count (replayed once) | Explicit; AIPerf **resamples** sessions to reach it |
| Concurrency | A ceiling on in-flight sessions | The **swept variable** (step until SLA breaks) |
| Purpose | Reproduce recorded traffic shape | Find capacity / latency knee (the headline number) |

## The validated smoke config

```yaml
qwen3-4b-2507-bf16-h100-vllm-mode1-smoke:
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-2507          # reuses the existing trace-replay launcher
  runner: h100-2x
  precision: bf16
  framework: vllm
  multinode: false
  scenarios:
    agentic-replay:
    - input-file: benchmarks/single_node/agentic/datasets/qwen3.5-4b-smoke.jsonl
      custom-dataset-type: mooncake_trace
      max-model-len: 8192
      benchmark-client: [aiperf]
      no-fixed-schedule: true          # Mode 1: suppress fixed-schedule timing
      strip-trace-delays: true         # Mode 1: drop per-turn `delay` at the source
      request-count: 50                # resample sessions to a fixed count
      num-warmup-sessions: 1           # warm prefix cache / CUDA graphs first
      search-space:
      - { tp: 1, conc-list: [8, 16, 32] }   # the CCU ladder
```

It is a **separate config key** from `qwen3-4b-2507-bf16-h100-vllm` (same
`model-prefix`, so it reuses `benchmarks/single_node/qwen3-4b-2507_bf16_h100_vllm.sh`)
so the smoke is isolated from the `conc=2` single-replay and the heavy 64k legs.

### The four Mode 1 config fields

These were added as **optional, default-off** fields (existing configs are
unaffected — the 64k / `conc=2` legs keep fixed-schedule single-replay behavior):

| Field | Default | Effect |
|---|---|---|
| `no-fixed-schedule` | `false` | Adds `--no-fixed-schedule` to `aiperf profile` (concurrency-driven timing). |
| `strip-trace-delays` | `false` | Launcher drops the `delay` key from each trace record → `/workspace/_trace_nodelay.jsonl`. |
| `request-count` | `null` | Explicit AIPerf `--request-count` (resampling); else = dataset record count. |
| `num-warmup-sessions` | `null` | Adds `--num-warmup-sessions` (warmup before the measured window). |

Validated in `utils/matrix_logic/validation.py` (`AgenticReplayConfig` and
`SingleNodeAgenticReplayMatrixEntry`), with a config-time guard
`validate_request_count_vs_conc` that rejects `request-count < max(conc-list)`.

## Reproduce / dispatch

```bash
gh api -X POST /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='exp/mode1-vs-mode3-agentic-64k' \
  -f 'inputs[ref]=exp/mode1-vs-mode3-agentic-64k' \
  -f 'inputs[test-name]=Qwen3-4B agentic Mode1 capacity smoke (conc 8/16/32)' \
  -f 'inputs[generate-cli-command]=test-config --config-keys qwen3-4b-2507-bf16-h100-vllm-mode1-smoke --config-files .github/configs/nvidia-master.yaml --no-evals --scenario-type agentic-replay' \
  -f 'inputs[duration-override]='
```

`workflow_dispatch` runs against any pushed branch ref — **no PR required**. The
top-level `ref` is the branch whose workflow YAML runs (must contain the
`agentic-replay` wiring); `inputs[ref]` is the repo code under test.

## Results (run 26933973732)

`Qwen/Qwen3-4B-Instruct-2507`, bf16, TP=1, vLLM `v0.21.0`, `h100-2x` runner (1 GPU
used for TP=1), `qwen3.5-4b-smoke.jsonl` (12 records, resampled to 50 requests),
delays stripped, `--no-fixed-schedule`.

| CCU | Total tput/GPU (tok/s) | Output tput/GPU (tok/s) | TTFT mean | TTFT p99 | ITL/TPOT mean | E2E mean |
|---|---|---|---|---|---|---|
| 8  | 24 987 | 1 261 | 36.3 ms  | 83.8 ms  | 5.34 ms | 506 ms |
| 16 | 37 412 | 1 912 | 77.2 ms  | 181.6 ms | 5.98 ms | 595 ms |
| 32 | 57 680 | 3 120 | 139.2 ms | 226.1 ms | 6.92 ms | 740 ms |

Throughput scales cleanly 8→16→32 while TTFT/ITL stay low — at this trace size the
knee is **not yet reached** by conc=32 (expected; this is a 50-request smoke, not a
capacity-finding run).

## ⚠️ Caveats

1. **Smoke = not a report number.** 50 requests resampled from a 12-record trace
   is far too few for TTFT/ITL to converge. Each AIPerf variation here is a
   **single run** — treat these numbers as a pipeline-validity check only. For a
   citable result, raise `request-count` (≥ a few hundred) and use a larger trace,
   ideally with repeats for a real mean±std.

2. **`request-count` must be ≥ the max swept concurrency.** Enforced in three
   places (`aiperf_adapter.py`, `benchmark_lib.sh`, and the config-time
   `validate_request_count_vs_conc`). The ladder here is `8,16,32` and
   `request-count: 50 ≥ 32`. If you extend to conc=64, raise `request-count`
   accordingly or the config is rejected before dispatch.

3. **Delay-strip is mandatory for Mode 1 and version-independent.** AIPerf 0.9.0
   honors Mooncake `delay` even under `--no-fixed-schedule` (concurrency uses the
   request-rate strategy, which sleeps `meta.delay_ms`), and has **no CLI flag** to
   ignore them (`--ignore-trace-delays` does **not** exist in any released version).
   So the launcher drops the `delay` field at the source when `strip-trace-delays`
   is set. Without it, the run is paced by recorded think-time, not concurrency.

4. **Per-leg `--max-num-seqs = conc`.** In CI each leg starts a fresh server with
   `--max-num-seqs` equal to that leg's concurrency, so every concurrency point is
   measured cold and isolated. A single-server local sweep (one server for all
   concurrencies) is **not** equivalent — it carries warm prefix cache between
   variations and inflates the high-concurrency legs. Use the CI matrix path for
   any apples-to-apples comparison.

5. **Power / tokens-per-Watt is unreliable on a short smoke.** `mean_power_w` is
   clipped to the last `duration` seconds, and a 50-request run is so short that
   the window catches inconsistent phases (this run shows power *falling* with
   concurrency, an artifact of windowing on sub-second-to-few-second runs). Ignore
   `tok_per_watt*` on smoke runs; only trust it on full-length sweeps.

6. **`total_token_throughput` is input-heavy.** Agentic traces have large
   accumulating contexts and small outputs (~88 output tokens/turn here), so
   `total tput/GPU` (input+output) reads ~20× `output tput/GPU`. State which
   convention any reported figure uses — see the Energy section of
   [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md#energy-efficiency-tokenswatt).

## Scaling to a real capacity sweep

1. Point `input-file` at a larger trace (`agentic-coding-64k.jsonl#<N>`), size
   `max-model-len` from the **session-cumulative** max (see the table in
   [`AIPERF_INTEGRATION.md`](AIPERF_INTEGRATION.md#context-length-requirements-per-dataset-size-max-model-len-from-the-session-cumulative-max)).
2. Raise `request-count` (e.g. 1000) and `num-warmup-sessions` (e.g. 16).
3. Extend `conc-list` upward (`[8, 16, 32, 64, 128, ...]`) and stop at the first
   leg that breaches the TTFT/ITL SLA — that leg is the capacity boundary.
4. Promote off the smoke key onto a report config key, append a
   `perf-changelog.yaml` entry, and dispatch with `full-sweep-enabled` if every
   intermediate concurrency point matters.
