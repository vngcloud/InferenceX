# AIPerf Benchmark Client Integration

This document covers the fixed-sequence AIPerf integration in InferenceX: current
status, how it is wired, and how to run it in GitHub Actions.

For agentic replay decisions, see
[`docs/adr/0001-agentic-on-official-aiperf.md`](adr/0001-agentic-on-official-aiperf.md).

## Status

As of 2026-06-02, AIPerf is integrated as a benchmark client for the
fixed-sequence single-node benchmark path. It is not a serving framework; vLLM,
SGLang, TRT, and other frameworks still start the model server, while AIPerf can
replace the native InferenceX HTTP load generator.

The current GreenNode config is `gemma4-bf16-h100-vllm` in
`.github/configs/nvidia-master.yaml`:

- `runner: h100-2x`, which maps to `h100-greennode_01` and exposes 2x H100.
- `benchmark-client: [aiperf]` for the final config.
- Sequence lengths: `1k1k` and `8k1k`.
- Concurrency values: `4, 8, 16, 32`.
- Full matrix size for this config: 8 entries.

Validated so far:

- H100 smoke runs were validated on the `h100-2x` runner hardware.
- A GitHub Actions smoke run
  ([26810883421](https://github.com/vngcloud/InferenceX/actions/runs/26810883421))
  on `h100-2x` passed 4/4 benchmark jobs for native and AIPerf at `1k1k`,
  concurrency `2` and `4`.
- The same smoke workflow failed only at `compare-results` because
  `NEON_PROD_RO_URL` was not configured, so `DATABASE_URL` was missing.

Pending if benchmark gating is required before merge:

- Add the repo secret `NEON_PROD_RO_URL`.
- Run the final AIPerf-only PR sweep, usually with `full-sweep-enabled` if every
  concurrency point matters.

## Integration Path

The fixed-sequence path is wired through one config field:

```yaml
benchmark-client: [aiperf]
```

The value flows through:

1. `.github/configs/*-master.yaml` scenario config.
2. `utils/matrix_logic/generate_sweep_configs.py`, which expands one matrix row
   per benchmark client.
3. `.github/workflows/run-sweep.yml` and `.github/workflows/benchmark-tmpl.yml`,
   where it becomes `BENCHMARK_CLIENT`.
4. The runner launcher, which passes `BENCHMARK_CLIENT` into the benchmark
   container.
5. The benchmark script, which calls `run_client_benchmark` from
   `benchmarks/benchmark_lib.sh`.

`run_client_benchmark` dispatches by client:

- `inferencex_native` runs `utils/bench_serving/benchmark_serving.py`.
- `aiperf` runs `utils/bench_serving/aiperf_adapter.py`, which invokes
  `aiperf profile` and converts `profile_export_aiperf.json` into the standard
  InferenceX intermediate result JSON.

`utils/process_result.py` then includes the selected client in aggregated output
as `benchmark_client`.

## Agentic Replay (mooncake_trace)

The `agentic-replay` scenario-type replays a recorded **mooncake_trace** JSONL
(`session_id`, `input_length`, `output_length`, `hash_ids`, `delay`) through
official AIPerf, riding the same `aiperf` client and the **standard** `bmk_*`
aggregation (`process_result.py`) — fully off the retired `cquil11/aiperf` fork
pipeline. See [`docs/adr/0001-agentic-on-official-aiperf.md`](adr/0001-agentic-on-official-aiperf.md).

The first committed config is `qwen3-4b-2507-bf16-h100-vllm` (a dense transformer
chosen so vLLM keeps prefix caching ON; the original `Qwen/Qwen3.5-4B` was a
hybrid-Mamba model for which vLLM auto-disables prefix caching):

```yaml
qwen3-4b-2507-bf16-h100-vllm:
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-2507
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
      search-space:
      - { tp: 1, conc-list: [2] }
```

How it differs from the fixed-sequence path:

- The trace is replayed **once**. `--request-count` equals the dataset record
  count (`grep -c .` on the JSONL — 12 for the smoke set), and `isl`/`osl` are
  **not** passed to AIPerf (the trace defines per-request lengths). The matrix
  entry still carries placeholder `isl=4096`/`osl=512` purely to satisfy
  downstream env checks in `process_result.py`.
- Wiring: a dedicated `single_node['agentic-replay']` bucket
  (`process_changelog.py`) feeds the `sweep-single-node-agentic-replay` job in
  `run-sweep.yml`, which uses `benchmark-tmpl.yml` with two new inputs
  (`input-file`, `custom-dataset-type`). Because the artifact gates key on
  `scenario-type != 'agentic-coding'`, results flow through `process_result.py`
  → `bmk_*` automatically. `SCENARIO_SUBDIR` stays empty, so the launcher
  resolves `benchmarks/single_node/qwen3-4b-2507_bf16_h100_vllm.sh`.
- The launcher starts vLLM (`Qwen/Qwen3-4B-Instruct-2507`, TP=1, bf16,
  `--max-model-len 8192`) and calls `run_client_benchmark --input-file ...
  --custom-dataset-type mooncake_trace --request-count <records>` with **no**
  `--isl/--osl`.

### Context-length requirements per dataset (size `max-model-len` from the *session-cumulative* max)

**Critical:** the mooncake_trace records of one `session_id` are replayed as a
**multi-turn conversation** — context **accumulates** across turns (each turn
carries the prior turns + their responses as prefix; this is exactly why prefix
caching matters here and why the cache-hit rate is ~95%). So the prompt size that
hits the server is **not** a record's `input_length` — it is the running
`sum(input_length + output_length)` over the session up to that turn. Size
`max-model-len` from that **session-cumulative max**, not the per-record max.

Empirically derived from the committed traces (per-record vs realized
session-cumulative `input+output`):

| Dataset | records | per-record max(in+out) | **session-cumulative max** | `max-model-len` to use |
|---|---|---|---|---|
| `qwen3.5-4b-smoke.jsonl` | 12 | 1,783 | 2,293 | **8192** (ample) |
| `agentic-coding-64k.jsonl` (`#2000` and full) | 2,000 / 18,595 | 38,613 | **66,655** | **73728** |
| `agentic-coding-128k.jsonl` (`#2000` and full) | 2,000 / 16,957 | 82,159 | **133,851** | **~147456** |

(Recompute with the one-off script in the handoff if traces change: group by
`session_id`, take the max running `sum(input_length+output_length)`.)

**Why this matters — silent truncation.** Sizing from the per-record length is the
trap that produced the 64k run [26874210796](https://github.com/vngcloud/InferenceX/actions/runs/26874210796):
`max-model-len 40960` (chosen from the 37,818 per-record input max) rejected
**1100/2000 (55%)** requests with HTTP 400 (`input+output > context window`, every
failure landing at exactly `max_model_len + 1`). AIPerf still emits metrics from
the survivors and the CI job goes **green**, so the failure is silent and the
reported numbers are biased to short early-turn requests. **Always verify the
server-log 200/400 ratio, not just job status.** Qwen3-4B-Instruct-2507 supports
256K context, so these `max-model-len` values carry no model-side limit; the real
constraint is KV pressure at high concurrency (expect preemptions for 128k at
conc=32 — lower the concurrency rather than the window).

Local dry-run against a running vLLM server (no CI):

```bash
source .venv/bin/activate
uv run python utils/bench_serving/aiperf_adapter.py \
  --model Qwen/Qwen3-4B-Instruct-2507 --url http://0.0.0.0:8000 --endpoint-type chat \
  --concurrency 2 --request-count 12 \
  --input-file benchmarks/single_node/agentic/datasets/qwen3.5-4b-smoke.jsonl \
  --custom-dataset-type mooncake_trace \
  --result-filename qwen-smoke --result-dir /tmp/qwen-smoke
```

To run the smoke in CI, append a `perf-changelog.yaml` entry for
`qwen3-4b-2507-bf16-h100-vllm` (with `scenario-type: [agentic-replay]`) and open a
PR to `dev` with the `sweep-enabled` label.

## Energy Efficiency (tokens/Watt)

Every benchmark run reports energy-efficiency metrics in the published
results table, in **two conventions** so the number is never ambiguous:

```text
tok/W total  = total_token_throughput  (input+output tok/s) / mean total GPU power (W)
tok/W output = output_token_throughput (decoded   tok/s) / mean total GPU power (W)
```

`tok/W total` is the GTC-2026 "AI Factory" efficiency KPI (tokens per watt). Both
ratios are GPU-count-invariant — per-GPU and whole-node give the same number — so
each is reported as a single whole-system value, not divided per GPU.

**Why both?** On input-heavy workloads (e.g. agentic-trace replay) or any run
where prefix caching is disabled, prefill tokens dominate the total, so
`tok/W total` reads high while `tok/W output` (useful decoded work per watt) can
be orders of magnitude lower. Always state which convention a reported figure
uses.

**Power source — no DCGM required.** Power comes from the `power.draw` column
that `start_gpu_monitor` (in `benchmarks/benchmark_lib.sh`) already logs to
`gpu_metrics.csv` via `nvidia-smi` on every run. This works for both the
`inferencex_native` and `aiperf` clients and for all serving frameworks, and is
independent of AIPerf's own GPU-telemetry subsystem (which, for the `aiperf`
client, additionally supports DCGM/pynvml — see `aiperf --gpu-telemetry`). When
AIPerf logs `DCGM telemetry skipped: no DCGM endpoints reachable`, the
`nvidia-smi` CSV is still captured and the metric is still computed.

**How it flows** (`utils/process_result.py`):

1. `mean_total_power_w()` parses `gpu_metrics.csv` and computes the per-GPU mean
   `power.draw`.
2. It sums only the **N busiest GPUs** — `N = TP` for single-node, `total_gpus`
   for multi-node — so an idle second card on a shared host (e.g. a TP=1 job on
   `h100-2x`) does not inflate power and understate efficiency.
3. Samples are clipped to the **last `duration` seconds** so model-load and
   warmup power (the monitor starts before the server) is excluded. The `aiperf`
   client supplies `duration` via AIPerf's `benchmark_duration` metric
   (`aiperf_adapter.py`); the native client already emits it.
4. The aggregated result JSON gains three fields, surfaced as the
   `Token/Watt total (tok/s/W)`, `Token/Watt output (tok/s/W)` and
   `Power Mean (W)` columns in `utils/summarize.py` (`tok_per_watt` is kept as an
   alias of `tok_per_watt_total` for backward compatibility):

   ```json
   { "tok_per_watt_total": 6.7093, "tok_per_watt_output": 5.1230, "mean_power_w": 355.0 }
   ```

Power telemetry is **best-effort**: a missing or empty `gpu_metrics.csv` (e.g. a
script that does not call `start_gpu_monitor`) leaves both fields `null`, which
renders as `0` in the table and never fails a run. Override the CSV path with the
`GPU_METRICS_CSV` environment variable if needed (default `gpu_metrics.csv`,
relative to the `process_result.py` working directory).

> Caveat: `nvidia-smi power.draw` is an instantaneous 1 Hz board-power gauge.
> For a more precise energy figure, AIPerf's `energy_consumption` hardware
> accumulator (DCGM/pynvml only) integrates power internally; it is not yet wired
> into the InferenceX result pipeline.

## AIPerf Installation

Serving images do not need to include AIPerf. `ensure_aiperf` in
`benchmarks/benchmark_lib.sh` resolves the CLI at runtime:

1. If `aiperf` is already on `PATH`, use it.
2. If `AIPERF_SOURCE_DIR` points to a Python project, install from that source.
3. Otherwise install `aiperf==${AIPERF_VERSION:-0.9.0}` from PyPI.

The default install target is an isolated in-container venv at
`/tmp/aiperf-venv`, so AIPerf dependencies do not mutate the serving image's
global Python packages.

Useful overrides:

- `AIPERF_VERSION=0.9.0` changes the PyPI version.
- `AIPERF_SOURCE_DIR=/path/to/aiperf` uses a local checkout.
- `AIPERF_VENV_DIR=/tmp/custom-aiperf-venv` changes the venv path.

## Config Usage

Use AIPerf only:

```yaml
scenarios:
  fixed-seq-len:
  - isl: 1024
    osl: 1024
    benchmark-client: [aiperf]
    search-space:
    - { tp: 2, conc-start: 4, conc-end: 32 }
```

Use both clients temporarily for a load-generator comparison:

```yaml
benchmark-client: [inferencex_native, aiperf]
```

Generate the current Gemma4 H100 matrix locally:

```bash
uv run python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix gemma4 \
  --precision bf16 \
  --framework vllm \
  --runner-type h100-2x
```

Expected grouping for the current config:

```text
4 h100-2x aiperf 1024 1024 4,8,16,32
4 h100-2x aiperf 8192 1024 4,8,16,32
```

## H100 Runner Notes

Use `h100-2x` for TP=2 jobs. The broad `h100` label also includes
`h100-greennode_00`, which is `h100-1x` and only exposes one GPU.

Ad-hoc validation should use the GitHub Actions `e2e-tests.yml` workflow with
`--runner-type h100-2x`, so the run follows the same launcher and artifact path
as CI.

## GitHub Actions Usage

PR sweeps do not run automatically. Add exactly one label:

- `sweep-enabled` runs the trimmed PR sweep.
- `full-sweep-enabled` runs every configured concurrency point.

Do not add both labels. The workflow rejects that combination.

For a manual one-off dispatch against this branch:

```bash
gh api -X POST \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=feat/aiperf-backend-integration' \
  -f 'inputs[test-name]=Gemma4 BF16 H100 AIPerf' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --single-node --model-prefix gemma4 --precision bf16 --framework vllm --runner-type h100-2x' \
  -f 'inputs[duration-override]='
```

`compare-results` requires `DATABASE_URL`, which `run-sweep.yml` maps from the
repo secret `NEON_PROD_RO_URL`. The database needs the baseline tables used by
`utils/compare_results.py`: `benchmark_results`, `configs`, and `workflow_runs`.

## Artifacts

A successful AIPerf run writes:

```text
<RESULT_FILENAME>.json
<RESULT_FILENAME>_aiperf/profile_export_aiperf.json
server.log
gpu_metrics.csv
```

`<RESULT_FILENAME>.json` is the adapted InferenceX result consumed by
`utils/process_result.py`. The `_aiperf` directory contains the raw AIPerf
artifact for debugging. `gpu_metrics.csv` is the `nvidia-smi` power/utilization
log; `process_result.py` reads its `power.draw` column to compute the
tokens/Watt metric (see [Energy Efficiency](#energy-efficiency-tokenswatt)).

## Troubleshooting

- `World size (2) > available GPUs (1)`: the job landed on `h100-1x`; use
  `h100-2x` for TP=2 Gemma4 jobs.
- `aiperf` install fails: set `AIPERF_SOURCE_DIR` to a local checkout, or set an
  installable `AIPERF_VERSION`.
- AIPerf job passes but `compare-results` fails on PostgreSQL connection or
  missing `DATABASE_URL`: configure repo secret `NEON_PROD_RO_URL`.
- Result aggregation misses the client name: confirm `BENCHMARK_CLIENT=aiperf`
  is present in the benchmark environment before `utils/process_result.py` runs.
