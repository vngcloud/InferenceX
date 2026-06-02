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
artifact for debugging.

## Troubleshooting

- `World size (2) > available GPUs (1)`: the job landed on `h100-1x`; use
  `h100-2x` for TP=2 Gemma4 jobs.
- `aiperf` install fails: set `AIPERF_SOURCE_DIR` to a local checkout, or set an
  installable `AIPERF_VERSION`.
- AIPerf job passes but `compare-results` fails on PostgreSQL connection or
  missing `DATABASE_URL`: configure repo secret `NEON_PROD_RO_URL`.
- Result aggregation misses the client name: confirm `BENCHMARK_CLIENT=aiperf`
  is present in the benchmark environment before `utils/process_result.py` runs.
