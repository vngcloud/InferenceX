# Design: standalone throughput test

**Smoke test and throughput test are two unrelated tests.** This is not a
probe inside `smoke-test.yml` — it's its own workflow
(`.github/workflows/throughput-test.yml`), its own config
(`.github/configs/throughput-tests.yaml`), and its own code
(`utils/throughput_test/`). It used to be a third smoke-test probe using a
tiny synthetic isl/osl sweep; moved out 2026-07-12 because throughput is an
important, heavier check against a shared production endpoint, and deserves
its own cadence, dataset, and ingest schema instead of riding along with a
fast correctness gate. See `design/smoke-test-matrix.md` for the (now
metadata + tool-calling only) smoke test.

Throughput testing on this team is standardized on `aiperf` — this workflow
uses `utils/bench_serving/aiperf_adapter.py` (a wrapper around `aiperf
profile`) plus the install pattern in `benchmarks/benchmark_lib.sh`'s
`ensure_aiperf()`.

## Why the dataset changed

The original smoke-test throughput probe used a flat synthetic sweep
(`isl=128, osl=128`) — deliberately minimal for a fast correctness ping, but
that also meant every sweep point measured the same fixed-shape padding
request, not anything resembling real traffic. This standalone workflow
instead uses `aiperf`'s `semianalysis_cc_traces_weka` public dataset: real
Claude Code coding-session traces (949 traces, 136k requests), hosted
publicly on HuggingFace
(`semianalysisai/cc-traces-weka-no-subagents-051226`, no auth required).
Prompt text is synthesized from a coding corpus against each trace's
preserved per-request hash_ids/timing (see
`utils/aiperf/src/aiperf/dataset/loader/semianalysis_cc_traces_weka.py`),
so sweep points reflect realistic coding-request shapes and richer output
(per-request latency percentiles across a real length distribution, not one
fixed isl/osl pair) instead of one synthetic corner case.

This is the lightweight, no-extra-setup sibling of the full agentic-replay
scenario (`--scenario inferencex-agentx-mvp`, `--custom-dataset-type
weka_trace`, file-based traces with subagent fan-out) used by the internal
sweep pipeline for heavier tp8/ep8 multi-GPU benchmarking — see e.g. run
28380380419 on `run-sweep.yml`. That system needs a self-hosted
`benchmark-client` runner reaching a private LAN, a `--scenario`-enforced
minimum duration (≥900s to reach steady state), and the `aiperf-mooncake`
submodule. None of that applies here: this workflow targets an
already-deployed, publicly reachable stack via `/discover`'s live Ingress
URL, on a normal hosted `ubuntu-latest` runner, with a much shorter
per-concurrency duration (see below) since it's a periodic live check, not a
from-scratch sweep.

## /discover and /version drive almost all of it

Nothing about *where to send requests* or *what model is being served*
should be hand-declared in `throughput-tests.yaml` — that all comes from the
live `/discover`/`version_url` self-report. Only *how hard to push*
(concurrency/duration) and *which dataset/how much of it* are InferenceX's
own input:

| `aiperf_adapter.py` flag | Source |
|---|---|
| `--url` | `discover.base_url` |
| `--endpoint` | `discover.endpoint` (pass through explicitly — see below) |
| `--model` | `discover.model` |
| `--endpoint-type` | derived from `discover.endpoint` shape (see below) |
| `--gpu-telemetry-url` | `discover.gpu_metrics_url` (per-stack, pre-filtered) |
| result-row labels (framework/precision/tp) | `discover.framework` / `discover.precision` / `discover.tp` |
| `--public-dataset` / `--num-dataset-entries` / `--concurrency` / `--benchmark-duration` | `throughput-tests.yaml` per-stack `throughput:`-shaped block (not discoverable) |

`--endpoint-type chat` makes `aiperf` default to appending
`/v1/chat/completions` to `--url` on its own — which happens to match every
current stack's discovered `endpoint`. Still pass `--endpoint` explicitly
from `discover.endpoint` rather than relying on the default, so a future
stack serving under a non-standard path isn't our problem to special-case.
Derive `--endpoint-type` from the discovered path's suffix: ends with
`chat/completions` → `chat`; ends with `completions` (not chat) →
`completions`; anything else → fail loudly rather than guess.

## GPU telemetry (tokens/watt)

Same per-stack `gpu_metrics_url` mechanism as before — see
`design/smoke-test-matrix.md`'s history for how this was resolved with
`inference-cicd`. Each `/discover` stack entry's `gpu_metrics_url` is
pre-filtered server-side to just that stack's own pod's GPU lines, and plugs
straight into `aiperf --gpu-telemetry-url` with no InferenceX-side
filtering needed.

## Run flow

1. `GET /discover`, select the stack entry by name (this workflow builds its
   own matrix independently from smoke-test's — see
   `utils/throughput_test/generate_matrix.py`).
2. `GET <version_url>` — snapshot **before**.
3. For each `conc` in `throughput-tests.yaml`'s `throughput.conc-list`
   (sequentially — these share one live deployment's capacity, so
   concurrent runs would contaminate each other's numbers):
   ```
   python3 utils/bench_serving/aiperf_adapter.py \
     --model <discover.model> \
     --url <discover.base_url> \
     --endpoint <discover.endpoint> \
     --endpoint-type chat \
     --gpu-telemetry-url <discover.gpu_metrics_url> \
     --concurrency <conc> \
     --benchmark-duration <duration> \
     --public-dataset semianalysis_cc_traces_weka \
     --num-dataset-entries <num-dataset-entries> \
     --tokenizer-trust-remote-code \
     --random-seed 42 \
     --result-filename throughput_<stack>_conc<conc> \
     --result-dir <dir>
   ```
   This is the same adapter the sweep pipeline uses — it shells out to
   `aiperf profile ...` and writes InferenceX-schema JSON
   (`utils/process_result.py`-compatible: `model_id`, `max_concurrency`,
   `total_token_throughput`, `output_throughput`, ttft/tpot/itl/e2el
   percentiles) directly, no separate conversion step.
4. `GET <version_url>` again — snapshot **after**. If it differs from the
   **before** snapshot, the stack redeployed mid-run — flag the result as
   `redeployed_mid_run: true` rather than silently reporting numbers that
   mixed two deployments.
5. Emit the per-conc result JSON(s) + a summary row (tokens/sec, TTFT, ITL)
   into `$GITHUB_STEP_SUMMARY`, plus a `--results-file` JSON artifact tagged
   `"run_type": "live-check"` and `"test_type": "throughput"` for
   `InferenceX-app` to pick up on its own tab.

## Trigger and cadence

Same trigger as smoke-test for now: `repository_dispatch` (`stack-deployed`)
fired by `inference-cicd` on deploy, plus `workflow_dispatch` for manual
runs. Runs as its own workflow job (`.github/workflows/throughput-test.yml`)
so its cadence, retries, and failures are independent of the
metadata/tool-calling correctness gate — a throughput hiccup shouldn't block
or get lost inside the fast smoke-test signal, and vice versa.

## Concurrency, duration, and dataset size

`throughput-tests.yaml` currently sets `conc-list: [1, 8, 32]` and
`benchmark-duration-s: 30` per stack, with `num-dataset-entries` defaulting
to 100 (of the full 949-trace corpus) in
`utils/throughput_test/run_throughput_test.py`. These are still a bounded
check against a shared production endpoint, not a from-scratch sweep — tune
once real numbers come in from a first run against `sglang-vanilla`. Raise
`num-dataset-entries` (up to 949) for a fuller run at the cost of a slower
sweep.

## Runner / install requirements

Same as before: `ensure_aiperf()`'s PyPI-fallback (`pip install
aiperf==0.9.0`) is all this workflow needs — no submodule, no Docker, no
self-hosted runner. Unlike the smoke-test workflow (which no longer runs any
`aiperf` at all after throughput moved out), this workflow **does** need
`HF_TOKEN` wired in (`secrets.HF_TOKEN`, mirroring `benchmark-tmpl.yml`/
`profile.yml`) — the `semianalysis_cc_traces_weka` dataset itself needs no
auth, but the model's own tokenizer download (e.g.
`RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8`) still hits HF Hub, and
running 3 stacks as parallel matrix jobs without a token risks the same
rate-limit crash the old smoke-test throughput probe hit (see the
2026-07-12 incident: `sglang-vanilla`/`sglang-mooncake-store` lost their
entire sweep to `HF_TOKEN`-less rate limiting while `sglang-pd-disaggregation`
in the same run succeeded).

## What's reusable from existing team work

- **Reuse directly**: `utils/bench_serving/aiperf_adapter.py` — already on
  `main`, already supports `--public-dataset`/`--num-dataset-entries`/
  `--tokenizer-trust-remote-code`/`--random-seed` with no changes needed.
- **Reuse the install pattern, not the runner wiring**: `ensure_aiperf()` in
  `benchmarks/benchmark_lib.sh` for the PyPI-fallback install logic. Skip
  `run_client_benchmark`'s `BENCHMARK_CLIENT=aiperf` branch and
  `runners/launch_remote.sh` entirely — both assume a server was (or will
  be) launched by the same job, or a self-hosted runner on the target's
  private network. Neither applies: this workflow never launches a server
  and the target is a public Ingress.
- **Not needed**: `--scenario inferencex-agentx-mvp`, `--custom-dataset-type
  weka_trace`/`--input-file` (file-based trace replay with subagent
  fan-out), `utils/aiperf-mooncake` submodule — all agentic-replay-specific,
  irrelevant to this workflow's plain concurrency-sweep-over-a-public-dataset
  mode.
- **Revisit later, not now**: `utils/process_result.py` /
  `utils/collect_results.py` for the deferred DB-ingest tagging
  (`run_type: live-check`) — `aiperf_adapter.py`'s output is already shaped
  to feed `process_result.py` directly, so that follow-up is a small step
  once `InferenceX-app` coordination happens, not a rewrite. See
  `InferenceX-app/design/new-test-design.md` for that side's schema
  decision.

## Open items

- `conc-list`/`benchmark-duration-s`/`num-dataset-entries` defaults are a
  first guess — tune once real latency/throughput numbers come back from a
  first live run against each stack.
- Whether to eventually raise `num-dataset-entries` toward the full
  949-trace corpus (richer signal, slower sweep, more load on the shared
  endpoint) is an open tradeoff, not decided yet.
- Exact `repository_dispatch` event name/payload shape (shared with
  smoke-test) still needs to be agreed with whoever owns the
  `inference-cicd` side of this.
