# Design: smoke-test throughput probe

Companion to `design/smoke-test-matrix.md` — this covers just the
`throughput` probe in that matrix: what it runs, where its parameters come
from, and what's reusable from existing team work.

Throughput testing on this team is standardized on `aiperf` — this probe
uses `utils/bench_serving/aiperf_adapter.py` (a wrapper around `aiperf
profile`) plus the install pattern in `benchmarks/benchmark_lib.sh`'s
`ensure_aiperf()`. Both currently live on the `dev` branch, not `main`.

## /discover and /version drive almost all of it

Nothing about *where to send requests* or *what model is being served*
should be hand-declared in `smoke-tests.yaml` — that all comes from the live
`/discover`/`version_url` self-report. Only *how hard to push*
(concurrency/duration/isl/osl) is InferenceX's own input:

| `aiperf_adapter.py` flag | Source |
|---|---|
| `--url` | `discover.base_url` |
| `--endpoint` | `discover.endpoint` (pass through explicitly — see below) |
| `--model` | `discover.model` |
| `--endpoint-type` | derived from `discover.endpoint` shape (see below) |
| result-row labels (framework/precision/tp) | `discover.framework` / `discover.precision` / `discover.tp` |
| `--isl` / `--osl` / `--concurrency` / `--benchmark-duration` | `smoke-tests.yaml` per-stack `throughput:` block (not discoverable) |

`--endpoint-type chat` makes `aiperf` default to appending
`/v1/chat/completions` to `--url` on its own — which happens to match every
current stack's discovered `endpoint`. Still pass `--endpoint` explicitly
from `discover.endpoint` rather than relying on the default: some future
stack may serve under a non-standard path (a Bailian/BytePlus-style
provider serving under `/api/v3/...`, as already seen in the team's other
remote configs), and passing it through makes that not our problem to
special-case. Derive `--endpoint-type` from the discovered path's suffix:
ends with `chat/completions` → `chat`; ends with `completions` (not chat) →
`completions`; anything else → fail loudly rather than guess.

## GPU telemetry (tokens/watt)

`aiperf` natively supports a `--gpu-telemetry <url>` flag pointed at a DCGM
exporter's `/metrics` endpoint, and `aiperf_adapter.py` already threads it
through as `--gpu-telemetry-url`. This is what powers-normalized metrics
like tokens/watt need — request-level metrics alone (tokens/sec, TTFT, ITL)
can't derive actual GPU power draw.

`/discover` now includes a top-level `gpu_metrics_url` (e.g.
`http://116.118.91.176.nip.io/gpu-metrics`), which returns an index of the
two GPU nodes with a per-node metrics URL each
(`/gpu-metrics/<node>`, real DCGM Prometheus text, `404` for unknown nodes).
This is genuine, live, per-GPU power data (`DCGM_FI_DEV_POWER_USAGE`),
already labeled with `pod`/`namespace` so power draw can be attributed to a
specific stack's pod.

**Nodes are multi-tenant — a raw per-node URL cannot be fed to
`--gpu-telemetry` unfiltered.** `aiperf` scrapes and aggregates every GPU
line at the URL it's given; it has no pod-label filter. Checked both nodes
live:

- `vks-ai-infrence-dev-5090-2x-eb1e0`: both GPUs belong to
  `sglang-pd-disaggregation` only (prefill + decode pods). Safe to point
  `--gpu-telemetry-url` straight at this node.
- `vks-ai-infrence-dev-5090-4x-3a11b`: **4 GPUs, shared** — GPU 0-1 →
  `sglang-vanilla`, GPU 2 → idle/unlabeled, GPU 3 →
  `sglang-mooncake-store`. Pointing this node's URL at either stack's
  `--gpu-telemetry-url` would silently mix in the other stack's (and an
  idle GPU's) power draw — wrong tokens/watt for both.

So today: safe to wire up for `sglang-pd-disaggregation`; **not** safe for
`sglang-vanilla`/`sglang-mooncake-store` without a change, since they share
a node.

The fix that keeps using `aiperf`'s native ingestion unchanged (rather than
bypassing it to hand-filter DCGM lines by pod label ourselves) is a request
to `inference-cicd`: add a **per-stack** `gpu_metrics_url` to each
`/discover` entry (same convention as `version_url`) that's already
pre-filtered to that stack's own pod's GPU lines. Until that exists, the
probe should only pass `--gpu-telemetry-url` through for stacks it can
verify are alone on their node (`sglang-pd-disaggregation` today), and skip
the metric for the rest rather than report a contaminated number.

## Probe flow

1. `GET /discover`, select the stack entry by name (shared fetch across all
   three probes in the job — `metadata`/`tool-calling` need it too).
2. `GET <version_url>` — snapshot **before**.
3. For each `conc` in `smoke-tests.yaml`'s `throughput.conc-list`
   (sequentially — these share one live deployment's capacity, so
   concurrent probe runs would contaminate each other's numbers):
   ```
   python3 utils/bench_serving/aiperf_adapter.py \
     --model <discover.model> \
     --url <discover.base_url> \
     --endpoint <discover.endpoint> \
     --endpoint-type chat \
     --concurrency <conc> \
     --benchmark-duration <short-duration-seconds> \
     --isl <isl> --osl <osl> \
     --result-filename smoke_<stack>_conc<conc> \
     --result-dir <dir>
   ```
   This is the same adapter the sweep pipeline uses — it shells out to
   `aiperf profile ...` and writes InferenceX-schema JSON
   (`utils/process_result.py`-compatible: `model_id`, `max_concurrency`,
   `total_token_throughput`, `output_throughput`, ttft/tpot/itl/e2el
   percentiles) directly, no separate conversion step needed.
   `--benchmark-duration` should be short (e.g. 15-30s per concurrency
   level) — a smoke check wants a fast sanity signal on a shared production
   endpoint, not a rigorous steady-state sweep.
4. `GET <version_url>` again — snapshot **after**. If `chart`/`image`/`model`
   differs from the **before** snapshot, the stack redeployed mid-probe —
   flag the throughput result as `invalid_redeployed_mid_run: true` rather
   than silently reporting numbers that mixed two deployments.
5. Emit the per-conc result JSON(s) + a summary row (tokens/sec, TTFT, ITL)
   into `$GITHUB_STEP_SUMMARY`.

## Runner / install requirements

`aiperf_adapter.py` only requires the `aiperf` CLI on `PATH`.
`benchmark_lib.sh`'s `ensure_aiperf()` resolves that with, in order: (1)
already on PATH — no-op; (2) `AIPERF_SOURCE_DIR` set to a local checkout —
editable install from there; (3) otherwise, plain `pip install
aiperf==0.9.0` from PyPI into a throwaway venv. Path (3) is all the
smoke-test workflow needs — **no submodule, no Docker, no self-hosted
runner required.** This is a real difference from the `remote:`/agentic-replay
path (`runners/launch_remote.sh`), which needs a self-hosted
`benchmark-client` runner reaching a private LAN and a heavier
`aiperf-mooncake` submodule editable install for trace-replay features we
don't use here. The smoke throughput probe only needs synthetic isl/osl
mode, which the plain PyPI `aiperf` package already supports — runs fine on
a normal hosted `ubuntu-latest` runner talking to the public `nip.io`
Ingress.

## What's reusable from existing team work

- **Reuse directly**: `utils/bench_serving/aiperf_adapter.py` (currently on
  `dev`) — this design assumes it's available, i.e. the smoke-test workflow
  branches off `dev`, not `main`, or this file gets cherry-picked/merged
  forward first.
- **Reuse the install pattern, not the runner wiring**: `ensure_aiperf()` in
  `benchmarks/benchmark_lib.sh` for the PyPI-fallback install logic. Skip
  `run_client_benchmark`'s `BENCHMARK_CLIENT=aiperf` branch and
  `runners/launch_remote.sh` entirely — both assume a server was (or will
  be) launched by the same job, or a self-hosted runner on the target's
  private network. Neither applies: the smoke job never launches a server
  and the target is a public Ingress.
- **Not needed**: `utils/aiperf-mooncake` submodule, `--scenario
  inferencex-agentx-mvp`, trace/dataset flags (`--public-dataset`,
  `--input-file`, `--custom-dataset-type`) — all agentic-replay-specific,
  irrelevant to a synthetic isl/osl smoke check.
- **Revisit later, not now**: `utils/process_result.py` /
  `utils/collect_results.py` for the deferred DB-ingest tagging
  (`run_type: live-check`) — `aiperf_adapter.py`'s output is already shaped
  to feed `process_result.py` directly, so that follow-up is a small step
  once `InferenceX-app` coordination happens, not a rewrite.

## Open item

`--concurrency`/`--benchmark-duration`/`isl`/`osl` defaults still need
picking (keep them small — e.g. isl=128, osl=128, duration=20s per
concurrency level — to bound added load on the shared production endpoint
per deploy). Tune once we see real latency numbers from a first run against
`sglang-vanilla`.
