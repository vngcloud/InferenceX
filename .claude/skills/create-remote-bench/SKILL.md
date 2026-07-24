---
name: create-remote-bench
description: Create and dispatch a new *-remote-bench.sh recipe to benchmark an already-running, externally-managed inference endpoint (BYO endpoint) instead of launching a server on the GPU runner. Use when bringing up remote-bench for a new model/precision/framework combo, or when asked to benchmark an existing deployment (e.g. a k8s/ArgoCD-managed SGLang/vLLM service) rather than a fresh InferenceX-launched server. Background and rationale: issue #26, PR #27.
---

# Create a remote-bench recipe

Remote-bench benchmarks a target InferenceX does not control the lifecycle of — no local
process, no docker, no GPU on the runner itself. Everything InferenceX normally derives by
launching the server (image, topology, context length) has to be **self-reported** by
whoever owns the endpoint instead.

## 1. Find out what's actually behind the endpoint

You need this even though you're not going to touch it — it feeds both the pre-flight
checks and the ingested artifact's identity fields.

- If the operator exposes a `/discover`-style endpoint (VNG's `inference-cicd` does), curl
  it — it typically returns `base_url`, `gpu_metrics_url`, `chart`, `framework`, `image`,
  `model`, `precision`, `servedName`, `tp` per stack.
- Otherwise ask the operator directly, or if you have cluster access,
  `kubectl get deployment <name> -n <namespace> -o yaml` and read the container `args`/`image`.
- **Context length is the one that matters most and is easy to miss**: read the deployed
  `--context-length` (SGLang) / `--max-model-len` (vLLM) argument directly. Do not guess or
  infer it from the model card — get the actual number the server was launched with.

## 2. Required parameters (the recipe's env-var contract)

Every `*-remote-bench.sh` recipe requires, self-reported by the endpoint's operator:

| Var | What it is | Why required |
|---|---|---|
| `REMOTE_BASE_URL` | e.g. `http://host/sglang-vanilla` | the actual target to hit |
| `REMOTE_GPU_TELEMETRY_URL` | DCGM `/metrics`-style endpoint | GPU telemetry is required for remote-bench, not optional (unlike aiperf's own soft-fail default for general use) |
| `REMOTE_ENGINE_METRICS_URL` | engine's own `/metrics` (e.g. SGLang's) | same — required, not optional |
| `REMOTE_RUNNER_TYPE` | real, `GPU_KEYS`-resolvable hw string, e.g. `h200-nv` | becomes `RUNNER_TYPE`/`hw` in the ingested artifact; the GH Actions runner label (`cluster:remote-bench`) is **not** a real hardware key and would break `hwToGpuKey()` in InferenceX-app's ingest if used directly |
| `REMOTE_MAX_CONTEXT_LENGTH` | a *safe* trace-length cap, not necessarily the model's full deployed context window | **confirmed by incident**: without this, aiperf replays trace turns longer than the model supports, relying on server-side auto-truncate — this triggered a silent 100%-GPU hang in SGLang's chunked-prefill continuation on oversized inputs. But setting it to the real deployed context window (e.g. `131072`) is **not automatically safe either** — on a single small/dev GPU (confirmed on an RTX 5090), individual traces near that limit (~120K tokens) still hung in decode after prefill completed cleanly (throughput collapsing to ~0.07 tok/s, never recovering). Also note the available public trace corpora only come in two sizes (unfiltered / `_256k`-capped, see `resolve_trace_source()` in `benchmark_lib.sh`) — there's no small-context variant, so capping below the corpus's shortest trace length (e.g. `32768`) fails outright with `DatasetLoaderError: All N traces exceed --max-context-length`. If a run hangs at the real context window, binary-search downward (e.g. try half the window) to find a cap this specific box's decode can actually sustain, rather than assuming the nominal window is safe. |

Optional:

| Var | What it is |
|---|---|
| `REMOTE_RESET_URL` | endpoint to reset KV/prefix cache + router affinity before each concurrency point — a remote target is one long-lived engine across the whole sweep, unlike local recipes which get a fresh process per `conc` job |

On the `remote-bench.yml` workflow_dispatch side, also required (these exist for every
recipe, but for remote-bench they're pure self-reported metadata rather than values that
configure anything InferenceX launches):

- `image` — the container image **actually deployed** behind the endpoint (from step 1).
  This is recorded verbatim into the ingested artifact's `image` field — never leave it as
  a placeholder.
- `model`, `model-prefix`, `framework`, `precision` — identity fields for ingest/labeling.
- `tp`, `ep`, `dp-attn` (default `tp=1`, `ep=1`, `dp-attn=false`) — topology metadata. Not
  enforced against the real deployment (InferenceX can't verify a black-box endpoint's
  actual topology), so report the real values or the per-GPU throughput math in the
  ingested artifact will be wrong.

## 3. Write the recipe file

One new file: `benchmarks/single_node/agentic/<model_prefix>_<precision>_<framework>-remote-bench.sh`.

Copy an existing one (`glm5.2_fp4_sglang-remote-bench.sh` or `dsv2lite_fp8_sglang-remote-bench.sh`)
and rename — **the body is model-agnostic and framework-agnostic by design**. Unlike local
recipes (one file per hardware target, because local server launch args are hw-specific),
remote-bench never launches a server, so there is no hw-specific tuning to encode. One file
per model+precision+framework combo is enough; do not create a new file per hardware/cluster.

The launcher naming formula (`runners/launch_bench-client.sh`) is:
```
benchmarks/single_node/agentic/${EXP_NAME%%_*}_${PRECISION}_${FRAMEWORK}-remote-bench.sh
```
So `exp-name`'s first underscore-delimited segment must equal `model-prefix`, and the
filename must match `<model-prefix>_<precision>_<framework>-remote-bench.sh` exactly.

Do not edit `benchmark_lib.sh` or any existing recipe for a new model — this workflow is
purely additive.

## 4. Runner

Remote-bench dispatches to the `cluster:remote-bench` label
(`configs/runners.yaml`), currently backed by one real non-GPU controller box
(`bench-client_01`). Reuse it — you don't need a new runner per model/target, since the
controller only drives aiperf over the network; it never touches the GPU itself. Only
register a new runner if the existing controller is saturated or unreachable from a new
target's network.

## 5. Dispatch

```bash
gh workflow run remote-bench.yml -R vngcloud/InferenceX --ref <branch> \
  -f exp-name=<model_prefix>_<short-desc> \
  -f image=<real deployed image> \
  -f model=<HF repo id or served name> \
  -f model-prefix=<model_prefix> \
  -f framework=sglang \
  -f precision=<precision> \
  -f conc=<N> \
  -f duration=<seconds> \
  -f remote-base-url=<url> \
  -f remote-gpu-telemetry-url=<url> \
  -f remote-engine-metrics-url=<url> \
  -f remote-runner-type=<hw string> \
  -f remote-max-context-length=<real context length>
```

`workflow_dispatch` only works once the workflow file exists on the **default branch**
(`main`) — you cannot dispatch a brand-new `remote-bench.yml`-style workflow from a feature
branch before it merges. For pre-merge validation, use the debug loop below instead.

## 6. Debug loop (do this before wiring into CI)

Mirrors `/debug-runs`'s tight-loop philosophy: reproduce directly rather than iterating
through full CI dispatch cycles you can't even trigger yet pre-merge.

1. SSH onto the controller box (`bench-client_01` or whichever `cluster:remote-bench`
   runner), clone/checkout the branch under test.
2. Run `runners/launch_bench-client.sh` directly, exporting every env var
   `benchmark-tmpl.yml` would normally set (`EXP_NAME`, `MODEL`, `MODEL_PREFIX`, `FRAMEWORK`,
   `PRECISION`, `CONC`, `DURATION`, `SCENARIO_TYPE=agentic-coding`,
   `SCENARIO_SUBDIR=agentic/`, `IS_AGENTIC=1`, `KV_OFFLOADING=none`, the `REMOTE_*` vars,
   and `GITHUB_WORKSPACE`/`RUNNER_NAME` pointing at your checkout) — plus
   `RESULT_DIR`/`INFMAX_CONTAINER_WORKSPACE` overridden to a real path on the box (the
   launcher already does this; every other launcher assumes a docker bind mount that
   doesn't exist here).
3. Before touching aiperf, curl the three required URLs yourself — `/health`,
   `REMOTE_GPU_TELEMETRY_URL`, `REMOTE_ENGINE_METRICS_URL` — with both `-I` (HEAD) and a
   plain GET. A proxy/exporter that answers GET but 501s on HEAD is a real thing you may
   hit; if aiperf's own reachability probe (HEAD-first, GET-fallback) still reports an
   endpoint unreachable despite curl succeeding, don't assume it's fixed — retest after any
   endpoint-side change, this has been flaky/order-dependent in practice.
4. If the server hangs mid-run (no crash, no new log lines, but GPU utilization pegged at
   ~100%), check `REMOTE_MAX_CONTEXT_LENGTH` against the traces actually being replayed
   first — this exact symptom was chunked-prefill continuation on an oversized,
   auto-truncated input. `kubectl logs <pod> -n <namespace>` on the endpoint's actual pod is
   the only place server-side errors (e.g. `Health check failed. Server couldn't get a
   response from detokenizer...`) show up; nothing about them reaches the aiperf client or
   GH Actions logs.
5. Iterate on the node until a run completes with `replay_rc=0` and real
   `profile_export_aiperf.{csv,json}` / `server_metrics_export.json` files with actual data
   in them (not empty) before considering the recipe done.

## 7. What a real, ingest-able run produces

For `scenario-type: agentic-coding` (which all remote-bench recipes are), `benchmark-tmpl.yml`
uploads:

- `bmk_agentic_<name>` — the aggregated `agg_*.json` result (throughput, latency, the
  identity fields from step 2 above). This is what ultimately reaches InferenceX-app's
  ingest.
- `agentic_<name>` — `results/**` (aiperf's raw artifacts: `profile_export_aiperf.{csv,json}`,
  `server_metrics_export.{csv,json}`, timeslices, `aiperf.log`).

`server_logs_<name>` is still uploaded, but for remote-bench it's just the aiperf **client's**
own `benchmark.log` (aiperf's own startup/runner log), not an actual inference-engine server
log — there's no locally-launched process to redirect. Don't confuse the two when reading it;
the real engine-side story only lives in `kubectl logs` on the endpoint's actual pod.
`gpu_metrics_<name>` (the local `nvidia-smi`/`amd-smi` capture) is **not** produced — no
local GPU on the controller box. That data still exists, just inside `agentic_<name>` via
aiperf's own GPU telemetry scrape (`gpu_telemetry_export.jsonl`) instead of a separate local
capture.

Sanity-check before calling a run "done": open `agg_*.json` and confirm `hw` is a real
`GPU_KEYS`-resolvable string (not `cluster:remote-bench`), `image` is the real deployed
image (not a placeholder), and the throughput numbers are non-zero.
