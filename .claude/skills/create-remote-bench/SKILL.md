---
name: create-remote-bench
description: Create a new *-remote-bench.sh recipe and run a full CCU-ladder sweep (not just a one-off smoke test) against an already-running, externally-managed inference endpoint (BYO endpoint) instead of launching a server on the GPU runner. Use when bringing up remote-bench for a new model/precision/framework combo, or when asked to benchmark an existing deployment (e.g. a k8s/ArgoCD-managed SGLang/vLLM service) rather than a fresh InferenceX-launched server. Background and rationale: issue #26/#28, PR #27/#30/#31.
---

# Create a remote-bench recipe

Remote-bench benchmarks a target InferenceX does not control the lifecycle of — no local
process, no docker, no GPU on the runner itself. Everything InferenceX normally derives by
launching the server (image, topology, context length) has to be **self-reported** by
whoever owns the endpoint instead.

## 0. Interview the operator (scrape first, then ask)

Most of what a benchmark config needs is already in the engine's own `/metrics`.
Scrape it before asking anything:

```bash
curl -sk "$REMOTE_ENGINE_METRICS_URL" > /tmp/wm.txt
grep -o '^sglang:[a-z_0-9]*' /tmp/wm.txt | sort -u          # what's exposed
grep -o 'model_name="[^"]*"' /tmp/wm.txt | sort -u           # model + precision
grep '^sglang:context_len{' /tmp/wm.txt | head -1            # engine window
grep '^sglang:max_total_num_tokens{' /tmp/wm.txt | head -1   # KV pool per rank
grep -o 'dp_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l     # DP degree
grep -o 'tp_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l     # TP degree
grep -o 'moe_ep_rank="[0-9]*"' /tmp/wm.txt | sort -u | wc -l # EP degree
grep '^sglang:hicache_host_total_tokens{' /tmp/wm.txt | head -1  # >0 => dram
grep '^sglang:spec_accept_length{' /tmp/wm.txt | head -1      # present => spec on
```

That gives you, without asking:

| Derived | From |
|---|---|
| model, precision | `model_name` label (`zai-org/GLM-5.2-FP8` → fp8) |
| engine context window | `sglang:context_len` |
| KV pool → CCU ladder | `sglang:max_total_num_tokens` × DP-rank count |
| TP / DP / EP | cardinality of the `*_rank` labels |
| `kv-offloading` | `sglang:hicache_host_total_tokens` > 0 → `dram` |
| spec decoding on/off | presence of `sglang:spec_accept_length` |

Show the operator that table and have them confirm it. Then ask only for what
metrics cannot reveal:

- **container image** actually deployed — recorded verbatim into the ingested
  artifact's `image` field; never leave it a placeholder
- **hardware string** for `REMOTE_RUNNER_TYPE` — must be `GPU_KEYS`-resolvable
  (e.g. `h200-nv`), not the runner label
- **`ep`**, when `moe_ep_rank` is not broken out per rank
- **spec-decoding algorithm name** — metrics prove it is on, not which one
- **`kv-offload-backend` name** — `native` for SGLang HiCache; the values in use
  across `configs/nvidia-master.yaml` are `native`, `vllm-simple`, `mooncake`
- **gateway URL and served model name**, when the target sits behind a gateway
- **tokenizer repo id**, when the served name is an alias → `remote-tokenizer`
- **API-key secret name** — currently `GREENNODE_API_KEY`
- **gateway-enforced context limit**, which may be *lower* than the engine's
  `context_len` and is not discoverable from metrics. This is the number that
  matters for `remote-max-context-length`, not the engine's.

A worked example: GreenNode's GLM-5.2 gateway exposes an unauthenticated
`worker-metrics` even though every other route 401s. Scraping it yielded
`zai-org/GLM-5.2-FP8` (→ fp8), `context_len` 500000, `max_total_num_tokens`
883840 across 8 DP ranks, `hicache_host_total_tokens` 2501568 (→ `dram`), and
live `spec_accept_length` values — leaving only image, hardware, `ep` and the
spec algorithm to ask about.

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
| `REMOTE_ENGINE_METRICS_URL` | engine's own `/metrics` (e.g. SGLang's) | required, not optional — without KV, cache-hit and queue data a remote-bench result is not interpretable, so `remote_bench_preflight` fails the job rather than let a multi-hour run finish without them |
| `REMOTE_RUNNER_TYPE` | real, `GPU_KEYS`-resolvable hw string, e.g. `h200-nv` | becomes `RUNNER_TYPE`/`hw` in the ingested artifact; the GH Actions runner label (`cluster:remote-bench`) is **not** a real hardware key and would break `hwToGpuKey()` in InferenceX-app's ingest if used directly |

Optional:

| Var | What it is |
|---|---|
| `REMOTE_RESET_URL` | endpoint to reset KV/prefix cache + router affinity before each concurrency point — a remote target is one long-lived engine across the whole sweep, unlike local recipes which get a fresh process per `conc` job |
| `REMOTE_GPU_TELEMETRY_URL` | DCGM `/metrics`-style endpoint. Absent means the run uses `--no-gpu-telemetry` and reports no GPU power or utilization — acceptable for a managed gateway that does not expose DCGM. A URL that *is* supplied but does not answer still fails the job: that is a misconfiguration, not an absent capability. |
| `REMOTE_MAX_CONTEXT_LENGTH` | a *safe* trace-length cap, not necessarily the model's full deployed context window. Three cases: target window **above** the corpus → omit it entirely; **below** the corpus → set it to the real enforced limit; run **hangs** at the nominal window → binary-search the value downward. **Confirmed by incident**: with a cap needed and absent, aiperf replays trace turns longer than the model supports, relying on server-side auto-truncate — this triggered a silent 100%-GPU hang in SGLang's chunked-prefill continuation on oversized inputs. But setting it to the real deployed context window (e.g. `131072`) is **not automatically safe either** — on a single small/dev GPU (confirmed on an RTX 5090), individual traces near that limit (~120K tokens) still hung in decode after prefill completed cleanly (throughput collapsing to ~0.07 tok/s, never recovering). Note also that capping below the corpus's shortest trace length (e.g. `32768`) fails outright with `DatasetLoaderError: All N traces exceed --max-context-length`. |
| `REMOTE_API_KEY` | bearer token, supplied by `benchmark-tmpl.yml` from the `GREENNODE_API_KEY` repo secret. Never commit a key. When set, the pre-flight probes `GET $REMOTE_BASE_URL/v1/models` with the token instead of `/health`, so a wrong or expired key fails in seconds rather than after a full-duration run of 401s. See §6b. |
| `REMOTE_TOKENIZER` | HuggingFace repo id, when the endpoint serves the model under an alias. aiperf's dataset manager loads a tokenizer by `--model` unless this overrides it, and an alias will not resolve on the Hub. |

On the `remote-bench-e2e.yml` workflow_dispatch side, also required per config (these exist for every
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

The body is model-, framework- and hardware-agnostic: everything shared lives in
`remote_bench_preflight()` in `benchmark_lib.sh`. A new recipe is just the pipeline:

```bash
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE
mkdir -p "$RESULT_DIR"
remote_bench_preflight
resolve_trace_source
install_agentic_deps
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
```

Unlike local recipes (one file per hardware target, because local server launch args are
hw-specific), remote-bench never launches a server, so there is no hw-specific tuning to
encode. One file per model+precision+framework combo is enough; do not create a new file
per hardware/cluster.

**Behavior changes belong in `remote_bench_preflight()`, not in a recipe.** A per-recipe fix
silently misses every other target. The recipes deliberately do not re-implement endpoint
probing, URL normalization, or the `RUNNER_TYPE` / `AIPERF_*` / `MAX_MODEL_LEN` exports —
`utils/test_benchmark_lib.py` asserts that none of them contain `curl`, `/health`,
`export AIPERF_`, `export RUNNER_TYPE` or `export MAX_MODEL_LEN` outside a comment.

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

## 5. Check the KV pool before picking a CCU ladder

Don't guess concurrency values for a sweep — find out how much KV cache capacity the
endpoint's engine actually has first. Too low and you leave throughput on the table; too
high and you push the engine into the queuing/decode-hang territory described in step 2's
`REMOTE_MAX_CONTEXT_LENGTH` entry.

For SGLang, the live `/metrics` endpoint (your `REMOTE_ENGINE_METRICS_URL`) exposes the KV
pool's total token capacity as a gauge set once at server startup — you can read it before
ever sending a request:

```bash
curl -s <REMOTE_ENGINE_METRICS_URL> | grep sglang:max_total_num_tokens
```

(This is the same number that later shows up in the aggregated result's
`server_metrics.kv_cache.gpu_total_tokens` — pulling it up front means you don't have to
run a throwaway job just to see it.)

Use it to size the ladder:

```
max_conc ≈ (kv_pool_tokens × target_utilization) / REMOTE_MAX_CONTEXT_LENGTH
```

`target_utilization` around 0.6-0.8 — pushing toward 1.0 is exactly the KV-pressure regime
that causes decode to collapse (see step 2). Build a doubling ladder up to that estimate —
`1, 2, 4, 8, ..., max_conc` — dropping the last step if it overshoots. Don't reuse a ladder
computed for a different model/hardware/`REMOTE_MAX_CONTEXT_LENGTH` combo; the math above
depends on all three.

**Check which corpus your `model-prefix` selects.** `resolve_trace_source()` picks the trace
corpus from `MODEL_PREFIX`, not from the target's actual deployed window:

```
dsv4*|glm5.2*|minimaxm3*)  → semianalysis_cc_traces_weka_062126        # unfiltered
*)                         → semianalysis_cc_traces_weka_062126_256k   # 256k-capped
```

So a `glm5.2`-prefixed target inherits the **unfiltered** corpus regardless of what the
deployment actually supports, because glm5.2's native window is ~1M. If your target's window
is smaller than the corpus, prefer `REMOTE_MAX_CONTEXT_LENGTH` — it keeps the same corpus and
filters only the traces above the cap, which stays comparable to a native run. Reach for
`WEKA_LOADER_OVERRIDE` to swap corpora only when the enforced limit is at or below 256k; that
is a different dataset, and results are then comparable only against other runs on the same
corpus.

vLLM targets don't expose an equivalent live token-count gauge the same way
(`vllm:kv_cache_usage_perc` is a percentage, not a token count; the raw number is normally
parsed from the server's own startup log, which remote-bench never captures — see step 8).
Ask the endpoint's operator directly for KV cache token capacity, or derive it from
`--gpu-memory-utilization` and the model's per-token KV footprint.

## 6. Dispatch

There's one workflow: `remote-bench-e2e.yml`. It takes a JSON array of configs and
matrix-fans each entry to its own job, then aggregates via `collect-results` +
`calc-success-rate` into one combined summary. A single-config smoke test is just a
one-element array — there's no separate single-dispatch workflow anymore (an earlier
`remote-bench.yml` existed for that and was removed once `remote-bench-e2e.yml` covered
both cases; don't recreate it).

Smoke test (one config):

```bash
gh workflow run remote-bench-e2e.yml -R vngcloud/InferenceX --ref <branch> \
  -f configs='[
    {"exp-name": "glm5.2_smoke", "conc": "1", "duration": "300",
     "image": "<real deployed image>", "model": "z-ai/glm-5.2",
     "model-prefix": "glm5.2", "framework": "sglang", "precision": "fp8",
     "tp": "8", "dp-attn": "true",
     "kv-offloading": "dram", "kv-offload-backend": "native",
     "total-cpu-dram-gb": "<host DRAM GB>",
     "remote-base-url": "https://maas-llm-aiplatform-hcm.api.vngcloud.vn",
     "remote-tokenizer": "zai-org/GLM-5.2-FP8",
     "remote-engine-metrics-url": "https://49.213.86.184.nip.io/glm/sglang/worker-metrics",
     "remote-runner-type": "<hw string>"}
  ]'
```

Only `exp-name`, `image`, `model`, `model-prefix`, `framework`, `precision`, `conc`,
`duration`, `remote-base-url`, `remote-engine-metrics-url` and `remote-runner-type` are
required. This example omits `remote-gpu-telemetry-url` (that gateway exposes no DCGM) and
`remote-max-context-length` (its 500k window exceeds the corpus) — both are now optional.
There is no `remote-api-key` field and there must never be one: the key arrives from the
`GREENNODE_API_KEY` secret, never through the dispatch payload. `total-cpu-dram-gb` is
operator-supplied (the endpoint's actual configured CPU DRAM capacity in GB) — it defaults
to `'0'` when omitted, and `kv-offloading: dram` fails at source time (in `benchmark_lib.sh`,
after the runner is already allocated) unless it is set to a positive integer alongside a
non-empty `kv-offload-backend`.

Full sweep (the CCU ladder) — same shape, one object per conc value:

```bash
gh workflow run remote-bench-e2e.yml -R vngcloud/InferenceX --ref <branch> \
  -f configs='[
    {"exp-name": "<model_prefix>_c1", "conc": "1", "duration": "<seconds>", ...},
    {"exp-name": "<model_prefix>_c2", "conc": "2", "duration": "<seconds>", ...},
    {"exp-name": "<model_prefix>_c4", "conc": "4", "duration": "<seconds>", ...}
  ]'
```

Every object needs the same field set as the smoke-test example above — only `conc` (and
`exp-name`, so results stay distinguishable) should vary across the ladder. Drop `duration`
back to the agentic default of `3600` for a real sweep; `300` only makes sense for a smoke
test, and anything under 900 forces `--unsafe-override`, which flags `submission_valid`
false.

**No `REMOTE_RESET_URL` configured means the ladder isn't a clean comparison.** A remote
target is one persistent engine across the whole sweep (unlike local recipes, which get a
fresh process per `conc` job) — without a reset endpoint, each subsequent point in the
ladder inherits KV/prefix cache warmth from every prior point (and from any earlier runs
against the same endpoint). Confirmed in practice: repeated runs against the same dataset
with no reset showed ~92-94% prefix-cache hit rate regardless of concurrency. If the
endpoint's operator can't provide a real reset endpoint, treat ladder results as smoke-test
validation that the sweep mechanics work, not as clean per-concurrency performance numbers.

`workflow_dispatch` only works once the workflow file exists on the **default branch**
(`main`) — you cannot dispatch a brand-new `remote-bench-e2e.yml`-style workflow from a
feature branch before it merges. For pre-merge validation, use the debug loop below instead.

## 6b. Authenticated endpoints

The key lives in exactly one place: the `GREENNODE_API_KEY` repo secret in
`vngcloud/InferenceX`. `benchmark-tmpl.yml` maps it to `REMOTE_API_KEY`, which the runner
inherits as process environment. Never put a key in a config, a dispatch payload, a recipe,
or a commit.

aiperf has no env-var transport for `api_key` — it must be an argv element. Two consequences,
both handled in `benchmark_lib.sh`:

- `benchmark_command.txt` is uploaded as an artifact and GitHub Actions does **not** mask
  artifact contents, so `redact_replay_cmd()` substitutes the literal `$REMOTE_API_KEY`
  before the file is written. **This is the primary control.**
- Shell xtrace expands the command and the bearer header. Every `*-remote-bench.sh` recipe
  runs `set -x` on line 3, so xtrace is already ON in all three functions that touch the key,
  and a guard that merely *skips* `set -x` would be a no-op against an already-enabled
  option. All three therefore run `set +x` actively:
  - `remote_bench_preflight` — around the authenticated probe, restoring prior state
  - `build_replay_cmd` — around the `-n` test, the whitespace test and the `--api-key`
    append, restoring prior state so a native recipe's tracing survives
  - `run_agentic_replay_and_write_outputs` — before the redaction call and the replay
    pipeline

  Note the suppression must start *before* the `[ -n "$REMOTE_API_KEY" ]` test, not just
  around the line that uses the value — the test itself traces as `+ '[' -n <key> ']'`.
  All of this is defense in depth: the trace goes to the job log, not to `benchmark.log`
  (verified on bash 3.2 — the tee'd file gets zero trace lines), and Actions already masks
  registered secrets in job logs. It earns its place because masking only holds while the
  value stays a registered secret.

Anyone with `ps` access on the controller box during a run can read the key from argv. This
is accepted: `bench-client_01` is a dedicated single-tenant controller, and aiperf offers no
alternative transport.

Note that a gateway typically 401s on `/health`, which is why the pre-flight switches to an
authenticated `GET /v1/models` probe when a key is present. Also supply `REMOTE_BASE_URL`
**without** a trailing `/v1` — `build_replay_cmd` appends `--endpoint /v1/chat/completions`.
The pre-flight strips a trailing `/v1` (with or without a trailing slash) and logs a NOTE
when it does, but do not rely on that.

After any change to this path, re-run the guard tests:

```bash
/Users/lap15120/greennode-code/.venv/bin/pytest utils/test_benchmark_lib.py -v -k "api_key or redact or preflight"
```

## 7. Debug loop (do this before wiring into CI)

Mirrors `/debug-runs`'s tight-loop philosophy: reproduce directly rather than iterating
through full CI dispatch cycles you can't even trigger yet pre-merge.

1. SSH onto the controller box (`bench-client_01` or whichever `cluster:remote-bench`
   runner), clone/checkout the branch under test.
2. Run `runners/launch_bench-client.sh` directly, exporting every env var
   `benchmark-tmpl.yml` would normally set (`EXP_NAME`, `MODEL`, `MODEL_PREFIX`, `FRAMEWORK`,
   `PRECISION`, `CONC`, `DURATION`, `SCENARIO_TYPE=agentic-coding`,
   `SCENARIO_SUBDIR=agentic/`, `IS_AGENTIC=1`, `KV_OFFLOADING` (and `KV_OFFLOAD_BACKEND`
   when it is `dram` — the pair is validated at source time), the `REMOTE_*` vars including
   `REMOTE_API_KEY` if the target is authenticated,
   and `GITHUB_WORKSPACE`/`RUNNER_NAME` pointing at your checkout) — plus
   `RESULT_DIR`/`INFMAX_CONTAINER_WORKSPACE` overridden to a real path on the box (the
   launcher already does this; every other launcher assumes a docker bind mount that
   doesn't exist here).
3. Before touching aiperf, curl each URL you supplied, yourself, with both `-I` (HEAD) and a
   plain GET: `REMOTE_ENGINE_METRICS_URL` (always), `REMOTE_GPU_TELEMETRY_URL` (only if you
   supplied one), and the target itself — `$REMOTE_BASE_URL/v1/models` with
   `-H "Authorization: Bearer $REMOTE_API_KEY"` for an authenticated gateway, or `/health`
   for an open one. A proxy/exporter that answers GET but 501s on HEAD is a real thing you
   may hit; if aiperf's own reachability probe (HEAD-first, GET-fallback) still reports an
   endpoint unreachable despite curl succeeding, don't assume it's fixed — retest after any
   endpoint-side change, this has been flaky/order-dependent in practice.
   Also confirm `benchmark_command.txt` came out redacted before you share any artifact:
   `grep -c "$REMOTE_API_KEY" "$RESULT_DIR"/benchmark_command.txt "$RESULT_DIR"/benchmark.log`
   must report 0 for both.
4. If the server hangs mid-run (no crash, no new log lines, but GPU utilization pegged at
   ~100%), check `REMOTE_MAX_CONTEXT_LENGTH` against the traces actually being replayed
   first — this exact symptom was chunked-prefill continuation on an oversized,
   auto-truncated input. If you left it unset because the target's window looked large
   enough, that is the first thing to reconsider: set it and binary-search downward. `kubectl logs <pod> -n <namespace>` on the endpoint's actual pod is
   the only place server-side errors (e.g. `Health check failed. Server couldn't get a
   response from detokenizer...`) show up; nothing about them reaches the aiperf client or
   GH Actions logs.
5. Iterate on the node until a run completes with `replay_rc=0` and real
   `profile_export_aiperf.{csv,json}` / `server_metrics_export.json` files with actual data
   in them (not empty) before considering the recipe done.

## 8. What a real, ingest-able run produces

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
