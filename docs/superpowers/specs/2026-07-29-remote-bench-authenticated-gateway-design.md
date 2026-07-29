# Remote-bench against an authenticated MaaS gateway

Date: 2026-07-29
Branch: `vng-benchmark`
Status: design approved, not yet implemented

## Problem

`remote-bench` today can only benchmark an endpoint that is unauthenticated, exposes a
DCGM GPU-telemetry endpoint, exposes its engine `/metrics`, answers `GET /health`, and
serves a model under its own HuggingFace repo id. GreenNode's MaaS gateway
(`https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1`) satisfies none of those except
engine metrics, which live on a different host entirely.

Four gaps block it:

1. **No API key path.** Every gateway route returns `401`. aiperf accepts `--api-key`, but
   nothing in InferenceX plumbs a secret to it, and the key must never reach the repo.
2. **GPU telemetry is a hard requirement.** The recipe fails the job when
   `REMOTE_GPU_TELEMETRY_URL` is unreachable. No DCGM exporter exists on this target.
3. **Served name != tokenizer name.** The gateway serves `z-ai/glm-5.2`; the tokenizer must
   resolve as `zai-org/GLM-5.2-FP8`. `build_replay_cmd` passes only `--model`, which aiperf
   uses for both.
4. **Context cap is mandatory.** `REMOTE_MAX_CONTEXT_LENGTH` is required, but a target whose
   window exceeds the corpus needs no cap at all.

## Goals

- Benchmark an authenticated, externally-managed OpenAI-compatible endpoint with the
  **same** aiperf agentic command InferenceX uses natively.
- Keep the API key out of the repository, out of uploaded artifacts, and out of logs.
- Teach `create-remote-bench` to derive serving config from engine metrics and interview the
  operator only for what metrics cannot reveal.

## Non-goals

- Auto-dispatching remote-bench results to InferenceX-app production ingest. Still open
  (issue #28); `remote-bench-e2e.yml` deliberately has no `trigger-agentic-ingest` job and
  this work does not add one.
- Verifying a black-box endpoint's self-reported topology. InferenceX cannot, and this does
  not try.
- Supporting more than one API-key secret. One repo secret covers the current need.

## Decisions

**Target the gateway, not the backing engine.** aiperf drives
`https://maas-llm-aiplatform-hcm.api.vngcloud.vn`, measuring what a customer experiences
(auth, routing, rate limiting) rather than the engine in isolation. Consequence: reported
TTFT includes gateway hops, and the scraped engine metrics come from a backend we cannot
prove is the one being routed to. Both must be stated in any report built on these numbers.

**The aiperf command deviates from native agentic in exactly two places**, both
operator-controlled: `--max-context-length` and `--unsafe-override`. Everything else stays
byte-identical — `--scenario inferencex-agentx-mvp`, `--random-seed 42`,
`--num-dataset-entries 393`, `--agentic-cache-warmup-duration 600`, the trajectory-start
ratios, `--use-server-token-count`, `--slice-duration 1.0`,
`--tokenizer-trust-remote-code`, `--failed-request-threshold`. `--tokenizer` is added to the
builder but emitted only when `AIPERF_TOKENIZER` is set, which native recipes never do, so
their command is unchanged.

**Engine metrics stay required; GPU telemetry becomes optional.** A remote-bench result
without KV, cache-hit and queue data is not interpretable, so `REMOTE_ENGINE_METRICS_URL`
keeps its hard-fail. GPU power and utilization are valuable but not load-bearing, so an
absent `REMOTE_GPU_TELEMETRY_URL` degrades to `--no-gpu-telemetry` rather than failing.

**Shared preflight in `benchmark_lib.sh`, not a third copy.** The two existing
`*-remote-bench.sh` files are ~95% identical boilerplate. Authentication and optional
telemetry are properties of remote-bench in general, so they belong in one function that all
recipes call. A per-recipe copy would leave the next author who copies the wrong file with
the old strict behavior. Net effect on the tree is a deletion.

**`--api-key` on the command line is the only available transport.** aiperf has no env-var
binding for `api_key`; the `AIPERF_*` settings are a separate internal-config system that
does not cover `EndpointConfig`. aiperf already redacts the key in its own exported config
(`@field_serializer("api_key")`), so only InferenceX's own two write points need fixing.

## Architecture

```
GH repo secret  GREENNODE_API_KEY          (vngcloud/InferenceX)
 └─ remote-bench-e2e.yml                    secrets: inherit
     └─ benchmark-tmpl.yml    env: REMOTE_API_KEY: ${{ secrets.GREENNODE_API_KEY }}
         └─ runners/launch_bench-client.sh  (plain env inheritance; `bash "$BENCH_SCRIPT"`)
             └─ benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh
                 ├─ remote_bench_preflight()   authenticated reachability probe
                 └─ build_replay_cmd()         appends --api-key / --tokenizer
                     └─ aiperf profile … --api-key <key>
```

The key exists only as process environment and as an argv element. It is never written to a
file in unredacted form.

## Components

### 1. `benchmarks/benchmark_lib.sh`

**`build_replay_cmd()`** — two additions, both conditional, both no-ops when the variable is
unset:

- `--api-key "$REMOTE_API_KEY"` when `REMOTE_API_KEY` is non-empty. Reject a key containing
  whitespace first: `$REPLAY_CMD` is word-split at exec, so a key with a space would silently
  split into two argv elements and authenticate with a truncated credential.
- `--tokenizer "$AIPERF_TOKENIZER"` when `AIPERF_TOKENIZER` is non-empty.

**`run_agentic_replay_and_write_outputs()`** — close the two leak points:

- Redact the key when writing `benchmark_command.txt`, substituting the literal string
  `$REMOTE_API_KEY` for the value. This file is uploaded as an artifact and GitHub Actions
  does not mask artifact contents.
- Suppress `set -x` around the exec when `REMOTE_API_KEY` is set. The existing
  `$REPLAY_CMD 2>&1 | tee benchmark.log` redirects the pipeline's stderr into `tee`, so the
  xtrace of the expanded command lands in `benchmark.log`. The command is already recorded
  (redacted) in `benchmark_command.txt`, so nothing is lost.

**New `remote_bench_preflight()`** — the body currently duplicated across remote-bench
recipes, plus the new behavior:

- Normalize `REMOTE_BASE_URL` by stripping a trailing `/v1`. `build_replay_cmd` appends
  `--endpoint /v1/chat/completions`, so a base URL ending in `/v1` produces `/v1/v1/…`. Warn
  when stripping so the operator sees it happened.
- Require `REMOTE_ENGINE_METRICS_URL` and verify it is reachable; hard-fail otherwise.
- If `REMOTE_GPU_TELEMETRY_URL` is set, verify and export `AIPERF_GPU_TELEMETRY_URL`.
  If unset, proceed without it (`build_replay_cmd` already emits `--no-gpu-telemetry` when
  `AIPERF_GPU_TELEMETRY_URL` is empty).
- Reachability probe: when `REMOTE_API_KEY` is set, `GET $REMOTE_BASE_URL/v1/models` with
  `Authorization: Bearer …`, requiring HTTP 200. Otherwise keep the existing unauthenticated
  `GET /health`. The authenticated probe fails a wrong key in seconds instead of after a
  3600 s run of `401`s.
- Export `RUNNER_TYPE="$REMOTE_RUNNER_TYPE"`, `AIPERF_SERVER_METRICS_URLS`,
  `AIPERF_TOKENIZER="$REMOTE_TOKENIZER"`, and `MAX_MODEL_LEN="$REMOTE_MAX_CONTEXT_LENGTH"`
  only when the latter is non-empty.
- Call `REMOTE_RESET_URL` when set, as today.

### 2. `.github/workflows/benchmark-tmpl.yml`

- New env `REMOTE_API_KEY: ${{ secrets.GREENNODE_API_KEY }}`.
- New input `remote-tokenizer` (string, not required), exported as `REMOTE_TOKENIZER`.
- `remote-max-context-length` and `remote-gpu-telemetry-url` descriptions updated to say
  optional. No schema change — they are already `required: false` string inputs.

### 3. `.github/workflows/remote-bench-e2e.yml`

- Stop hardcoding `kv-offloading: none`. Pass `${{ matrix.config.kv-offloading || 'none' }}`,
  plus `kv-offload-backend` and `kv-offload-backend-metadata`, which
  `benchmark-tmpl.yml` already accepts. Without this the ingested artifact claims no
  offloading for a target with HiCache demonstrably enabled.
- New matrix field `remote-tokenizer`, defaulting to empty.

### 4. `benchmarks/single_node/agentic/glm5.2_fp8_sglang-remote-bench.sh`

New file. Required by the launcher's naming formula
`${EXP_NAME%%_*}_${PRECISION}_${FRAMEWORK}-remote-bench.sh` — the existing
`glm5.2_fp4_sglang-remote-bench.sh` cannot be reused for an fp8 target. With
`remote_bench_preflight()` carrying the shared logic the file is roughly:

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

Note `REMOTE_GPU_TELEMETRY_URL` and `REMOTE_MAX_CONTEXT_LENGTH` leave `check_env_vars`.

### 5. Existing remote-bench recipes

`glm5.2_fp4_sglang-remote-bench.sh` and `dsv2lite_fp8_sglang-remote-bench.sh` are reduced to
the same shape, calling `remote_bench_preflight()`. Their behavior is unchanged when their
current env vars are all set, because the function is a superset of what they do today. The
one intentional behavior change is that an unset `REMOTE_GPU_TELEMETRY_URL` no longer fails
them — previously impossible, since it was required.

### 6. `.claude/skills/create-remote-bench/SKILL.md`

New **step 0, "Interview the operator"**, ahead of the current step 1. It scrapes
`REMOTE_ENGINE_METRICS_URL` and derives:

| Derived | From |
|---|---|
| model, precision | `model_name` label (e.g. `zai-org/GLM-5.2-FP8` → fp8) |
| engine context window | `sglang:context_len` |
| KV pool, CCU ladder | `sglang:max_total_num_tokens` × DP-rank count |
| TP / DP / EP | cardinality of `tp_rank` / `dp_rank` / `moe_ep_rank` labels |
| `kv-offloading` | `sglang:hicache_host_total_tokens` > 0 → `dram` |
| spec decoding on/off | presence of `sglang:spec_accept_length` |

It prints these as a table for confirmation, then asks only for what metrics cannot reveal:
container image, hardware string (`REMOTE_RUNNER_TYPE`, must be `GPU_KEYS`-resolvable),
`ep` when the label is not broken out per rank, the spec-decoding algorithm name, the
`kv-offload-backend` name, the gateway URL and served model name, the tokenizer repo id when
it differs from the served name, the API-key secret name, and — for gateway targets — the
**gateway-enforced** context limit, which may be lower than the engine's `context_len` and
is not discoverable from metrics.

Existing sections change:

- `REMOTE_MAX_CONTEXT_LENGTH` moves from required to optional, with the three cases stated:
  target window above the corpus → omit; below → set it; run hangs → binary-search the cap
  downward. The existing incident notes stay.
- GPU telemetry moves to optional; engine metrics stay required.
- New section on authenticated endpoints: the `GREENNODE_API_KEY` secret, the prohibition on
  committing keys, the authenticated preflight, and the redaction guarantee.
- The corpus-selection note gains the `resolve_trace_source()` detail that `MODEL_PREFIX`
  matching `dsv4*|glm5.2*|minimaxm3*` selects the **unfiltered** corpus and everything else
  the 256k-capped variant — so a glm5.2-prefixed target inherits the unfiltered corpus
  regardless of its actual deployed window.
- Dispatch examples updated for the new and now-optional fields.

## The glm-5.2 dispatch config

| Field | Value | Source |
|---|---|---|
| `remote-base-url` | `https://maas-llm-aiplatform-hcm.api.vngcloud.vn` | user-supplied, `/v1` stripped |
| `model` | `z-ai/glm-5.2` | gateway served name |
| `remote-tokenizer` | `zai-org/GLM-5.2-FP8` | `model_name` label |
| `remote-engine-metrics-url` | `https://49.213.86.184.nip.io/glm/sglang/worker-metrics` | verified reachable, unauthenticated |
| `remote-gpu-telemetry-url` | *(empty)* | no DCGM exporter found |
| `remote-max-context-length` | *(empty)* | engine window 500000; no cap for this target |
| `model-prefix` | `glm5.2` | → unfiltered corpus, → recipe filename |
| `precision` / `framework` | `fp8` / `sglang` | → `glm5.2_fp8_sglang-remote-bench.sh` |
| `tp` / `dp-attn` | `8` / `true` | `dp_rank` 0–7 × `tp_rank` 0–7 |
| `kv-offloading` | `dram` | `hicache_host_total_tokens` = 2,501,568 |
| `kv-offload-backend` | `native` | SGLang HiCache; the value used by existing `native` entries in `nvidia-master.yaml` |
| `conc` | `1, 2, 4, 8, 16, 32` | matches `qwen3.5-fp8-b200-sglang-agentic` |
| `duration` | `3600` | agentic default |

Observed engine state at design time: `max_total_num_tokens` 883,840 per DP rank
(≈7.07 M GPU tokens across 8 ranks), `page_size` 64, `num_pages` 13,810,
`hicache_host_total_tokens` 2,501,568, `spec_accept_length` 2.5–3.4. CCU 32 has ample KV
headroom, so the native ladder needs no trimming — which is what preserves the
apples-to-apples comparison.

Operator-supplied at dispatch, not derivable: container image, `REMOTE_RUNNER_TYPE`, `ep`,
spec-decoding algorithm name.

## Error handling

| Condition | Behavior |
|---|---|
| `REMOTE_API_KEY` contains whitespace | fail in `build_replay_cmd` before aiperf starts |
| `REMOTE_API_KEY` wrong or expired | authenticated `/v1/models` probe returns non-200, job fails in seconds |
| `REMOTE_ENGINE_METRICS_URL` unreachable | hard fail, as today |
| `REMOTE_GPU_TELEMETRY_URL` empty | proceed with `--no-gpu-telemetry` |
| `REMOTE_GPU_TELEMETRY_URL` set but unreachable | hard fail — an explicitly supplied URL that does not work is an error, not a degradation |
| `REMOTE_BASE_URL` ends in `/v1` | stripped with a warning |
| `REMOTE_MAX_CONTEXT_LENGTH` empty | `--max-context-length` omitted |
| Trace exceeds the target's real window | aiperf 4xx or, per issue #26, a silent 100%-GPU hang. Remedy is to set and binary-search `REMOTE_MAX_CONTEXT_LENGTH` downward |

## Testing

One runnable check, `benchmarks/test_remote_bench_secret.sh`: set `REMOTE_API_KEY` to a
known sentinel, call `build_replay_cmd` against a temp dir, and assert the sentinel appears
in `$REPLAY_CMD` but does **not** appear in the written `benchmark_command.txt`. This is the
only new logic that can fail silently and leak a credential, so it is the piece that gets a
test. It also covers the whitespace-rejection guard.

Manual sequence before the real ladder:

1. Run the recipe on the `bench-client_01` controller with env exported by hand, per the
   skill's existing debug loop. Confirm the authenticated preflight passes and
   `benchmark_command.txt` is redacted.
2. Dispatch one config at `conc: 1`, `duration: 300` (which triggers `--unsafe-override`;
   `submission_valid` will be false, as expected for a smoke test).
3. Confirm `agg_*.json` has a `GPU_KEYS`-resolvable `hw`, the real deployed `image`,
   non-zero throughput, and populated `server_metrics.kv_cache`.
4. Dispatch the full ladder.

`workflow_dispatch` reads the workflow file from the default branch, so the
`remote-bench-e2e.yml` changes must merge to `main` before the new matrix fields can be used
from a dispatch. Steps 1–2 do not depend on that.

## Security

- The key lives in one place: the `GREENNODE_API_KEY` repo secret in `vngcloud/InferenceX`.
- It reaches the runner as process environment only, and reaches aiperf as an argv element.
- Actions masks it in job logs. Artifacts are not masked, which is why
  `benchmark_command.txt` is redacted explicitly and xtrace is suppressed.
- aiperf redacts `api_key` in its own exported configuration by default.
- Anyone with `ps` access on the controller box during a run can read the key from argv.
  This is accepted: `bench-client_01` is a dedicated single-tenant controller, and aiperf
  offers no alternative transport.
