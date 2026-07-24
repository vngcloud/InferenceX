---
name: fork-changelog
description: Reference list of every vngcloud-fork-only edit on main relative to upstream (SemiAnalysisAI/InferenceX), with the reason each exists. Not a procedure — check this DURING a sync-upstream run to know which files/lines must survive the merge. Update it whenever a new fork-only change lands on main, or when one gets superseded/upstreamed.
---

# Fork-only changes on main (vs upstream)

`main` here is no longer a pure upstream mirror — as of the remote-bench feature (issue #26),
fork-specific commits live directly on `main`. This file is the ground truth for what those
changes are and why, so `sync-upstream` can tell "an upstream change touched one of our
lines" apart from "an upstream change is unrelated."

Diff anchor: `git diff <merge-base-with-upstream> origin/main --stat` shows exactly this set
of files (as of the remote-bench feature, merge-base was `673a6019`).

## Entries

### `benchmarks/benchmark_lib.sh`
- **Line ~1767**, `build_replay_cmd()`: `REPLAY_CMD+=" --url ${REMOTE_BASE_URL:-http://localhost:$PORT}"`.
  **Why**: lets a recipe point aiperf at an externally-managed endpoint (`REMOTE_BASE_URL`)
  instead of the locally-launched server on `$PORT`. This is the single hardcode that caused
  the original remote-bench bug (issue #26) before it was fixed — if upstream changes how
  `build_replay_cmd` constructs `--url`, this fallback pattern must be re-applied, not dropped.

### `.github/workflows/benchmark-tmpl.yml`
- 6 `workflow_call` inputs + matching `env:` entries, all optional/empty-default:
  `remote-base-url`, `remote-gpu-telemetry-url`, `remote-engine-metrics-url`,
  `remote-reset-url`, `remote-runner-type`, `remote-max-context-length` →
  `REMOTE_BASE_URL`, `REMOTE_GPU_TELEMETRY_URL`, `REMOTE_ENGINE_METRICS_URL`,
  `REMOTE_RESET_URL`, `REMOTE_RUNNER_TYPE`, `REMOTE_MAX_CONTEXT_LENGTH`.
  **Why**: only consumed by `*-remote-bench.sh` recipes; every other recipe just sees them
  empty. If upstream restructures this file's `inputs:`/`env:` blocks, all 6 pairs need to
  be re-added in the same input-name → env-var-name shape, since `remote-bench.yml` and
  `remote-bench-e2e.yml` both reference these exact input names.

### `.github/workflows/remote-bench.yml` (new file)
Single-config `workflow_dispatch` entrypoint for remote-bench. Deliberately standalone
instead of a `generate_sweep_configs.py` scenario type — the master-config schema
(`validation.py`) has no fields for per-run remote endpoint values, and extending it was
explicitly rejected in favor of this. **Why it's a separate file, not folded into
`run-sweep.yml`/`e2e-tests.yml`**: production dispatch goes through `perf-changelog.yaml`,
which has no concept of "externally-managed endpoint."

### `.github/workflows/remote-bench-e2e.yml` (new file)
Batch counterpart: takes a JSON array of configs directly (not via
`generate_sweep_configs.py`), matrix-fans to `benchmark-tmpl.yml`, then
`collect-results` + `calc-success-rate` (mirroring `e2e-tests.yml`'s aggregation).
**Deliberately omits** a `trigger-agentic-ingest` job — whether remote-bench results should
auto-dispatch to InferenceX-app's production ingest is the open design question in issue #28,
not something to decide as a side effect of a merge conflict resolution. If upstream's
`e2e-tests.yml` gains a new stage after `calc-success-rate`, evaluate on purpose whether
`remote-bench-e2e.yml` should get it too — don't copy it reflexively.

### `benchmarks/single_node/agentic/dsv2lite_fp8_sglang-remote-bench.sh` and `glm5.2_fp4_sglang-remote-bench.sh` (new files)
Model-agnostic remote-bench recipe bodies (identical to each other by design — remote-bench
never launches a server, so there's no hw-specific tuning to encode). Low conflict risk
since upstream has no equivalent files, unless upstream independently adds its own
`-remote-bench.sh` convention — if so, reconcile naming/contract by hand, don't silently
pick one side.

### `configs/runners.yaml`
- `labels:` block gained `cluster:remote-bench: [bench-client_01]`. Deliberately **no**
  matching `hardware:` entry — agentic hardware validation only requires entries that are
  actually referenced elsewhere, and remote-bench's hardware is self-reported per-run, not a
  fixed cluster spec. If upstream reorganizes `runners.yaml`'s structure, re-add just the
  label line, not a hardware block.

### `runners/launch_bench-client.sh` (new file)
Non-GPU controller launcher — no docker/GPU-mount setup unlike every other
`runners/launch_*.sh`, and overrides `RESULT_DIR`/`INFMAX_CONTAINER_WORKSPACE` to real host
paths since `benchmark-tmpl.yml` otherwise assumes a docker bind-mount
(`-v "$GITHUB_WORKSPACE:/workspace"`) that doesn't exist on this bare-metal box.

### `.claude/skills/create-remote-bench/SKILL.md` (new file)
Not code — no merge-conflict risk from upstream, but keep it in this list because it
documents the *rationale* behind several of the entries above (esp. why
`REMOTE_MAX_CONTEXT_LENGTH` is required and why it isn't automatically safe at the model's
full context window — confirmed by a real incident, not theoretical).

## Known non-code gap (not a merge risk, but relevant context)

No `REMOTE_RESET_URL` has ever actually been configured for the `sglang-vanilla` dev
endpoint — every remote-bench smoke/e2e run so far shares one persistent, never-reset
engine, so throughput/latency numbers from those runs are cache-warm (~92-94% prefix cache
hit rate observed), not clean baselines. Not a fork-vs-upstream concern, just don't mistake
those numbers for real perf data later.
