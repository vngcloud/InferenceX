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

## Features (chronological)

| Feature | PR(s) | Commit (on `main`) | Issue |
|---|---|---|---|
| Remote-bench (BYO endpoint): recipe, workflow, runner, batch/CCU-ladder dispatch, sync-upstream/fork-changelog docs | [#27](https://github.com/vngcloud/InferenceX/pull/27), [#30](https://github.com/vngcloud/InferenceX/pull/30), [#31](https://github.com/vngcloud/InferenceX/pull/31), [#32](https://github.com/vngcloud/InferenceX/pull/32), [#33](https://github.com/vngcloud/InferenceX/pull/33), [#34](https://github.com/vngcloud/InferenceX/pull/34) | one squashed commit — search `git log --oneline` on `main` for "support benchmarking remote/existing inference endpoints" | #26, #28 |

(PR #29 merged `vng-benchmark` wholesale into `main` by accident, pulling in 21 unrelated
commits — reverted via a `main` reset. Not a real feature; noted here only so the PR-number
sequence doesn't look like a gap. #27/#30/#31/#32/#33/#34 all squashed into the single commit
above — everything for this feature is one commit by design, per an explicit request to keep
history minimal ahead of an eventual upstream sync.)

Add a row here whenever a new fork feature lands on `main`. When a feature is later
superseded or upstreamed, don't delete the row — mark it `(resolved, see PR #NNN)` so the
history of what happened stays legible.

## Entries (file-level detail)

### `benchmarks/benchmark_lib.sh`
- **Line ~1767**, `build_replay_cmd()`: `REPLAY_CMD+=" --url ${REMOTE_BASE_URL:-http://localhost:$PORT}"`.
  **Why**: lets a recipe point aiperf at an externally-managed endpoint (`REMOTE_BASE_URL`)
  instead of the locally-launched server on `$PORT`. This is the single hardcode that caused
  the original remote-bench bug (issue #26) before it was fixed — if upstream changes how
  `build_replay_cmd` constructs `--url`, this fallback pattern must be re-applied, not dropped.
- **`run_agentic_replay_and_write_outputs()`**: the replay runs backgrounded with process
  substitution (`$REPLAY_CMD > >(tee …) 2>&1 &` + `wait "$AIPERF_REPLAY_PID"`) instead of
  upstream's foreground `$REPLAY_CMD 2>&1 | tee` + `PIPESTATUS[0]`, with `INT`/`TERM` handlers
  armed around it (snapshotted via `trap -p` and restored after, so recipes that manage a local
  server keep their own handlers). Plus `AIPERF_REPLAY_PID` / `AIPERF_REPLAY_PID_FILE` state vars
  and four helpers: `_write_agentic_replay_pid_file`, `stop_agentic_replay`,
  `_agentic_replay_signal_exit`, `reap_orphan_agentic_replays`.
  **Why**: a cancelled job left the aiperf tree running against production. Actions cancels with
  SIGINT → SIGTERM → SIGKILL, and **bash defers a trap until the current foreground command
  finishes** — verified on bash 3.2 and on the runner's bash 5.1.16: with a foreground pipeline the
  handler fires only when the pipeline ends, so a trap alone is useless without also
  un-foregrounding the replay. `$!` must come from process substitution, not a pipe, or it is
  `tee`'s PID. If upstream rewrites this function, **all three parts must be re-applied together**
  (background + `wait`, the trap pair, the PID file) — any one alone does nothing. Run
  30508401430 is the incident; `utils/test_benchmark_lib.py::test_sigterm_during_replay_kills_the_whole_tree`
  is the regression test.

### `.github/workflows/benchmark-tmpl.yml`
- 6 `workflow_call` inputs + matching `env:` entries, all optional/empty-default:
  `remote-base-url`, `remote-gpu-telemetry-url`, `remote-engine-metrics-url`,
  `remote-reset-url`, `remote-runner-type`, `remote-max-context-length` →
  `REMOTE_BASE_URL`, `REMOTE_GPU_TELEMETRY_URL`, `REMOTE_ENGINE_METRICS_URL`,
  `REMOTE_RESET_URL`, `REMOTE_RUNNER_TYPE`, `REMOTE_MAX_CONTEXT_LENGTH`.
  **Why**: only consumed by `*-remote-bench.sh` recipes; every other recipe just sees them
  empty. If upstream restructures this file's `inputs:`/`env:` blocks, all 6 pairs need to
  be re-added in the same input-name → env-var-name shape, since `remote-bench-e2e.yml`
  references these exact input names.
- **Top of the `&resource-cleanup` anchor body**: `bash runners/reap_orphan_aiperf.sh || true`,
  guarded by `[ -f ... ]`.
  **Why**: the backstop for the same cancellation bug — the one layer that survives SIGKILL of the
  benchmark shell. Placed in the anchor on purpose so the pre-run (`:258`) and post-run
  (`:417`, `if: always()`) steps share it: post-run reaps this job's orphan, pre-run reaps a
  previously cancelled job's before it can double-load the endpoint. The `-f` guard exists because
  the pre-run copy runs **before** `actions/checkout`. If upstream restructures the anchor, re-add
  this **first** in the body (before the docker loop, which can block for minutes).

### `runners/reap_orphan_aiperf.sh` (new file)
Sources `benchmark_lib.sh` and calls `reap_orphan_agentic_replays`. Exports `IS_AGENTIC=0` and
`SCENARIO_TYPE=""` before sourcing — the lib's source-time agentic gate `exit 1`s on an
inconsistent KV pair, and a sourced `exit` would take the cleanup step with it. No `set -e`:
cleanup must never fail a job.

### `.github/workflows/remote-bench-e2e.yml` (new file)
The single remote-bench entrypoint: takes a JSON array of configs directly (not via
`generate_sweep_configs.py`), matrix-fans each to `benchmark-tmpl.yml`, then
`collect-results` + `calc-success-rate` (mirroring `e2e-tests.yml`'s aggregation). A
single-config smoke test is just a one-element array — an earlier standalone
`remote-bench.yml` existed for that case and was removed once this covered both; don't
recreate it. Deliberately standalone from `generate_sweep_configs.py`/`validation.py` — the
master-config schema has no fields for per-run remote endpoint values, and extending it was
explicitly rejected in favor of this. **Deliberately omits** a `trigger-agentic-ingest` job —
whether remote-bench results should auto-dispatch to InferenceX-app's production ingest is
the open design question in issue #28, not something to decide as a side effect of a merge
conflict resolution. If upstream's `e2e-tests.yml` gains a new stage after
`calc-success-rate`, evaluate on purpose whether `remote-bench-e2e.yml` should get it too —
don't copy it reflexively.

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
