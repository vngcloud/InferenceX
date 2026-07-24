---
name: sync-upstream
description: Fetch and merge upstream (SemiAnalysisAI/InferenceX) into this fork's main, resolve conflicts, verify, and open a PR. Invoke this when actually doing an upstream sync — use alongside fork-changelog, which lists what fork-only edits must survive the merge.
---

# Sync upstream into main

`main` here is **not** a pure upstream mirror anymore — fork-specific commits (remote-bench,
etc.) live directly on it. That means syncing upstream is a real merge with real conflict
risk, not a fast-forward push. Read `fork-changelog` first — it lists every fork-only edit
and why, so you know which conflicts to resolve carefully instead of blindly taking one side.

## 1. Check how far behind

```bash
git fetch upstream main --quiet
git fetch origin main --quiet
git log --oneline origin/main..upstream/main | wc -l   # commits we're missing
git log --oneline upstream/main..origin/main | wc -l   # our fork-only commits (cross-check
                                                         # this count/list against fork-changelog)
```

If the second command's commit list doesn't match what `fork-changelog` describes, stop and
reconcile that discrepancy first — the changelog is stale or something landed on `main`
outside the expected process.

## 2. Branch and merge (don't rebase, don't force-push main)

`main`'s fork commits are already public (merged via normal PRs). Rebasing them onto
upstream would rewrite published history for no benefit — merge instead:

```bash
git checkout -b sync/upstream-<date> origin/main
git merge upstream/main
```

## 3. Resolve conflicts against fork-changelog

For each conflict, check `fork-changelog` for that file before resolving:

- **`benchmarks/benchmark_lib.sh`**: if the conflict is at the `REMOTE_BASE_URL` fallback in
  `build_replay_cmd`, re-apply `${REMOTE_BASE_URL:-http://localhost:$PORT}` on top of
  whatever upstream changed around it — don't just take upstream's side and drop the
  fallback.
- **`.github/workflows/benchmark-tmpl.yml`**: if upstream restructured `inputs:`/`env:`,
  re-add all 6 `remote-*` input/env pairs listed in `fork-changelog` in the same shape.
- **`configs/runners.yaml`**: re-add just the `cluster:remote-bench` label line — no
  `hardware:` entry (deliberate, see `fork-changelog`).
- New files (`remote-bench.yml`, `remote-bench-e2e.yml`, the two `-remote-bench.sh` recipes,
  `launch_bench-client.sh`) shouldn't conflict unless upstream independently added something
  with the same name — if that happens, reconcile by hand, don't let git silently pick one.

If a conflict is in a file `fork-changelog` doesn't mention, it's an upstream-only change —
just take upstream's version.

## 4. Verify before opening the PR

A clean merge doesn't prove the fork features still work:

- Confirm the 6 `remote-*` inputs in `remote-bench.yml`/`remote-bench-e2e.yml` still line up
  end-to-end with `benchmark-tmpl.yml`'s input names (a silent rename only breaks at dispatch
  time, not at merge time).
- Grep for the `REMOTE_BASE_URL` fallback in `benchmark_lib.sh` — confirm it's still there.
- If there's time, re-run the SSH smoke-test loop (`create-remote-bench` skill, section 6)
  against a real endpoint once.

## 5. Open the PR

```bash
git push -u origin sync/upstream-<date>
gh pr create -R vngcloud/InferenceX --base main --head sync/upstream-<date> \
  --title "sync: merge upstream/main into main" \
  --body "Merges upstream through <upstream-sha>. Conflicts resolved per fork-changelog: <list any that needed hand-resolution>."
```

Goes through the normal review flow (branch protection requires it) — no force-push, no
branch-protection toggling needed for a routine sync like this.

## 6. After merging

Update `fork-changelog` if the sync changed the shape of any entry — e.g. upstream added a
native equivalent to something we patched in, and the fork-only edit can be marked resolved
or removed instead of carried forward into the next sync.
