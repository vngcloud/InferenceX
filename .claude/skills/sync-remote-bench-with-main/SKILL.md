---
name: sync-remote-bench-with-main
description: Land remote-bench changes onto main, or rebase the remote-bench feature onto a newer main, without dragging in unrelated commits from vng-benchmark. Use before merging any remote-bench-related PR into main, or when vng-benchmark has fallen behind main and needs a rebase.
---

# Land remote-bench on main — cherry-pick, never merge the whole branch

**The rule:** `vng-benchmark` is a long-lived branch with lots of unrelated in-flight work
(GLM-5.2 sweeps, GreenNode recipes, HiCache experiments, docs, ...). It is never safe to
merge the whole branch into `main`. Only cherry-pick the exact commits that belong to
remote-bench.

This actually happened: PR #29 merged all of `vng-benchmark` (25 commits) into `main` to
unblock `remote-bench.yml`'s `workflow_dispatch`, and pulled in 21 unrelated commits along
with it. Had to be cleaned up after the fact by resetting `main` and cherry-picking back just
the 3 real commits. Don't repeat this.

## Landing new remote-bench work on main

1. Identify the exact commit SHAs that belong to remote-bench (not the whole branch tip):
   ```bash
   git log --oneline origin/main..origin/vng-benchmark -- \
     benchmarks/single_node/agentic/*-remote-bench.sh \
     benchmarks/benchmark_lib.sh \
     configs/runners.yaml \
     .github/workflows/remote-bench.yml \
     .github/workflows/benchmark-tmpl.yml \
     runners/launch_bench-client.sh \
     .claude/skills/create-remote-bench/SKILL.md
   ```
2. Branch off `main` (not `vng-benchmark`) and cherry-pick just those SHAs, oldest first:
   ```bash
   git checkout -b feat/remote-bench-<topic> origin/main
   git cherry-pick <sha1> <sha2> ...
   ```
3. Before opening the PR, confirm no file outside that list changed:
   ```bash
   git diff --stat origin/main feat/remote-bench-<topic>
   ```
   If something unexpected shows up, you cherry-picked a commit that touched more than
   remote-bench — drop it and cherry-pick a narrower one instead.
4. Open the PR against `main` normally.

## Rebasing vng-benchmark itself onto a newer main

`main` gets fast-forwarded from `upstream/main` periodically
(`git push origin upstream/main:main`, no PR — those commits are already reviewed upstream).
When that happens, `vng-benchmark` falls behind:

```bash
git fetch origin main vng-benchmark --quiet
git log --oneline origin/vng-benchmark..origin/main | wc -l   # 0 = nothing to do
```

If non-zero, rebase `vng-benchmark` itself (this branch *is* meant to carry everything, so a
normal rebase is fine here — the cherry-pick rule above is only for what leaves the branch
and lands on `main`):

```bash
git checkout -b sync/vng-benchmark-<date> origin/vng-benchmark
git rebase origin/main
git push origin sync/vng-benchmark-<date>:vng-benchmark --force-with-lease
```

Likely conflicts land in whatever remote-bench touched: `benchmark_lib.sh` (the
`REMOTE_BASE_URL` override in `build_replay_cmd`), `configs/runners.yaml`
(`cluster:remote-bench`), `.github/workflows/benchmark-tmpl.yml` (the `remote-*` inputs). If
upstream changed any of these, read the conflict instead of blindly taking "ours" — a
renamed `workflow_call` input fails silently at dispatch time, not at rebase time.

After rebasing, re-run the SSH smoke-test loop (`/create-remote-bench` skill, section 6)
before trusting it — a clean rebase doesn't prove the merged result still runs.
