# 3. Give the mooncake path its own clean-0.9.0 aiperf fork submodule

Date: 2026-06-22

## Status

Accepted. Partially supersedes [ADR-0001](0001-agentic-on-official-aiperf.md): the
"mooncake runs on **official** AIPerf (PyPI 0.9.0), low-maintenance install" decision is
reversed. The mooncake path now installs aiperf from a team-owned **clean fork of
v0.9.0** so we can patch aiperf internals. ADR-0002's two-disjoint-paths structure
stands and is reinforced.

## Context

We want to modify aiperf behaviour **inside** aiperf (request construction from
`mooncake_trace`, assistant-turn threading, loadgen/loader internals) for the mooncake
path — not something expressible in `aiperf_adapter.py` or the launch script. Today the
mooncake path installs **stock PyPI `aiperf==0.9.0`** via `ensure_aiperf` (isolated venv),
which leaves no place for our own code.

The existing `utils/aiperf` submodule cannot host this: it is pinned to the
SemiAnalysis/NVIDIA **agentx line** (`vngcloud/aiperf` @ `7d880a1e`, 81 commits ahead of
upstream `main`, **zero** GreenNode-authored commits) and the weka path
(`install_agentic_deps` → `pip install -e`) **requires** those commits (the
`inferencex-agentx-mvp` scenario + weka loader). Replacing it with clean 0.9.0 would
break weka and violate ADR-0002.

## Decision

Add a **second** aiperf submodule, `utils/aiperf-mooncake`, tracking branch
`benchtool/aiperf-0.9.0` — a clean branch cut from tag **`v0.9.0`** (no agentx commits)
in **`thangquang09/aiperf`**, a personal fork of `ai-dynamo/aiperf`. (We lack push access
to `vngcloud/aiperf`; the personal fork is the interim home — see alternatives.) The
mooncake launch script sets
`AIPERF_SOURCE_DIR=$INFMAX_CONTAINER_WORKSPACE/utils/aiperf-mooncake`, so `ensure_aiperf`
installs our fork into the isolated venv via `pip install <dir>` (non-editable). The
serving image (vLLM/SGLang/…) is untouched; the client venv stays decoupled from serving
deps.

The two submodules are disjoint by purpose:

| | `utils/aiperf` (weka) | `utils/aiperf-mooncake` (mooncake) |
|---|---|---|
| repo | `vngcloud/aiperf` | `thangquang09/aiperf` (personal, interim) |
| base | SemiAnalysis agentx line | clean upstream `v0.9.0` |
| install | `pip install -e` (global, `--ignore-installed`) | `pip install <dir>` into `/tmp/aiperf-venv` (isolated) |
| owner | mirror of SemiAnalysis | GreenNode patches |
| wired via | `install_agentic_deps` | `AIPERF_SOURCE_DIR` in launch script |

Weka is **unaffected**: `utils/aiperf` is unchanged, and CI only needs read access to
`vngcloud/aiperf` (which we have) to clone it at the pinned commit.

### Considered alternatives

- **Patch in `aiperf_adapter.py` / launch script** — rejected: the change is aiperf-internal.
- **Reuse the existing `utils/aiperf` submodule** — rejected: it's the agentx line weka needs; sharing couples mooncake to weka's fork and contradicts ADR-0002.
- **Clean branch inside `vngcloud/aiperf`** — preferred home, but **blocked**: no push access to that org repo.
- **Org-owned repo (`vngcloud/aiperf-mooncake` or push grant)** — the proper long-term home; deferred (needs an admin). Re-pointing the submodule URL there later is cheap.

## Consequences

- We take on **fork maintenance** (rebasing onto future aiperf releases) — the burden ADR-0001 deliberately avoided.
- Mooncake numbers no longer come from stock aiperf; **comparability-to-official is lost** for this path. (ADR-0001 already recorded we don't require SemiAnalysis leaderboard comparability.)
- Two aiperf submodules to keep pinned. A future reader sees two and must read this ADR to know why.
- Initial fork is clean 0.9.0 with **no GreenNode patches yet** — wiring is set up ahead of the first real patch (intentional).
- The mooncake fork currently lives under a **personal account** (`thangquang09/aiperf`) — a bus-factor/ownership risk for a team CI gate. Accepted as interim; reversible by re-pointing the submodule URL to an org repo once available.
