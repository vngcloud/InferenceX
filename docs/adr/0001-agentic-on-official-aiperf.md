# 1. Run agentic benchmarks on official AIPerf, retire the fork scenario

Date: 2026-06-02

## Status

Accepted, partially superseded by [ADR-0002](0002-keep-both-agentic-paths.md). The
decision to **retire the fork scenario and drop the `utils/aiperf` submodule** is
reversed (the team now wants agentx-weka kept for SemiAnalysis comparability). The
decision to run our **own** datasets on official AIPerf `mooncake_trace` still stands.

## Context

InferenceX's agentic (multi-turn coding-agent replay) benchmark currently depends on
the `cquil11/aiperf` **fork**, installed via `install_agentic_deps` from the
`utils/aiperf` submodule. The agentic path runs `aiperf profile --scenario
inferencex-agentx-mvp --public-dataset semianalysis_cc_traces_weka`. Both the
scenario plugin and the dataset exist **only** in the fork.

The AIPerf-as-benchmark-client work (fixed-sequence path) deliberately installs the
**official** `ai-dynamo/aiperf` v0.9.0 from PyPI via `ensure_aiperf` (isolated venv).
This created a conflict: the submodule was repointed fork → official, which silently
breaks every `agentic/*.sh` script (unknown scenario / missing dataset). Merging that
repoint to `main` is a latent landmine.

We need agentic benchmarks for three goals (in priority order from the team):
KV/prefix-cache reuse fidelity, throughput/latency under realistic multi-turn load,
and self-control over the dataset + a low-maintenance install. We do **not** require
direct numeric comparability with SemiAnalysis's published agentic leaderboard.

Investigation of the official AIPerf v0.9.0 source found that the capability we
believed was fork-exclusive — threading the server's own generated response back into
the next turn's context for realistic multi-turn KV reuse — **is present in official**:
`openai_chat.build_assistant_turn()` captures the server's full assistant message
(text **and** `tool_calls`) so a "FORK-mode DAG child" inherits the parent's real
response. Official also has native multi-turn session support, sticky routing, and
inter-turn delay controls. The fork's `inferencex-agentx-mvp` scenario is therefore a
**preset bundle** (locked params + the SemiAnalysis weka corpus), not a unique
capability.

## Decision

Run agentic benchmarks on **official AIPerf v0.9.0** using its native `mooncake_trace`
custom dataset + multi-turn FORK-mode replay (`build_assistant_turn`). Retire the
fork's `inferencex-agentx-mvp` scenario path. Use the team's own dataset generator
(`inference-benchmark/aiperf-service/datasets/agentic-code/`) as the canonical agentic
dataset source.

Consequently, **drop the `utils/aiperf` fork submodule** — both fixed-seq and agentic
share one official PyPI install via `ensure_aiperf`.

## Consequences

**Positive**
- One install path (official PyPI) for both fixed-seq and agentic; the submodule and
  its per-upgrade maintenance tax disappear.
- Removes the merge blocker: no fork-vs-official submodule conflict on `main`.
- Full control over the agentic dataset (the team's generator already models
  prefix-cache layers, agentic-vs-human inter-turn delays, reset/restart behavior).
- KV/prefix-cache fidelity preserved via official FORK-mode DAG replay (with the bonus
  that official threads `tool_calls`, not just text — relevant for coding agents).

**Negative / accepted trade-offs**
- Loss of **direct numeric comparability** with SemiAnalysis's published agentic
  numbers (different dataset + param lock). Accepted: not a stated goal.
- The team now **owns dataset fidelity** — there is no upstream "blessed" corpus.
- If comparability is ever needed, the recovery path is to convert the
  `semianalysis_cc_traces_weka` corpus to `mooncake_trace` JSONL and replay it on the
  same official path — without reintroducing the fork.

## Alternatives considered

- **Inherit the fork scenario** (keep submodule on `cquil11`): gives SemiAnalysis
  comparability and a battle-tested preset, but locks us off official PyPI, perpetuates
  submodule maintenance, and the handoff flags the fork as an uncontrolled MVP.
- **Keep official submodule + quarantine agentic**: would lose agentic entirely, which
  the team requires.
