# 2. Keep both agentic benchmark paths; do not retire the fork or drop the submodule

Date: 2026-06-22

## Status

Accepted. Partially supersedes [ADR-0001](0001-agentic-on-official-aiperf.md) — the
"retire the fork scenario / drop the `utils/aiperf` submodule" decision is reversed.
The "use official mooncake_trace for our own datasets" decision stands.

## Context

ADR-0001 assumed we did **not** need numeric comparability with SemiAnalysis's
published agentic leaderboard, and on that basis decided to retire the fork's
`inferencex-agentx-mvp` scenario and drop the `utils/aiperf` submodule. That
retirement was never executed: `dev` still carries the submodule
(`vngcloud/aiperf` @ `cjq/weka-live-assistant-responses`), and the
`exp/minimax-2.5-sglang-8xh200-semianalysis_cc_traces_weka` branch ran the fork
scenario successfully on H200.

The requirement has since changed. We now want **two** agentic scenarios runnable as
CI gates on `dev`:

- **mooncake (our own prod-derived traces):** minimax claude-code, and later gemma
  chat/RAG. Replayed via official AIPerf `--custom-dataset-type mooncake_trace
  --input-file …`, committed in-repo.
- **agentx-weka (SemiAnalysis comparability):** the fork's `inferencex-agentx-mvp`
  scenario + public `semianalysis_cc_traces_weka` corpus, fetched at runtime.

These run through two **disjoint** pipelines that share only `benchmark_lib.sh` and the
workflow/matrix plumbing:

| | mooncake | agentx-weka |
|---|---|---|
| aiperf call | `run_aiperf_benchmark` → `aiperf_adapter.py` (wraps `aiperf profile --custom-dataset-type mooncake_trace --input-file`) | `build_replay_cmd` → `aiperf profile --scenario inferencex-agentx-mvp --public-dataset` |
| result producer | `aiperf_adapter.py` → InferenceX result JSON | `write_agentic_result_json` → `process_agentic_result.py` |
| dataset | committed JSONL, `--input-file` (optionally `#N` subset) | public HF corpus, fetched at runtime |
| adapter used? | yes | **no** |

The `inferencex-agentx-mvp` scenario hard-binds to the weka loader
(`require_loader=("semianalysis_cc_traces_weka","weka_trace")` in
`src/aiperf/common/scenario/inferencex_agentx_mvp.py`) and rejects `mooncake_trace`,
so the two paths cannot trivially share one invocation.

## Decision

Keep **both** agentic result paths permanently and keep the `utils/aiperf` fork
submodule. The mooncake datasets go through the **adapter** path; agentx-weka stays on
the **scenario** path, untouched. Do not unify them.

We explicitly considered unifying — adding a new `inferencex-mooncake-agentic`
`ScenarioSpec` (`require_loader=("mooncake_trace",)`) so the minimax/gemma traces would
flow through the same `write_agentic_result_json` path and the adapter's mooncake
branch could be deleted — and **rejected it**: it edits the fork (new re-test surface),
forces a single param-lock + cache-bust semantics onto recorded traces whose fidelity
differs from weka's live-assistant threading, and buys nothing the team needs right
now. The two paths answer different questions; keeping them separate is cheaper than
one over-general path.

### Hardware constraint (current)

H200 runners are **decommissioned**. The only GreenNode runner available is
`h100-greennode_00` — a single H100 80GB — which cannot serve MiniMax-M2.5. The first
cut is therefore a **plumbing smoke test of the mooncake/adapter path**, not a capacity
run:

- model `Qwen/Qwen3-4B-Instruct-2507` (vLLM, TP=1, 1×H100 80GB)
- concurrency ladder `[4, 8]` (→ 2 fan-out jobs)
- `benchmark-duration` 90s

MiniMax-M2.5 8×H200 capacity runs (full ladder `[1,4,8,16,24,32,64]`, the capacity
model in CLAUDE.md) are **deferred until H200 capacity returns**.

## Consequences

**Positive**
- agentx-weka comparability retained; our own prod-trace flow added; both gated in CI.
- No fork edit, no re-test of the proven weka path.

**Negative / accepted**
- Two result shapes and two post-processing paths to maintain.
- SLA/goodput is **not** surfaced in the result JSON. We pass `--goodput` only as an
  inert placeholder for now; the team pulls the **retained raw AIPerf artifact** and
  computes SLA (tok/s/user, goodput) offline. **Raw-artifact retention is therefore a
  hard requirement, not a nicety.**

---

## Handoff — implementation plan (next session: read, then plan code)

Branch off `dev` (no push to `main`). Port from
`origin/exp/minimax-2.5-sglang-8xh200-semianalysis_cc_traces_weka` (a strict superset
of `dev`, merge-base == `dev` HEAD).

### Port strategy = curated

Take MODIFIED infra files **wholesale** (they carry both mooncake + weka refinements,
already co-tested): `benchmarks/benchmark_lib.sh`, `utils/bench_serving/aiperf_adapter.py`,
`utils/matrix_logic/generate_sweep_configs.py`, `utils/matrix_logic/validation.py`,
`utils/process_result.py`, `utils/summarize.py`, `.github/workflows/benchmark-tmpl.yml`,
`.github/workflows/e2e-tests.yml`, `.github/workflows/run-sweep.yml`,
`.github/configs/nvidia-master.yaml`, `utils/process_agentic_result.py` (+ their tests).

Add net-new: `benchmarks/single_node/agentic/minimaxm2.5-weka_fp8_h100_sglang.sh`,
`benchmarks/single_node/qwen3-4b-2507_bf16_h100_vllm.sh`,
`benchmarks/single_node/agentic/patches/aiperf-skip-nonfinite-server-metrics.patch`.

**Skip the cruft:** the 8 `agentic-coding-*config*.jsonl` variant datasets, the
`qwen3.5-4b-smoke.jsonl`, the `minimaxm2.5-agentic_fp8_h100_vllm.sh` variant, the 3
`docs/AGENTIC_*_VI.md` exp docs. (The minimax sglang script is H200/8×GPU — keep it
out of the first cut; it can't run on 1×H100.)

### Work items

1. **Commit the dataset.** Copy `InferenceOptiAIPlan/workload-gen/traces/minimax_claude_code_prod_v3.jsonl`
   (6.0 MB, 19,662 lines) → `benchmarks/single_node/agentic/datasets/`. Full file;
   use the `#N` suffix (already supported by the qwen smoke script) for the smoke run.

2. **Plumb placeholder SLA flags.** Add `--goodput` (and the other canonical-command
   flags: `--inter-turn-delay-cap-seconds`, `--temperature`,
   `--dataset-sampling-strategy`, `--benchmark-grace-period`, `--workers-max`)
   pass-through args to `aiperf_adapter.py` + `run_aiperf_benchmark` +
   `run_client_benchmark`. **Do not activate them** in the smoke config — wire the hook,
   leave them unset. (Reference canonical command in the session notes / CLAUDE.md.)

3. **Fix raw-artifact retention (hard requirement).** The adapter writes the raw
   artifact to `results/<result_filename>_aiperf/profile_export_aiperf.json`
   (`aiperf_adapter.py` `run_aiperf`, ~line 127), but the "Upload agentic raw results"
   step in `benchmark-tmpl.yml` only collects `results/trace_replay/…`. **Extend that
   upload step's `path:` list** to also include
   `results/*_aiperf/profile_export_aiperf.{json,csv,jsonl}` and
   `results/*_aiperf/profile_export_aiperf_timeslices.json`. Leave the weka
   `trace_replay/` paths as-is.

4. **Add the smoke config** to `.github/configs/nvidia-master.yaml` (schema mirrors the
   existing qwen smoke entries), targeting the only live runner:

   ```yaml
   qwen3-4b-2507-bf16-h100-greennode-vllm-smoke:
     model: Qwen/Qwen3-4B-Instruct-2507
     model-prefix: qwen3-4b-2507    # → reuses qwen3-4b-2507_bf16_h100_vllm.sh
     runner: h100-greennode_00
     scenarios:
       agentic-replay:
       - input-file: benchmarks/single_node/agentic/datasets/minimax_claude_code_prod_v3.jsonl#<N>
         benchmark-client: [aiperf]
         - { tp: 1, conc-list: [4, 8] }
   ```
   Append the matching `perf-changelog.yaml` entry (append-only, preserve whitespace;
   `scenario-type: [agentic-replay]`). Dispatch with `duration-override=90`.

   **Smoke caveat:** Qwen3-4B on 1×H100 has a modest context window — set
   `MAX_MODEL_LEN` (e.g. 32768) and choose the `#N` subset so replayed sessions don't
   exceed it (the minimax coding trace contains long/128k sessions that will error
   otherwise). Pick short sessions or a small N for the smoke.

5. **Dispatch on a branch, not `main`** (agentic-replay routing is missing on `main` —
   see the InferenceX dispatch gotcha). Confirm: 2 fan-out jobs (`conc-4`, `conc-8`),
   raw artifact uploaded under `agentic_<RESULT_FILENAME>`, adapter result JSON present.

6. **Follow-ups (not this cut):** gemma chat (`short.jsonl`) + RAG (`blend_prod.jsonl`)
   — note gemma reuse is per-*user*, single-turn, **not** multi-turn agentic (see
   `workload-gen/GEMMA-RAG-DATASET.md`), so its invocation shape differs. MiniMax-M2.5
   8×H200 capacity runs once H200 returns.
