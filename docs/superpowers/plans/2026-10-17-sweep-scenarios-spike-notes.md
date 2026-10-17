# Sweep-scenarios spike notes (2026-10-17/18)

Verified AIPerf facts for the sweep-scenarios plan (Task 1 spike). Everything
below was verified empirically against the vendored AIPerf at `utils/aiperf`
(pinned) and its mock server, on this workstation. Later tasks quote these
answers by the `UPPER_SNAKE` names.

Environment: AIPerf mock server (scheduler mode, knee at concurrency 32) +
`Qwen/Qwen3.8-27B` tokenizer (local HF cache, no network). Reference scenario:
`scenarios/sweep-isl8k-osl512-cache50.yaml` (committed with this note).

## Mock server start command (Step 1)

Run from `utils/aiperf` (the mock server must be installed into the aiperf
venv first: `uv pip install -e "tests/aiperf_mock_server"`; running it from
its own standalone project env fails because it imports `aiperf`):

```bash
cd utils/aiperf
uv run aiperf-mock-server \
  --port 8877 \
  --tokenizer Qwen/Qwen3.8-27B \
  --record-requests /tmp/sweep-spike/requests.jsonl \
  --scheduler-enabled \
  --scheduler-step-ms 5 \
  --scheduler-max-batch-size 32 \
  --scheduler-max-prefill-chunks-per-step 8 \
  --scheduler-prefill-chunk-tokens 512
```

Why these knobs: the open-loop latency model always passes the SLO, so the
batched step scheduler is required to give the search a genuine knee. With
`max_batch_size=32`, `step_ms=5`, decode ITL is ~5 ms at concurrency <= 32 and
scales linearly past it, so the `inter_token_latency p90 < 50 ms` SLA breaks at
~320. The search converged with `feasible_max=256`, `infeasible_min=264`
(first observed breach at 320 with ITL p90 = 61.3 ms). `--record-requests`
captures per-request wire ISL/OSL (forces `--workers=1`, needs a real
tokenizer).

## Validation command that worked (Step 2)

```bash
cd utils/aiperf
uv run aiperf profile \
  --config scenarios/sweep-isl8k-osl512-cache50.yaml \
  --url http://localhost:8877 \
  --model Qwen/Qwen3.8-27B \
  --tokenizer Qwen/Qwen3.8-27B \
  --no-gpu-telemetry
```

Full run wall clock: ~70 min (16 iterations: 60 s first-probe warmup each +
up to 300 s profiling each). Artifacts land in
`utils/aiperf/artifacts/Qwen_Qwen3.8-27B-openai-chat-concurrency1/`.

## SWEEP_KEY

**Answer: top level.** `sweep:` is a sibling of `benchmark:` on the
`AIPerfConfig` envelope (`src/aiperf/config/config.py`, field
`AIPerfConfig.sweep: SweepConfig | None`), NOT `benchmark.sweep`.

Verified by: the committed scenario YAML (top-level `sweep:` block) validated
and ran the full 16-iteration search; `search_history.json` `config.search_space[0].path`
echoes `phases.profiling.concurrency`. A `benchmark.sweep` placement would be
rejected (`extra="forbid"` on `BenchmarkConfig`).

## ISL_INCLUDES_PREFIX

**Answer: false — the shared prefix is added ON TOP of `isl`.** Total wire
prompt tokens = `isl` + `prefix_prompts.shared_system_length` + ~50 chat-template
tokens.

Verified by: the mock server's request recorder (tokenizes the assembled
messages with the real Qwen chat template) recorded `isl` = 8245-8249 (mean
~8246) for `isl: 4096` + `shared_system_length: 4096`; 8192 + ~54 template
tokens. Command: inspect `/tmp/sweep-spike/requests.jsonl` (one JSON per
request, `isl` field).

CAVEAT for later tasks: AIPerf's own `input_sequence_length` metric (in each
probe's `profile_export.jsonl` / `profile_export_aiperf.json`) counts only the
unique per-entry part (mean 4102.4 for the same run) — it does NOT include the
shared prefix. Do not use `input_sequence_length` to reason about total prompt
tokens when `prefix_prompts` is set; use `isl + shared_system_length` (+
~50 template tokens, negligible vs 8k+).

So for grid target ISL T with cache share p: `isl = T - round(T*p)` (unique
part), `shared_system_length = round(T*p)`. E.g. 8192/50% -> 4096 + 4096,
65536/90% -> 6554 + 58982.

## PREFIX_FIELD

**Answer: `benchmark.datasets[N].prefix_prompts.shared_system_length`**
(two-part form; `user_context_length` is the optional second field). The pool
form is `prefix_prompts.{pool_size, length}` and is mutually exclusive with
`shared_system_length`/`user_context_length` (validator in
`src/aiperf/config/dataset/content.py`).

Verified by: the committed scenario uses `prefix_prompts.shared_system_length:
4096` and produced the shared-prefix wire prompts above; the mock recorder
shows one byte-identical system message across all 1093 captured baseline
requests (capture via a log proxy, see CACHE0_FIELD).

## SLO_OVERRIDE

**Answer: neither — CLI `--search-sla` is SILENTLY IGNORED when the YAML
`--config` already carries a `sweep:` block.** It neither replaces nor adds to
YAML `sweep.sla_filters`; no warning is logged.

Verified by: re-ran the scenario (fast variant) with
`--search-sla "time_to_first_token:avg:lt:4000"` added; the resulting
`search_history.json` `config.sla_filters` contained only the two YAML filters
(TTFT avg < 8000, ITL p90 < 50) — the 4000 filter never appeared, and the run
log had zero warnings.

Mechanism (code-reading, `src/aiperf/config/flags/_converter_optionals.py`
`build_sweep`): the CLI-side sweep dict is only built when `--search-space` or
another *defining* `--search-*` flag is set; `--search-sla` alone builds no
sweep, so nothing is merged over the YAML block. (With a full CLI
`--search-space` invocation the CLI-built sweep would deep-merge over the
YAML, and list fields like `sla_filters` are replaced wholesale by
`deep_merge` — but that path is irrelevant to this plan, which authors YAML.)

**Consequence for the plan: SLOs must be authored in the YAML
`sweep.sla_filters` (per scenario). CLI `--search-sla` cannot override them.**

## ARTIFACT_LAYOUT

For an `adaptive_search` run with artifact dir `<A>` (default
`utils/aiperf/artifacts/<Model>-openai-chat-concurrency<maxC>/`):

- `<A>/search_history.json` — the primary artifact: `config` (planner,
  objectives, search_space, sla_filters), `iterations` (one per probe, with
  `variation_values`, `objective_values`, `feasible`), `best_trials`,
  `boundary_summary` (`feasible_max`, `infeasible_min` with `first_breach`,
  `boundary_type`, `binding_constraint`), `convergence_reason`.
- `<A>/search_iter_NNNN/profile_runs/run_MMMM/profile_export_aiperf.json`
  (also `.csv`, `.jsonl`, `profile_export_console.txt`) — per-probe metrics.
  This is "the winning probe's metrics" a result writer consumes; the winning
  probe is `boundary_summary.feasible_max.iteration_idx` (here iter 0008,
  concurrency 256).
- `<A>/search_iter_NNNN/profile_runs/run_MMMM/phases/{warmup,profiling}/profile_export_aiperf.json`
  — per-phase splits (warmup is excluded from results).
- `<A>/search_iter_NNNN/profile_runs/run_MMMM/run_config.json` — the mutated
  per-probe config (shows the planner-injected `requests` and warmup phase).
- `<A>/aggregate/profile_export_aiperf_aggregate.{json,csv}` — a confidence
  aggregate that pools ALL search iterations as if they were repeated runs of
  one benchmark (`num_profile_runs: 16`). CAUTION: mixing all concurrency
  points; not a per-variation sweep table.
- `<A>/logs/aiperf.log`.

**DEVIATION from the brief/docs: there is NO
`<A>/sweep_aggregate/profile_export_aiperf_sweep.json` for `adaptive_search`
runs.** The sweep-aggregate exporter only runs when `plan.is_sweep` (grid/zip/
scenarios expansion, `len(plan.configs) > 1`); an adaptive search expands to a
single config (`plan.is_adaptive_search`), so the multi-run path takes the
plain confidence-aggregate branch (`src/aiperf/cli_runner/_multi_run.py`).
Fixtures therefore capture `search_history.json` + the winning probe's
`profile_export_aiperf.json` instead of a sweep aggregate.

Verified by: `find <A> -name '*.json' | sort` after the full run (output
recorded above) and by reading `_multi_run.py` / `_sweep_aggregate.py`.

## ENDPOINT

**Answer: chat.** `benchmark.endpoint.type: chat`, `path:
/v1/chat/completions`, `streaming: true`, `useServerTokenCount: true`. The
mock server serves this endpoint and the full search ran against it; the
recorder logged `endpoint: /v1/chat/completions`, `stream: true` on every
request. Streaming is required for the TTFT/ITL SLA filters (they are
streaming-only metrics). `useServerTokenCount: true` works against the mock
(it emits usage in the stream).

## CACHE0_FIELD

**Answer: for `type: synthetic` datasets (default timing mode, no scenario
lock), `prompts.cache_bust.target` is a NO-OP — the cache-bust marker is only
minted by the `agentic_replay` timing strategy (trace datasets). Cache-0 must
be expressed by omitting `prefix_prompts` entirely.**

Verified by wire capture: ran the same small scenario three times through a
log-only capture proxy (records every request body; canned SSE responses):

1. baseline (`prefix_prompts.shared_system_length: 4096`): all 1093 requests
   carry one byte-identical system message (shared 4096-token prefix), 20
   distinct user prompts (entries cycle).
2. `prompts.cache_bust.target: system_prefix` added: request bodies are
   byte-identical to baseline — NO marker is injected anywhere. Code path:
   markers are keyed off trajectory/recycle state that only exists under
   agentic replay (`src/aiperf/timing/strategies/cache_bust.py`,
   `branch_orchestrator.py`); the worker applies a marker only when
   `credit.cache_bust_marker is not None`, which never happens for synthetic.
3. `prefix_prompts` omitted: each request has a single user message, 20
   distinct prompts, no shared system prefix — this is the cache-0 shape.

**Consequence for Task 3: cache 0 => no `prefix_prompts` block (and do NOT
add `cache_bust` — it does nothing here); cache 50/90 =>
`prefix_prompts.shared_system_length` only.**

YAML location for completeness (works only under trace/agentic scenarios):
`benchmark.datasets[N].prompts.cache_bust.target`, values `none |
system_prefix | system_suffix | first_turn_prefix | first_turn_suffix`.

## MIN_REQUESTS_FIELD

**Answer: `phases[N].requests` (the profiling phase's request cap), with
`sweep.sla_precision` as the preset form.** Semantics: "stop after this many
requests sent" — a CAP, not a completed-request floor. There is no explicit
"at least N completed requests" gate in AIPerf.

- `sla_precision: tight|normal|coarse` maps to 10000 / 1000 / 300 requests
  per probe (`Environment.SEARCH_PLANNER.SLA_PRECISION_REQUESTS`,
  `src/aiperf/common/environment.py`), injected as `phases.profiling.requests`
  by the smooth_isotonic planner — ONLY when the phase does not set `requests`
  explicitly; an explicit `phases.profiling.requests` always wins
  (`_shared_warmup.apply_sla_precision`).
- Verified at runtime: the probe's `run_config.json` shows the planner
  injected `requests: 1000` (from `sla_precision: normal`) plus a prepended
  60 s warmup phase (first-probe warmup floor; replicate probes 15 s).
- The spec's floor "100 completed requests within 300 s" is expressed as
  `duration: 300` + enough `requests`: `sla_precision: normal` (1000) is
  duration-capped at low concurrency (observed: 115 completed requests in
  300 s at concurrency 1 — floor met). `coarse` (300) also clears 100 on any
  SUT fast enough to complete 100 in 300 s; if a hard floor matters, set
  `phases[].requests` explicitly (>= 100) — but note it caps the probe early
  on fast SUTs, reducing the sample size.

FLAG: AIPerf has no "minimum completed requests" enforcement. If a probe
completes < 100 requests in 300 s (slow SUT + high ISL), the search still
scores it. Downstream consumers should treat request_count from the probe
export as the sample-size signal.

## Scenario finding (stop-condition (a) check)

`benchmark.scenario: inferencex-agentx-mvp` REJECTS synthetic datasets — hard,
unbypassable even with `unsafeOverride: true` (it also demands duration >=
900 s and `cache_bust: first_turn_prefix`). Observed error: "got synthetic
(CLI default...); --unsafe-override cannot bypass a missing weka loader".

The alternative needs NO fork changes: omit `scenario:` entirely (the
committed scenario does). This is a finding, not a blocker: the sweep
scenarios are plain synthetic concurrency benchmarks; the agentx scenario
lock exists for trace-replay workloads.

## Field-name deviations from the brief's draft YAML

1. `benchmark.scenario: inferencex-agentx-mvp` — removed (rejects synthetic;
   see above).
2. `sweep.objectives` — REQUIRED (`min_length=1`); the brief's draft omitted
   it. Added `- {metric: output_token_throughput, stat: avg, direction:
   maximize}` (same objective the `max-concurrency-under-sla` recipe expands
   to; smooth_isotonic rejects >1 objectives).
3. `benchmark.endpoint.extra.ignore_eos: true` — added. Without it AIPerf
   warns "OSL cannot be guaranteed" and the mock would generate 0.8-1.2x
   prompt-length outputs; with it the mock generates exactly `max_tokens`
   (512), keeping OSL exact.

Everything else in the brief's draft validated as-is (`isl`/`osl` dataset
shorthands, `prefix_prompts.shared_system_length`, top-level `sweep:` with
`planner/search_space/sla_filters/max_iterations/sla_precision`,
`randomSeed`, `useServerTokenCount`).
