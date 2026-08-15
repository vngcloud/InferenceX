---
name: bench-8k1k
description: Bring up and run a single-node 8k1k (isl=8192/osl=1024) fixed-seq-len benchmark for a model+hardware+framework combo, dispatched directly from a branch without opening a PR or touching main. Use when asked to benchmark 8k1k throughput/latency for a new or existing config-key, extend an existing concurrency ladder, or debug a fixed-seq-len sweep that isn't landing on InferenceX-app. Composes with add-model-hardware (which this skill assumes was already run, or runs the same steps inline for a day-zero recipe) and debug-runs (same root-cause loop, applied here to CI matrix jobs instead of SSH).
---

# Bench 8k1k (fixed-seq-len, no-merge dispatch)

Everything below was learned bringing up `gemma4-fp8block-h200-vllm` (day-zero, single H200,
TP1) end-to-end without ever opening a PR or merging to main.

## 0. When you don't need a PR at all

`run-sweep.yml` only triggers on `push`/`pull_request` against `main`. To validate a recipe
purely for exploration — no CI gate, no ingest, no review — dispatch `e2e-tests.yml` directly
against your branch instead:

```bash
gh api -X POST /repos/<org>/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='<branch>' \
  -f 'inputs[ref]=<branch>' \
  -f 'inputs[test-name]=<model-prefix> 8k1k <purpose>' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nvidia-master.yaml --model-prefix <model-prefix>' \
  -f 'inputs[duration-override]='
```

**`--config-keys` is not a `full-sweep` flag** — it only exists on the `test-config`
subcommand. Filter `full-sweep` by `--model-prefix` / `--precision` / `--framework` /
`--runner-type` instead, or it errors `unrecognized arguments` at the very first CI step
(`get-jobs`) before anything touches a GPU.

This path is completely disconnected from InferenceX-app: `trigger-ingest` /
`trigger-agentic-ingest` in `run-sweep.yml` gate on `github.event_name == 'push' && github.ref
== 'refs/heads/main'`. A `workflow_dispatch` run — on any branch, including `main` — never
fires them. Merging via `/reuse-sweep-run` is the only path that reuses a validated run's
artifacts into a real ingest; this skill's dispatch is purely for your own inspection.

## 1. Local validation before spending any GPU time

```bash
bash -n benchmarks/single_node/fixed_seq_len/<script>.sh
python3 -c "import yaml; yaml.safe_load(open('configs/nvidia-master.yaml'))"
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml --config-keys <config-key>
```

Sanity-check the printed matrix: `max-model-len` should be the scenario-computed
`isl+osl+slack` (never the model's full context — fixed-seq-len doesn't need it), and
`run-eval` should be `true` on exactly two conc points per the policy in
`mark_eval_entries()`: **the highest conc** and **the median conc**, only among points
`>= MIN_EVAL_CONC`. This recomputes automatically whenever you change the ladder — no
manual tagging needed.

## 2. Extending an existing ladder without re-running what already passed

Once conc `[1,2,...,64]` already has green results, don't turn `conc-end: 64` into
`conc-end: 256` and redispatch `full-sweep` with no filter — that regenerates and reruns
*every* point, including the ones you already have data for. Instead:

- Update the master-config entry itself to the correct **long-run** definition
  (`conc-start: 1, conc-end: 256`) — this is the recipe's steady-state description going
  forward, for whoever runs a real full sweep on it later.
- For *this* dispatch, filter to only the new points with `--min-conc`/`--max-conc`:

```bash
python3 utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml --model-prefix <model-prefix> \
  --min-conc 128 --max-conc 256   # verify locally: only the new points print
```

Pass the identical `--min-conc`/`--max-conc` flags in the dispatched `generate-cli-command`.
Eval marking re-runs on the *filtered* set, so the highest point of your filtered range picks
up `run-eval: true` even if it wasn't the eval point in the unfiltered run — that's fine, it's
still a legitimate new highest-conc data point.

## 3. Sizing how far to extend, from KV cache headroom — not by guessing

Before proposing new conc points, check how much KV cache the *last* successful run actually
used. Download its server log and look at the peak `GPU KV cache usage`:

```bash
gh run download <RUN_ID> --repo <org>/InferenceX \
  -n server_logs_<full_exp_name> -D ./server_log
grep -oE "GPU KV cache usage: [0-9.]+%" ./server_log/*.log | grep -oE "[0-9.]+" | sort -n | tail -5
```

vLLM's own `Maximum concurrency for N tokens per request: X.XXx` line at server-startup is a
**worst-case** estimate (every request at the full context window simultaneously) — it is not
the real ceiling once prefix caching is in play. Trust the observed peak usage over the sweep
instead. Peak usage well under 100% (e.g. ~50% at conc 64) means there's real headroom for a
further doubling; peak usage already near 100% means the next point is likely to start
queuing/thrashing rather than scaling throughput — that's still useful data (it's the knee),
just don't be surprised when the numbers plateau or wobble.

## 4. What "matrix job" parallelism actually means here

Every conc point is declared as an independent matrix job with its own freshly-spun engine —
by design, so one point's state never leaks into another's. That does **not** mean they run
concurrently. Self-hosted runner pools (e.g. `h200`, `cluster:h200-greennode`) are typically
one or a handful of physical boxes; jobs queue and execute serially against whichever runner
frees up next. Check `startedAt` across jobs in the same run before assuming anything is
actually parallel:

```bash
gh run view <RUN_ID> --repo <org>/InferenceX --json jobs \
  --jq '.jobs[] | {name, startedAt, completedAt}'
```

`num-prompts` is `CONC * 10` everywhere in this repo (`run_benchmark_serving --num-prompts
"$((CONC * 10))"`) — 10x concurrency gives enough steady-state samples for stable
TTFT/TPOT percentiles post-warmup. It is not a bug that conc 256 sends 2560 requests; it does
mean wall-clock scales roughly linearly with conc (a conc-64/640-request throughput job
measured ~28 min including fresh engine boot + CUDA-graph capture — budget accordingly for
higher points, and warn the user before dispatching a ladder extension that will run for
hours, especially serialized behind other points on the same limited pool).

## 5. Runner-specific gaps that only surface on the *second* SKU/framework combo

A launcher that has only ever served one framework/scenario-type combination often has
silent gaps that a brand-new combination will hit. Check these explicitly before assuming a
`bash -n`-clean recipe is CI-ready, especially on any launcher whose existing configs are
all one framework (e.g. a `cluster:h200-greennode`-style launcher that's only ever run
`sglang` agentic-coding):

- **Filename suffix logic**: does it unconditionally append `_${FRAMEWORK}` (breaks the
  default/bare-framework recipe filename), instead of conditionally suffixing only
  non-default frameworks like the other launchers in the same SKU family do?
- **Docker/container env whitelist**: if the launcher passes env vars through an explicit
  `-e NAME` list (rather than `--export=ALL`/`srun`'s full passthrough), does that list
  include the vars *your* scenario type needs? An agentic-only launcher's whitelist has no
  reason to carry `ISL`/`OSL`/`MAX_MODEL_LEN`/`RANDOM_RANGE_RATIO` until a fixed-seq-len
  config is the first to use it.
- **Recipe script itself**: does it have the `EVAL_ONLY`/`RUN_EVAL` block every sibling
  fixed-seq-len script has?
  ```bash
  if [ "${EVAL_ONLY}" = "true" ]; then
      setup_eval_context
      MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
  fi
  # ... serve, then run_benchmark_serving ...
  if [ "${RUN_EVAL}" = "true" ]; then
      run_eval --framework lm-eval --port "$PORT"
      append_lm_eval_summary
  fi
  ```
  Missing it doesn't fail loudly at serve time — the server starts fine, `run_benchmark_serving`
  itself no-ops throughput under `EVAL_ONLY=true` ("EVAL_ONLY mode: skipping throughput
  benchmark"), and the job only dies afterward with `Eval-only run failed: no results*.json
  files found.` — a confusing failure to root-cause blind if you don't know to look for the
  missing block first.

All three of the above are real bugs found bringing up `gemma4-fp8block-h200-vllm` on
`cluster:h200-greennode` (a launcher whose only prior configs were `sglang` agentic — see
`runners/launch_h200-greennode.sh` history). Fix them in the launcher/recipe, not with a
workaround in the dispatch command.

## 6. Root-causing a matrix job failure

```bash
gh run view <RUN_ID> --repo <org>/InferenceX --json jobs \
  --jq '.jobs[] | select(.conclusion=="failure") | "\(.databaseId)\t\(.name)"'
gh api repos/<org>/InferenceX/actions/jobs/<job_id>/logs > /tmp/job.log
grep -niE "error|traceback|no such file|not found|refused|timeout|exit code" /tmp/job.log
```

Prefer the *lowest*-conc failing job first — it isolates the bug from scale-dependent noise
(OOM, queueing) and finishes fastest. If every job in the matrix fails identically, that's a
recipe/launcher bug, not a flake — fix and redispatch rather than rerunning.

## 7. Pulling real numbers once it's green

```bash
gh run download <RUN_ID> --repo <org>/InferenceX -n results_bmk -D ./results
cat ./results/agg_bmk.json | jq -r '.[] | [.conc, (.tput_per_gpu|round), \
  (.output_tput_per_gpu|round), (.mean_ttft*1000|round), (.mean_tpot*1000|round)] | @tsv' \
  | sort -n | column -t -N "conc,tput_tok/s,out_tput_tok/s,ttft_ms,tpot_ms"
```
