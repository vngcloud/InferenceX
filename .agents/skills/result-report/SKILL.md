---
name: result-report
description: Turn InferenceX benchmark runs into one standard report. Input is GitHub Actions run links/IDs or downloaded artifact/aiperf folders; output is report.md (+ optional report.html with charts) and data.json with throughput, cache hit by tier (HBM/DRAM/L3/router), TTFT/ITL/E2E avg-p50-p90-p95-p99, max CCU under SLA for search runs, and A/B or grid comparisons. Use when asked to report, summarize, compare or read the results of a benchmark run, an AgentX/tau2/livekit ladder, a sweep-isl*/fixed-isl* grid, or an aiperf output folder.
---

# result-report

One command turns benchmark results into a report the team can paste into Redmine or an MR. Tables
come from the script, so every report has the same columns and units. You write only the Key findings.

## Run it

    python3 .agents/skills/result-report/scripts/result_report.py <input>... [--html] [--title T] [--baseline ARM]

`<input>` is any mix of:
- a run URL `https://github.com/vngcloud/InferenceX/actions/runs/<id>` or a bare run id (repo `--repo`, default vngcloud/InferenceX);
- a folder of downloaded artifacts (`bmk_agentic_*`, `agentic_*`) or hand-named copies;
- a raw aiperf output folder (contains `profile_export_aiperf.json`).

Run links need the `gh` CLI, logged in. The script reads single files out of the artifact zips with
HTTP Range requests: a 266 MB search artifact costs about 1 MB. It never opens the heavy files
(`server_metrics_export.json` of fixed runs, `profile_export.jsonl`, `inputs.json`, timeslices).

Output goes to `~/bench-reports/<date>-<slug>/` (or `--out DIR`): `report.md`, `data.json`,
`report.html` with `--html`. Downloads are cached in `<out>/.cache/`, so a re-run is fast.

## Several inputs: arms and grids

Inputs are grouped automatically:
- **Arm** = one serving config. Runs whose `serving.yaml` differ only in `name`, `scenario` or `conc`
  are the same arm (so one config dispatched once per scenario stays one arm).
- **Scenario** = the workload. `sweep-isl*-osl*-cache*` / `fixed-isl*-osl*-cache*` names become grid axes.

Same arm + many scenarios → a grid summary. Several arms + the same scenario → a comparison: every
non-baseline cell shows `value (Δ%)` against the baseline arm (first arm, or `--baseline <label>`).

## Write the Key findings

1. Run the command once.
2. Read `data.json` (`points[]`: one per config; `arms[]`; `notes[]`) and the tables in `report.md`.
3. Write 3–6 bullets to a file, in English, then re-run the same command with `--findings <file>`
   (cached, takes seconds). Rules:
   - Lead with the decision: the best arm per scenario, or the max CCU under SLA.
   - Every bullet names the arm, scenario and CCU and quotes a number from the tables.
   - Quote deltas only when |Δ| ≥ 5 %.
   - For search runs, name the binding SLA and where it broke (`ITL p90 < 50 ms, 50.85 ms at c61`).
   - Repeat any warning from Notes that changes the conclusion: errors, `⚠` cache mismatch,
     aggregate-only rows (no p99), failed or expired jobs.
   - No adjectives without a number.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | report written | — |
| 1 | no configs found | check the run has `bmk_agentic_*`/`agentic_*` artifacts, or the folder layout |
| 2 | bad input or unknown `--baseline` | fix the argument; the error lists the arm labels |
| 3 | Range reads failed and the full zip is over `--max-full-zip-mb` (100) | ask the user, then re-run with a bigger `--max-full-zip-mb` |
| 4 | `gh` missing or failing | `gh auth status`; install / log in |

## Reading the tables

- Latency is ms; columns avg/p50/p90/p95/p99. `—` means the value does not exist, never zero.
- Rows built from the aggregate JSON only (no aiperf export) have no p99; Notes says so.
- Cache columns: overall, GPU (HBM), CPU (DRAM), external (L3), router. Columns empty for every row are hidden.
  vLLM cannot split CPU from L3: its offload hits show as external.
- Grid tables compare the nominal cache % in the scenario name against the measured **overall** hit; `⚠` when they differ by more than 10 points.
- Search runs: the headline CCU is the largest CCU that passed every SLA (`feasible_max`), not the highest-throughput probe; the probe table marks both.

## Tests

    python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v
