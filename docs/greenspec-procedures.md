# GreenSpec Procedures

Use this guide to benchmark speculative decoding on GreenSpec: add a cell, pick a subset, dispatch it, read the result, and keep the data off public artifacts. GreenSpec is an internal speculative-decoding bench cut from production chat traffic (tools, multi-turn history, real system prompts), released as one private dataset on Hugging Face. [`benchmarks/benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) (`run_greenspec_aiperf`) and [`infx/bench_serving/greenspec_aiperf.py`](../infx/bench_serving/greenspec_aiperf.py) are authoritative; this page explains how to use them.

> **Required:** GreenSpec rows are internal and this repository's artifacts are public. Never add `--export-level raw`, aiperf `inputs.json` uploads, or vLLM request logging (`--enable-log-requests`, `--enable-log-outputs`) to a GreenSpec recipe, and never upload `results/greenspec_aiperf/` or the server log by another path. The guard at the end of `run_greenspec_aiperf` is the last line of defense, not a license to relax the recipe.

## Procedure index

| Procedure | Use when |
| --- | --- |
| [Understand the flow](#how-a-greenspec-cell-runs) | Reading a run's log or changing the client |
| [Check prerequisites](#prerequisites) | Before the first GreenSpec cell on a runner |
| [Choose a selection](#choose-a-selection) | Picking the subset, size, and decode parameters |
| [Add a GreenSpec cell](#add-a-greenspec-cell) | Benchmarking a new model, engine, or draft |
| [Dispatch and read results](#dispatch-and-read-results) | Running cells in CI and comparing arms |
| [Troubleshoot](#troubleshooting) | A cell fails before, during, or after the client |

## Governing sources

| Source | What it governs |
| --- | --- |
| [`benchmarks/benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) | `resolve_greenspec_dataset`, `run_greenspec_aiperf`, the pinned dataset revision, every `GREENSPEC_*` variable |
| [`infx/bench_serving/greenspec_aiperf.py`](../infx/bench_serving/greenspec_aiperf.py) | Subset selection, preset names, the `raw_payload` fallback, the artifact guard and its allowlist |
| [`infx/bench_serving/speedbench_aiperf.py`](../infx/bench_serving/speedbench_aiperf.py) | Mapping the aiperf export and spec-decode counters onto the `benchmark_serving` result keys |
| [`utils/aiperf`](../.gitmodules) | The pinned aiperf fork that ships `GreenSpecLoader` and the `greenspec_*` presets |
| [`.github/workflows/benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml) | `GREENSPEC_HF_TOKEN` wiring and the allowlisted `greenspec_aiperf_*` artifact upload |
| [`utils/test_greenspec_aiperf.py`](../utils/test_greenspec_aiperf.py) | Selection, preset, payload, and guard contracts |

## How a GreenSpec cell runs

1. The recipe calls `resolve_greenspec_dataset` before starting the server. It installs the pinned aiperf venv and fetches `greenspec.jsonl` from `Green-Lab/GreenSpec` at the commit pinned in `GREENSPEC_HF_REVISION` (`HF_HUB_CACHE` first, then the network), then checks `SHA256SUMS`. A missing token or a bad pin fails here, before the GPU is held.
2. The recipe starts the engine and calls `run_greenspec_aiperf` once the server is ready.
3. `run_greenspec_aiperf` counts the selection (`greenspec_aiperf pool`) and fails on an empty one. It then picks the aiperf loader: the `greenspec_*` preset that covers the selection when the pinned aiperf has it, otherwise a pre-filtered `raw_payload` file written to a temp dir outside `results/`. Both paths send the same request bodies.
4. aiperf replays the selection in file order (`--dataset-sampling-strategy sequential`, seed 42) at `CONC`, streaming, with `--use-server-token-count`. Output length is **not** forced: no `--osl`, no `ignore_eos`; each request stops on its own, capped by `GREENSPEC_MAX_TOKENS`. The engine's spec-decode counters are scraped from `/metrics` before and after.
5. `speedbench_aiperf` writes `$RESULT_FILENAME.json` in the `benchmark_serving` shape, adds `acceptance_length` and `acceptance_rate` from the counters, and records the selection as `gs_*` keys. The cell then flows through the normal fixed-seq-len collect path.
6. `greenspec_aiperf guard` runs last. It deletes everything in `results/greenspec_aiperf/` except the allowlist, replaces error messages in the aiperf export with `<withheld>`, replaces the server log with a stub (or moves it to `GREENSPEC_KEEP_LOG_DIR`), and fails the job if any GreenSpec `question_id` or any file over 2 MiB is left in what gets uploaded.

## Prerequisites

| Requirement | Why |
| --- | --- |
| Repository secret `GREENSPEC_HF_TOKEN`, a read token for the private `Green-Lab/GreenSpec` repo | `resolve_greenspec_dataset` falls back to `HF_TOKEN`, which normally cannot read it |
| The runner launcher forwards `GREENSPEC_HF_TOKEN` into the container | `launch_h200-greennode.sh` lists it in `RUN_ENV`; `launch_hardware-hcm.sh` uses `--export=ALL`. Add it to any launcher with an explicit list |
| The `utils/aiperf` pin includes the `greenspec_*` presets | Without them every cell takes the `raw_payload` fallback; results stay valid, and `gs_aiperf_loader` in the result shows which path ran |
| The engine exposes spec-decode counters on `/metrics` (vLLM `vllm:spec_decode_*`) | `acceptance_length` is `null` otherwise |
| `--max-model-len` covers the longest bucket you select | Rows reach the 64k+ buckets; a short context turns them into failed requests |

## Choose a selection

The release has four subsets. `diverse` is every use case across every input-length bucket with equal-size buckets; the other three are the rows of one use case. `lite` is a fixed quarter of every bucket, for fast iteration; use the full set for headline numbers.

| `GREENSPEC_SUBSET` | Content | Buckets (`GREENSPEC_BUCKET`) |
| --- | --- | --- |
| `diverse` (default) | All three use cases | Any bucket below |
| `agentic_coding` | Coding-agent traffic | `agentic_coding__0-8k`, `__8k-32k`, `__32k-64k`, `__64k+` |
| `agentic_tools` | Tool-calling traffic | `agentic_tools__0-4k`, `__4k-16k`, `__16k-32k`, `__32k+` |
| `llm_task` | Plain chat, no tools | `llm_task__0-1k`, `__1k-4k`, `__4k-16k`, `__16k+` |

| Variable | Default | Meaning |
| --- | --- | --- |
| `GREENSPEC_LITE` | `0` | `1` keeps the lite quarter of every bucket |
| `GREENSPEC_BUCKET` | unset | One bucket. The bucket presets cover the full bucket, so a lite or `recorded` bucket selection runs through `raw_payload` |
| `GREENSPEC_MODE` | `target-default` | `target-default` sends messages, tools, and `response_format` only, so decode parameters come from the server. `recorded` adds each request's own production parameters (temperature, top-p, its own output cap); with a bucket it runs through `raw_payload` |
| `GREENSPEC_MAX_TOKENS` | `8192` | Output cap per request. In `recorded` mode it lowers a request's own cap, never raises it |
| `GREENSPEC_EXTRA_INPUTS` | `{}` | JSON object merged into every request body, e.g. `{"temperature": 0}` |
| `GREENSPEC_NUM_PROMPTS` | whole selection | Requests per cell. Rows are interleaved across buckets, so any prefix stays balanced and every arm sees the same rows |
| `GREENSPEC_WARMUP_REQUESTS` | `0` | Warmup requests excluded from the metrics |

The full list, including `GREENSPEC_METRICS_URL`, `GREENSPEC_SERVER_PID`, `GREENSPEC_SERVER_LOG`, `GREENSPEC_KEEP_LOG_DIR`, and `GREENSPEC_META`, is in the comment above `run_greenspec_aiperf`.

For a speculative-decoding comparison, keep everything but the draft identical between arms: same subset, mode, cap, `CONC`, engine flags, and image. Comparing acceptance across `target-default` and `recorded` measures the sampling change, not the draft.

## Add a GreenSpec cell

Follow [Configuration procedures](./configuration-procedures.md#add-a-model--hardware-recipe) for the general recipe rules. GreenSpec-specific steps:

1. **Write a shared body** under `benchmarks/single_node/fixed_seq_len/`, e.g. `<model>gs_body.sh`. It sources `benchmark_lib.sh`, maps the wrapper's variables to `GREENSPEC_*`, calls `resolve_greenspec_dataset` before the server starts, starts the engine with logs in `/workspace/server.log`, waits for readiness, then runs:

   ```bash
   GREENSPEC_SERVER_PID="$SERVER_PID" \
   GREENSPEC_SERVER_LOG="$SERVER_LOG" \
   GREENSPEC_META="gs_arm=\"$GS_ARM\" num_speculative_tokens=$NUM_SPEC_TOKENS" \
       run_greenspec_aiperf
   ```

   Exit with its return code: a guard failure must fail the job. Keep the engine flags of the arm's non-GreenSpec sibling so results are comparable; the only engine difference between arms is `--speculative-config`.
2. **Add one wrapper per arm and selection.** Each sets the arm and selection and sources the body:

   ```bash
   GS_ARM=mtp
   GS_SUBSET=agentic_tools
   GS_LITE=1
   NUM_SPEC_TOKENS=4
   source "$(dirname "$0")/<model>gs_body.sh"
   ```

   The wrapper name must match the master-config key, as for any fixed-seq-len recipe.
3. **Add the master-config keys**, one per arm, with `spec-decoding` set on the draft arm. `isl` and `osl` are labels only: every row brings its own prompt, and output is capped by `GREENSPEC_MAX_TOKENS`. Say so in a comment above the keys.
4. **Add a `perf-changelog.yaml` entry** naming the image, draft, subset, mode, `CONC`, and output cap.
5. **Run the local checks:** `python -m pytest utils/test_greenspec_aiperf.py utils/test_speedbench_aiperf.py` with the aiperf venv, plus the matrix generation in [CI procedures](./ci-procedures.md#local-matrix-generation).

A worked example (Gemma-4 31B, base vs MTP depth 4, `agentic_tools` lite, conc 8) lives on the branch `bench/gemma4-greenspec-hcm`: `gemma4gs_body.sh`, `gemma4gsatlbase_fp8block_h200.sh`, `gemma4gsatlmtp_fp8block_h200_mtp.sh`, and the keys `gemma4gsatl{base,mtp}-fp8block-hardware-hcm-vllm`.

## Dispatch and read results

Dispatch both arms in one run so they land on the same node and image (see [CI procedures](./ci-procedures.md#manual-end-to-end-dispatch)):

```bash
gh workflow run e2e-tests.yml -R vngcloud/InferenceX --ref <branch> \
  -f generate-cli-command="test-config --config-files configs/nvidia-master.yaml \
    --config-keys <base-key> <draft-key> --runner-node-filter <runner-node> --no-evals" \
  -f test-name="greenspec <model> <subset> base vs <draft>"
```

Each cell uploads `greenspec_aiperf_<result>` with exactly these files:

| File | Holds |
| --- | --- |
| `result.json` | The `benchmark_serving`-shaped result, including `acceptance_length`, `acceptance_rate`, and the `gs_*` selection keys |
| `profile_export_aiperf.json` | aiperf's aggregate export, error messages withheld |
| `spec_metrics_before.json`, `spec_metrics_after.json` | Spec-decode counter snapshots |
| `benchmark_command.txt` | The aiperf command line |

The same result also goes through the normal fixed-seq-len collect path. Compare arms on `output_throughput` (the speedup), `median_ttft_ms`, `median_tpot_ms`, `median_itl_ms`, and `acceptance_length`. Check `completed` equals `num_prompts` in both arms before reading any ratio; a cell with failed requests compares different work. `gs_aiperf_loader`, `gs_dataset_repo`, `gs_mode`, and `gs_max_tokens` show what actually ran.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `could not fetch greenspec.jsonl from Green-Lab/GreenSpec@...` | The token cannot read the repo: check `GREENSPEC_HF_TOKEN` exists and the launcher forwards it |
| `GreenSpec subset=... matches no rows` | Typo in `GREENSPEC_SUBSET` or `GREENSPEC_BUCKET`, or a bucket from another use case. Use the names in [Choose a selection](#choose-a-selection) |
| `gs_aiperf_loader` is `raw_payload` when a preset was expected | The `utils/aiperf` pin lacks the preset, or the selection has none (a bucket with `GREENSPEC_LITE=1` or `recorded`). Results are still valid |
| Many failed requests in long buckets | `--max-model-len` below the bucket's input length, or the server ran out of KV cache. Check the kept server log on the runner |
| `GreenSpec guard failed; nothing from this run may be uploaded` | A row id or an oversized file is in an upload path. Do not bypass the guard: find what wrote it (a new aiperf export file, an extra upload path, a debug dump) and remove it from the recipe |
| `acceptance_length` is `null` | The engine does not expose spec-decode counters at `GREENSPEC_METRICS_URL`, or it is the base arm |

The server log is withheld from artifacts by design. To read it, set `GREENSPEC_KEEP_LOG_DIR` to a runner-local path in the recipe and read it on the runner; never attach it to an issue or PR.

## Review checklist

- [ ] Arms differ only in the draft: same image, engine flags, subset, mode, cap, and `CONC`.
- [ ] No raw export, request logging, or extra upload path for `results/greenspec_aiperf/` or the server log.
- [ ] The recipe exits with `run_greenspec_aiperf`'s return code.
- [ ] `--max-model-len` covers the selected buckets.
- [ ] Master-config comment states that `isl`/`osl` are labels; `perf-changelog.yaml` names the selection.
- [ ] Both arms report `completed == num_prompts` before any comparison is quoted.
