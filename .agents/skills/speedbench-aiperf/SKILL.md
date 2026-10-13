---
name: speedbench-aiperf
description: Use when benchmarking speculative decoding (MTP, EAGLE3, DFlash, draft models) on SPEED-Bench on the vngcloud fork, writing a new SPEED-Bench recipe or config key, dispatching a SPEED-Bench cell, reading its acceptance length, or when a SPEED-Bench job fails on the dataset fetch (404, SHA256 mismatch), on an empty category, or with no profile_export_aiperf.json.
---

# SPEED-Bench on aiperf (vngcloud fork)

SPEED-Bench cells are ordinary `fixed-seq-len` jobs whose recipe drives aiperf through `run_speedbench_aiperf` (`benchmarks/benchmark_lib.sh`) instead of `vllm bench serve`. The recipe owns the engine launch; the shared client owns the dataset, the load, and the result. The result is a benchmark_serving-shaped `$RESULT_FILENAME.json` with `acceptance_length` added, so `process_result` and the collect path treat it like any other 8k1k row.

This skill covers only what is specific to SPEED-Bench. For config keys, recipe naming, runner pools, dispatch and debugging, follow `inferencex-dispatch`.

## Where things live

| Piece | Location | How it reaches a run |
|---|---|---|
| Workflow YAML (`SPEEDBENCH_HF_TOKEN` secret, `speedbench_aiperf_*` artifact upload) | `vng-benchmark` | `--ref vng-benchmark`, always |
| Shared client (`resolve_speedbench_dataset`, `run_speedbench_aiperf`, `infx/bench_serving/speedbench_aiperf.py`, launcher `RUN_ENV`) | `feat/speedbench-aiperf` | base of every recipe commit |
| Recipe + config key + `perf-changelog.yaml` entry | a commit on top of `feat/speedbench-aiperf`, pushed to `refs/bench/<name>` | `-f ref=<sha>` |

Keep recipes off `feat/speedbench-aiperf`. Reference arm: `refs/bench/gemma4-speedbench-aiperf` (Gemma-4 31B FP8-block, base/MTP × high/low entropy).

## Dataset

- `Noridom1/speed-bench-prepared`, a **private** HF dataset, pinned at revision `6e5c3a81f3d529e4daba914e1ff1046823b0165b` (`SPEEDBENCH_HF_REPO` / `SPEEDBENCH_HF_REVISION`). It holds all six splits with prompts resolved from their sources and no `SPECDEC_BENCH` placeholder rows. `SHA256SUMS` is checked after every fetch.
- Read with the `SPEEDBENCH_HF_TOKEN` repo secret, falling back to `HF_TOKEN`. The fetch runs in a subshell with `set +x`. Never echo, commit or paste the token, and do not make the repo public to work around a 404.
- The cache comes first: `HF_HUB_CACHE=/mnt/hf_hub_cache` persists on the runners. A directory set in `SPEEDBENCH_DIR` that holds `<config>.jsonl` beats both the cache and the network.

| `SPEEDBENCH_CONFIG` | `SPEEDBENCH_CATEGORY` (unset = whole split) | aiperf preset |
|---|---|---|
| `throughput_1k` `_2k` `_8k` `_16k` `_32k` | `low_entropy` `mixed` `high_entropy` | `speed_bench_throughput_8k_high_entropy` |
| `qualitative` | `coding` `humanities` `math` `multilingual` `qa` `rag` `reasoning` `roleplay` `stem` `summarization` `writing` | `speed_bench_coding` |

The presets exist only in the pinned `utils/aiperf` fork that `install_agentic_deps` installs, not in upstream aiperf.

## Recipe contract

The launcher derives the recipe path from `model-prefix` and has no slot for the arm or category. Write one shared body plus one thin wrapper per cell; see `gemma4sba_body.sh` and `gemma4sbamtphi_fp8block_h200_mtp.sh` on the reference ref. The wrapper sets the cell's variables and runs `source "$(dirname "$0")/<body>.sh"`.

The body must:

1. Call `resolve_speedbench_dataset || exit 1` **before** starting the server. It installs the aiperf venv and fetches the file, so a token or dataset problem fails in seconds instead of after model load.
2. Export `SPEEDBENCH_CONFIG`, plus `SPEEDBENCH_CATEGORY` and `SPEEDBENCH_IGNORE_EOS` when needed.
3. Start the server with Prometheus metrics on (vLLM: never `--disable-log-stats`). Acceptance is the delta of the `vllm:spec_decode_*` counters between two snapshots; on SGLang it falls back to the `spec_accept_length` gauge. Leave prefix caching off so no cell reuses another's prefill.
4. Size `--max-model-len` for the split plus OSL. The reference uses 65536 so `throughput_32k` fits.
5. After `wait_for_server_ready`, run:

   ```bash
   SPEEDBENCH_SERVER_PID="$SERVER_PID" \
   SPEEDBENCH_META="sb_arm=$ARM num_speculative_tokens=$N draft_model=${DRAFT:-null}" \
       run_speedbench_aiperf
   ```

   Exit with its return code, and keep the `EVAL_ONLY` / `RUN_EVAL` block of a sibling `fixed_seq_len` recipe.

Knobs read by `run_speedbench_aiperf` (the full list is in its header comment):

| Var | Default | Notes |
|---|---|---|
| `SPEEDBENCH_NUM_PROMPTS` | `CONC*10`, clamped to [64, pool] | rows taken in file order, so every arm at one conc sees the same prompts |
| `SPEEDBENCH_IGNORE_EOS` | `1` | forces OSL output tokens per turn |
| `SPEEDBENCH_EXTRA_INPUTS` | none | JSON merged into every request, e.g. `{"temperature":0}` |
| `SPEEDBENCH_WARMUP_REQUESTS` | `0` | |
| `SPEEDBENCH_METRICS_URL` | `http://localhost:$PORT/metrics` | source of the acceptance counters |
| `SERVED_MODEL_NAME`, `AIPERF_TOKENIZER` | `$MODEL` | set when the server uses an alias or a local path |

A new env var read by the recipe must be added to the launcher's `RUN_ENV` on `vng-benchmark`. `launch_h200-greennode-slurm-1x.sh` forwards the whole environment instead.

## Config key

Use a normal `fixed-seq-len` key, one per cell: `model-prefix` without `_`, `spec-decoding: mtp` on MTP cells so the `_mtp` recipe is picked, and `tp: 1` for a 1-GPU arm. `isl`/`osl` label the row: `osl` is the real per-request output length, while the real prompt lengths come from the split and land in `total_input_tokens`. The runner pool (`configs/runners.yaml`) is read from your `ref`, so a node only in your commit's pool still works, but pool changes belong on `vng-benchmark`.

## Check, push, dispatch

```bash
uv run -q --python 3.11 --with pytest --no-project python -m pytest utils/test_speedbench_aiperf.py -q   # after touching the shared client
bash -n benchmarks/single_node/fixed_seq_len/<body>.sh
git push origin HEAD:refs/bench/<name> && SHA=$(git rev-parse HEAD)
gh workflow run e2e-tests.yml --repo vngcloud/InferenceX --ref vng-benchmark \
  -f ref=$SHA \
  -f test-name="<key> 8k1k conc 8 speedbench-aiperf <what changed> ${SHA:0:8}" \
  -f generate-cli-command="test-config --config-files configs/nvidia-master.yaml --config-keys <key> --conc 8 --seq-lens 8k1k --scenario-type fixed-seq-len --no-evals --runner-node-filter <node>"
```

Smoke with one cell at conc 8 before a ladder. A warm node takes about 10 minutes: about 5.5 min server boot and about 3.5 min for 80 prompts on `throughput_8k`.

## Reading a run

The job log prints `===== SPEED-Bench cell =====` (preset, pool, num_prompts), then the aiperf tables, then the converted summary:

```json
{"completed": 80, "num_prompts": 80, "output_throughput": 387.0, "total_token_throughput": 3796.8,
 "mean_ttft_ms": 2588.0, "mean_tpot_ms": 17.84, "acceptance_length": 2.39, "acceptance_rate": 0.347}
```

These numbers are the reference smoke: `gemma4sbamtphi`, MTP with 4 speculative tokens, `high_entropy`, conc 8, 1×H200 on `h200-greennode_08`, run 36543057636. `acceptance_length` counts the bonus token, so its ceiling is `num_speculative_tokens + 1`. It is `null` for a base arm.

The `speedbench_aiperf_<RESULT_FILENAME>` artifact contains `profile_export_aiperf.{json,csv}`, `profile_export.jsonl` (per request), `server_metrics_export.*`, `spec_metrics_{before,after}.json`, `benchmark_command.txt`, `result.json` and `logs/aiperf.log`. `inputs.json` is excluded because it holds the prompts. The metadata keys `sb_dataset_subset`, `sb_category`, `sb_aiperf_preset`, `sb_num_conversations`, `sb_ignore_eos`, `sb_dataset_repo` and `sb_extra_inputs`, plus your `SPEEDBENCH_META`, are recorded in the result.

## Symptom → fix

| Log line | Cause → fix |
|---|---|
| `could not fetch ... speed-bench-prepared` / 404 | No token reached the container. Check that the job env shows `SPEEDBENCH_HF_TOKEN: ***`. If it is missing, the workflow on `--ref` does not wire the secret; dispatch with `--ref vng-benchmark`. If it is present, the token cannot read the repo. |
| `does not match SHA256SUMS` | Corrupted cache or a different revision. Remove that snapshot from `/mnt/hf_hub_cache` on the node and rerun. |
| `unknown SPEEDBENCH_CONFIG` | Typo; see the table above. |
| `category '<x>' matches no rows` | That category is not in this split, e.g. a qualitative category on a `throughput_*` split. |
| `aiperf exited with code N and wrote no profile_export_aiperf.json` | Read `logs/aiperf.log` in the artifact and the server log. The server usually died (`SPEEDBENCH_SERVER_PID` stops the client) or rejected the requests, e.g. prompt plus OSL over `--max-model-len`. |
| `acceptance_length: null` on a spec arm | The metrics URL was unreachable or the engine exposes no spec counters. Compare `spec_metrics_before.json` and `spec_metrics_after.json`. |
