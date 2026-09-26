---
name: inferencex-dispatch
description: Use when running an InferenceX benchmark on the vngcloud fork (agentic-coding or 8k1k fixed-seq-len) from a branch, adding a config key or recipe for a VNG runner (h200-greennode, h200-greennode-slurm, b300-netperf), or when an e2e-tests run fails at get-jobs, on a missing recipe or env var, on enroot/pyxis, or with OOM on a GreenNode node.
---

# InferenceX dispatch (vngcloud fork)

One arm = one config key + one recipe on a branch, dispatched through `e2e-tests.yml`. Results stay in run artifacts; the fork ingests nothing.

## Rules

1. Always pass `--repo vngcloud/InferenceX`. The local `gh` default may be `SemiAnalysisAI/InferenceX`.
2. Infra changes (`runners/`, `benchmark_lib.sh`, `.github/`, `configs/runners.yaml`) are committed to `vng-benchmark` first; experiment branches then rebase. Experiment branches carry only the config key, the recipe, and the `perf-changelog.yaml` entry.
3. `git fetch` then branch from `origin/vng-benchmark` as `bench/<model>-<topic>`. When results are in, tag the SHA and delete the branch.
4. Smoke first: one mid CCU at `duration-override=90` (8k1k: one conc). Run the full ladder only after the smoke is green.
5. Pin exactly one node with `--runner-node-filter`. Exception: a single-node pool whose label contains the node name (`cluster:h200-greennode_04`). The filter is a substring match, so it emits both the label and the node, which doubles every job; omit the filter there.
6. Change one thing per retry and name that change in `test-name`.
7. Commit and push before dispatching, then confirm that the run's `headSha` equals `git rev-parse HEAD`.

## Files per arm

- **Config key** in `configs/nvidia-master.yaml`: `image`, `model`, `model-prefix`, `runner: cluster:*`, `precision` (match the checkpoint naming of sibling keys, e.g. `fp8block`), `framework`, `multinode: false`, one scenario, explicit `conc-list`. One key maps to one runner pool, so the same arm on another pool needs a second key.
- **perf-changelog entry**: use the `fork-changelog` skill.
- **Recipe path.** The launcher computes the path, so name the file to match it (`${EXP_NAME%%_*}` = `model-prefix`, which therefore **must not contain `_`**):

| Launcher | Recipe |
|---|---|
| `h200-greennode`, `h200-greennode-slurm` | `benchmarks/single_node/<agentic\|fixed_seq_len>/<prefix>_<precision>_h200[_<fw>][_mtp\|_specdec].sh`. `_<fw>` only when not vllm. `_mtp` = `spec-decoding: mtp`, `_specdec` = `draft_model`. Falls back to the file without the spec suffix. |
| `b300-netperf` | `<prefix>_<precision>_b300-netperf_<fw>.sh` (always has `_<fw>`, including vllm) |
| `bench-client` (remote) | `<prefix>_<precision>_<fw>-remote-bench.sh` |

**Recipe must have.** For agentic:
- `export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k`. Use the variant without `_256k` only if the model's context is about 1M tokens. `benchmark_lib.sh` lists the allowed loaders.
- `AIPERF_SERVER_METRICS_URLS=http://localhost:$PORT/metrics` and `AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics`.
- sglang: `--enable-metrics --enable-cache-report`. vllm: never `--disable-log-stats`.
- DRAM KV offload: in the config, `kv-offloading: dram`, `kv-offload-backend: { name: hicache }`, and `dram-utilization: 0.8` (required). The generator sets `TOTAL_CPU_DRAM_GB = dram-utilization × available-cpu-dram-mib` of the pool. In the recipe, `require_agentic_kv_offload_backend hicache` and `--hicache-size` per rank at most `TOTAL_CPU_DRAM_GB / TP`; halve it again for hybrid (Mamba/GDN) models, which have two host pools.
- Local weights: `MODEL_PATH=/models/...` (host `/mnt/models`, mounted read-only). Docker launchers only; the slurm launcher mounts just the HF cache, so use the HF repo id there.

For 8k1k: copy a sibling `fixed_seq_len` recipe, including its `EVAL_ONLY`/`RUN_EVAL` block.

**New env var.** If a recipe reads a new env var, add it to the launcher's `RUN_ENV` on `vng-benchmark`.

## Check locally, then push

```bash
PY="uv run -q --python 3.11 --with pyyaml --with pydantic --no-project python"  # generator needs Python >= 3.11
bash -n <recipe>
$PY utils/matrix_logic/generate_sweep_configs.py <GEN>   # read it: conc, runner, exp-name, no eval rows
git push && git ls-tree origin/<branch> <recipe>          # recipe exists at the pushed SHA
```

## Dispatch

`<GEN>` is always `test-config`. `full-sweep` has no `--config-keys` and fans out to every key that shares the prefix.

- agentic: `test-config --config-files configs/nvidia-master.yaml --config-keys <key> --conc 20 33 50 --runner-node-filter <node> --scenario-type agentic-coding --no-evals`
- 8k1k: `test-config --config-files configs/nvidia-master.yaml --config-keys <key> --conc 8 64 --seq-lens 8k1k --scenario-type fixed-seq-len --no-evals --runner-node-filter <node>`. Drop `--no-evals` only when you want the eval jobs.

```bash
gh workflow run e2e-tests.yml --repo vngcloud/InferenceX --ref <branch> \
  -f ref=<branch> \
  -f test-name="<key> agentic CCU 20,33,50 1800s <what changed>" \
  -f generate-cli-command="<GEN>" \
  -f duration-override=1800          # agentic: 90 smoke / 1800 / 3600; 8k1k: omit
# 8k1k test-name: "<key> 8k1k conc 8,32,64 <what changed>"
gh run list --repo vngcloud/InferenceX -w e2e-tests.yml -b <branch> -L1 --json databaseId,headSha,status
```

If the branch's `e2e-tests.yml` still contains `trigger-agentic-ingest` (the branch predates its removal), add `-f skip-agentic-ingest=true` or rebase.

**Time budget.** Each job takes the duration plus 15–25 min for pull, load, and warmup: 90s is about 17 min, 1800s about 47 min, 3600s about 87 min. CCUs on one node run one after another.

## Runners

| Pool | Nodes | Notes |
|---|---|---|
| `cluster:h200-greennode` | `_01 _03 _04 _06` | Docker path, 8×H200. SSH `_01`: `stackops@103.196.239.193 -p 234` |
| `cluster:h200-greennode_04` | `_04` | 4×H200, most reliable node |
| `h200-1x` | `_03 _05` | Single-GPU 8k1k |
| `cluster:h200-greennode-slurm` | `slurm_1a`–`1d` | 4-GPU Slurm slices of the **same machine as `_06`** (han-1) |
| `cluster:b300-netperf` | `b300-netperf_00` | B300 |
| `cluster:remote-bench` | `bench-client_01` | Remote endpoint benchmarks |

## Symptom → fix

| Log line | Cause → fix |
|---|---|
| `bash: benchmarks/...sh: No such file or directory` | Recipe name does not match the launcher formula, or the recipe was not pushed |
| `required environment variables are not set: PP_SIZE` | Launcher `RUN_ENV` is missing the variable → add it on `vng-benchmark` |
| get-jobs `unrecognized arguments` / `required: --config-files` | Wrong subcommand or flag → use the `<GEN>` templates |
| enroot `401 ... registry-1.docker.io/v2/vcr.vngcloud.vn/...` | Branch has the old Slurm launcher → rebase onto `vng-benchmark` |
| `Not enough host memory ... hierarchical cache`, exit 137 | HiCache larger than free DRAM, or a docker `_06` job running at the same time as a `slurm_1x` job → resize; never run `_06` docker alongside Slurm |
| `memory capacity is unbalanced ... occupied by other processes` | A leftover process holds the GPUs → check `nvidia-smi` on the node, then rerun |
| pyxis `nvidia-container-cli: driver rpc error: timed out` | Driver flake on han-1 → rerun; if it repeats, report to the node owner |
| `Process died before .../health became ready` / `Run aborted (warmup_failure)` | Server-side failure (OOM, context length, parser) → read the server log artifact; not a runner issue |

## Debug a red run

```bash
gh run view <id> --repo vngcloud/InferenceX --json jobs --jq '.jobs[]|select(.conclusion=="failure")|"\(.databaseId) \(.name)"'
gh api repos/vngcloud/InferenceX/actions/jobs/<job>/logs | grep -nE '##\[error\]|Error|Traceback|Killed|No such file' | tail
```

Start with the lowest-CCU failing job. If every job fails the same way, the bug is in the recipe or launcher: fix it rather than rerunning. Job logs expire (HTTP 410), so download the artifacts you need while they last.

## 8k1k ladders

- Extend a ladder by setting `conc-list` to the full long-run ladder, then dispatching only the new points with `--conc`.
- Size the next points from the peak `GPU KV cache usage` in the server log (`gh run download <id> --repo vngcloud/InferenceX -n server_logs_<exp>`), not from vLLM's startup "Maximum concurrency" line.
