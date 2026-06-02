# Local H100 PR-like Benchmark Runbook

Run an InferenceX benchmark on the `h100` host the same way a GitHub Actions
job would, **before** opening or labeling a PR. Covers both the native
InferenceX benchmark client and the AIPerf client, for Gemma4 31B BF16 on
2× H100.

Goal: mirror `benchmark-tmpl.yml -> runners/launch_<runner>.sh` as closely as
practical without registering a self-hosted runner. `runners/launch_h100-local.sh`
is the local stand-in for the CI launcher.

---

## 0. Prerequisites (one time)

- SSH access: `ssh h100` (2× H100 80GB).
- CI mapping: `ssh h100` is the `h100-greennode_01` runner, grouped as
  `h100-2x` in `.github/configs/runners.yaml`. Do not run TP=2 Gemma4 jobs on
  the broad `h100` label because `h100-greennode_00` is `h100-1x` and exposes
  only one GPU to jobs.
- Models under `/mnt/models` (e.g. `/mnt/models/google/gemma-4-31B-it`); requires
  `newgrp benchteam`.
- A local AIPerf checkout on the host for the source/offline path, e.g.
  `/mnt/users/thanglq5/aiperf` (only needed if you don't use the PyPI path).

---

## 1. Sync code to H100

`rsync` excludes `.git`, so the synced workspace has **no** submodule machinery —
`git submodule update` is unavailable there. AIPerf source is synced separately
and passed via `AIPERF_SOURCE_DIR` (or skipped entirely with the PyPI path).

```bash
# from the local workspace
rsync -avz --exclude '.git' InferenceX/ h100:/mnt/users/thanglq5/InferenceX/
# AIPerf source (only for the source/offline path)
rsync -avz --exclude '.git' aiperf/ h100:/mnt/users/thanglq5/aiperf/
```

---

## 2. Native InferenceX benchmark (already working)

The native client (`utils/bench_serving/benchmark_serving.py`) ships inside the
workspace and needs no extra install. This path is fully working today.

```bash
ssh h100
newgrp benchteam
cd /mnt/users/thanglq5/InferenceX

bash runners/launch_h100-local.sh
```

Defaults baked into the launcher (override via env):

```bash
MODEL=/mnt/models/google/gemma-4-31B-it   SERVED_MODEL_NAME=google/gemma-4-31B-it
MODEL_PREFIX=gemma4   EXP_NAME=gemma4   FRAMEWORK=vllm   PRECISION=bf16
IMAGE=vllm/vllm-openai:v0.21.0
TP=2   CONC=4   ISL=1024   OSL=1024   MAX_MODEL_LEN=8192
RUN_EVAL=false   EVAL_ONLY=false   PORT=8888
BENCHMARK_CLIENT=inferencex_native
```

The launcher runs `benchmarks/single_node/gemma4_bf16_h100.sh`, which starts
vLLM, waits for `/health`, dispatches to the selected benchmark client, collects
GPU metrics, and exits non-zero on server/benchmark failure.

**Why native already works:** no third-party benchmark binary — the client is a
plain Python script in the workspace, so the only runtime dependency is the
serving image's Python, which every vLLM image has.

---

## 3. AIPerf benchmark (validated 2026-06-02)

Serving images do **not** ship AIPerf. `run_aiperf_benchmark -> ensure_aiperf`
(in `benchmarks/benchmark_lib.sh`) installs it at run time into an **isolated
in-container venv** (`/tmp/aiperf-venv`) so the client's dependency tree never
mutates the serving image's packages. Resolution order:

1. `aiperf` already on PATH → use as-is.
2. `AIPERF_SOURCE_DIR` is a Python project → install from that source (local/offline).
3. Otherwise → `pip install aiperf==${AIPERF_VERSION:-0.9.0}` from PyPI (default).

### 3a. Source / offline path (what was validated E2E)

```bash
cd /mnt/users/thanglq5/InferenceX
export BENCHMARK_CLIENT=aiperf
export AIPERF_SOURCE_DIR=/mnt/users/thanglq5/aiperf   # synced in step 1
bash runners/launch_h100-local.sh
```

Expected install log: `[aiperf] CLI missing; installing from source: /aiperf`,
**no** dependency-conflict warnings (it installs into the venv).

### 3b. PyPI path (mirrors real CI — no source needed)

```bash
cd /mnt/users/thanglq5/InferenceX
export BENCHMARK_CLIENT=aiperf
export AIPERF_FORCE_PYPI=1          # skip the source mount
bash runners/launch_h100-local.sh
```

Expected install log: `[aiperf] CLI missing; installing aiperf==0.9.0 from PyPI`.

The launcher runs `benchmarks/single_node/gemma4_bf16_h100.sh`, which starts
vLLM once and dispatches through `run_client_benchmark`. With
`BENCHMARK_CLIENT=aiperf`, the dispatcher drives AIPerf through
`run_aiperf_benchmark` (fixed CONC/ISL/OSL, request-count `CONC*10`, warmup
`CONC*2`), then adapts the AIPerf artifact into the InferenceX result schema.

---

## 4. Artifacts that prove success

```text
<RESULT_FILENAME>.json                                  # InferenceX result (process_result.py reads this)
<RESULT_FILENAME>_aiperf/profile_export_aiperf.json     # raw AIPerf artifact (aiperf runs only)
server.log
gpu_metrics.csv
```

Verify the result schema is consumable:

```bash
export RUNNER_TYPE=h100-local FRAMEWORK=vllm PRECISION=bf16 SPEC_DECODING=none \
       ISL=1024 OSL=1024 DISAGG=false MODEL_PREFIX=gemma4 \
       IMAGE=vllm/vllm-openai:v0.21.0 TP=2 EP_SIZE=1 DP_ATTENTION=false \
       BENCHMARK_CLIENT=aiperf
python3 utils/process_result.py
```

Validated reference numbers (Gemma4 31B, TP=2, conc=4, 1k1k): tput/GPU
~319–325 tok/s, mean TTFT ~108–119 ms, mean ITL ~14.7 ms, mean E2EL ~10.1 s.
Reproducible across native, aiperf-source, and aiperf-PyPI runs.

---

## 5. Generate the matrix the way a PR sweep would

The Gemma4 config key (`gemma4-bf16-h100-vllm`) exists in
`.github/configs/nvidia-master.yaml`. Generate its matrix locally:

```bash
python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys gemma4-bf16-h100-vllm
# -> 16 entries: benchmark-client {inferencex_native, aiperf}
#    x isl/osl {1k1k, 8k1k} x conc {4, 8, 16, 32}, tp=2
```

Each generated entry carries `model, image, tp, conc, isl, osl, max-model-len,
precision, framework, benchmark-client, runner, exp-name, ...` — the fields `benchmark-tmpl.yml`
consumes. `exp-name` (`gemma4_1k1k` / `gemma4_8k1k`) maps via the launcher to
`gemma4_bf16_h100.sh`.

---

## 6. AIPerf packaging — current decision and what remains

**Decided (validated):** runtime install into an isolated in-container venv
(`ensure_aiperf`), default source PyPI `aiperf==0.9.0`, `AIPERF_SOURCE_DIR`
override for local/offline. Keeps the single-container model and works across
vLLM/SGLang/Triton/TRT-LLM because the venv isolates client deps.

`benchmark-client` is plumbed through the fixed-seq single-node CI path
(scenario config → generator → run-sweep/e2e → benchmark-tmpl → launcher env).
The Gemma4 config opts into both native and AIPerf clients so PR sweeps run the
same serving setup with both load generators.

**Why not the alternatives:**
- Backend-specific images with AIPerf preinstalled — N backends × M versions, does not scale.
- Dedicated AIPerf client container / sidecar — the clean long-term direction
  (split the adapter into run/convert modes), but deferred; the venv gets the
  isolation benefit now without 2-container orchestration.

---

## 7. Target

This runbook reproduces GitHub Actions submit-job behavior as closely as
possible on local H100 hardware: same env shape, same benchmark-script
entrypoints, same result artifacts. Native is fully validated; AIPerf is
validated end-to-end for the fixed-sequence Gemma4 path.
