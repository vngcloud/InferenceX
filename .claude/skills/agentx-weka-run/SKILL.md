---
name: agentx-weka-run
description: Configure and dispatch an InferenceX agentx-weka benchmark — the SemiAnalysis `inferencex-agentx-mvp` scenario replaying the `semianalysisai/cc-traces-weka-no-subagents-051226` corpus (949 Claude-Code traces) through the vngcloud `utils/aiperf` fork — on any model + serving stack the user specifies. Use when the user wants to run / dispatch / smoke the agentx-weka, agentx-mvp, cc-traces-weka, or weka public-dataset benchmark. NOT for mooncake_trace / agentic-coding-64k/128k / Gemma-blend replays — those are the separate `agentic-replay-run` skill (different aiperf submodule + scenario).
---

# Agentx-weka run

Drives the official SemiAnalysis `inferencex-agentx-mvp` scenario over the upstream weka corpus, resolved + invoked entirely through aiperf (`resolve_trace_source` + `build_replay_cmd` in `benchmark_lib.sh`). The only load knob is `--concurrency`; the scenario locks cache-bust, inter-turn-delay-cap, ignore_eos, live-assistant mode, etc. Flow: master-config entry **and** launch script → `perf-changelog.yaml` entry (`scenario-type: agentic-coding`) → commit on `exp/<name>` → dispatch `e2e-tests.yml` with `ref=<branch>`.

> **Inherits from `bench-config`** — read it first for the generic mechanics: script-name derivation, *what-to-edit-where* (sweepable `search-space` vs fixed serve flags in the script), the `runner`↔`tp` rule, the **exit 127** missing-script failure, and engine gotchas.

> **Sibling skill `agentic-replay-run`** covers local trace replay (`mooncake_trace` and local `weka_trace`, `utils/aiperf-mooncake` fork, `scenario-type: agentic-replay`). This skill is only for the public AgentX Weka dataset path (`--public-dataset semianalysis_cc_traces_weka`) through `utils/aiperf`. The shared DCGM-sidecar section and dispatch/watch mechanics in that skill apply verbatim here.

## Ask the user first (AskUserQuestion)

1. **Model + serving config** — HF model id, engine (vLLM/SGLang) + image, precision, TP, special serve flags. User usually pastes a `vllm serve …` / sglang line → becomes the serve block in the script.
   - **Sanity-check `--model` / `--tokenizer`**: must be a plain HF slug (`namespace/repo`), not a local path (`/mnt/...`, `/models/...`). A raw path makes HF raise `OSError: Repo id must be in the form ...`. Strip leading dirs and confirm. (Same rule as `agentic-replay-run` Q2.)
2. **Runner** — `runner:` field, box label verbatim; `search-space.tp` MUST match its GPU count. GreenNode: `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100), `rtx5090-greennode_00` (1×RTX5090). Full list in `.github/configs/runners.yaml`.
3. **Duration** — `900` (standard) or `90` (smoke). Set in `scenarios.agentic-coding.duration` (NOT `duration-override` for this path).
4. **Concurrency ladder** — `conc-list` in `search-space` (e.g. `[2, 4]` smoke, `[8, 16, 32, ...]` capacity).
5. **Trace count** — full 949 (`900s` capacity run) or a smoke subset via `WEKA_NUM_DATASET_ENTRIES` (e.g. 64). See **Gotcha 3**.
6. **New branch?** — yes, `exp/<name>`. Commit + dispatch from it (never `main`).
7. **DCGM?** — default no. If yes, see the DCGM section in `agentic-replay-run` (identical sidecar block; only varies which `runners/launch_<hw>-greennode.sh`).

## A) Master-config entry — `.github/configs/nvidia-master.yaml`

```yaml
<model-prefix>-weka-<hw>-<framework>[-<tag>]:   # KEY → used in --config-keys / --model-prefix
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507    # HF slug; aiperf tokenizes with it
  model-prefix: qwen3-4b-weka           # dashboard group AND first part of script name (see B)
  precision: bf16
  framework: vllm                       # vllm | sglang → script-name suffix
  runner: h100-greennode_00             # → runners.yaml; tp MUST match its GPU count
  multinode: false
  scenarios:
    agentic-coding:                     # <- weka is scenario-type agentic-coding, NOT agentic-replay
      duration: 90                      # 90 smoke / 900 standard
      search-space:
        { tp: 1, ep: 1, offloading: none, conc-list: [2, 4] }
```

`offloading: none` only — CPU/SSD KV offload is not wired for the weka launch path. The corpus, scenario, and live-assistant flags are all resolved inside aiperf; there is **no `input-file`** (unlike mooncake).

## B) Launch script — `benchmarks/single_node/agentic/<model-prefix>_<precision>_h100_<framework>.sh`

Two working references — **copy the one matching the engine**:
- **SGLang** → `minimaxm2.5-weka_fp8_h100_sglang.sh`
- **vLLM** → `qwen3-4b-weka_bf16_h100_vllm.sh`

Both: `check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR`, then `resolve_trace_source` → `install_agentic_deps` → `build_replay_cmd "$RESULT_DIR"` → `$REPLAY_CMD | tee benchmark.log` → `write_agentic_result_json`. Change **only the serve block** to the user's command; keep the rest verbatim. The submodule `utils/aiperf` is already pinned (vngcloud fork) — **no `AIPERF_SOURCE_DIR` export** here (that is the mooncake skill's mechanism; weka uses the submodule directly).

**The vLLM vs SGLang difference is structural — see Gotchas 1 & 2 before writing the script.**

## Gotchas (this path's hard-won failure modes)

**1. vLLM must isolate aiperf in a clean venv.** `install_agentic_deps` upgrades aiperf's web stack (anyio / starlette / fastapi). SGLang **tolerates** the upgrade — it installs globally and runs fine. vLLM v0.21.0 does **not**: its API server imports the new anyio/starlette lazily *while serving* → `_IncludedRouter has no attribute 'path'` on `/health`, then `cannot import name 'TaskHandle' from anyio._core._tasks` on the first request → aiperf sees 100% errors and `--failed-request-threshold` aborts. Fix: install aiperf into a **clean venv (no `--system-site-packages`)** so vLLM keeps the image's untouched system python; they share only the localhost socket:

```bash
# (vLLM only) after the server is healthy:
AIPERF_VENV="${TMPDIR:-/tmp}/aiperf-venv"   # /tmp, NOT /workspace (no new dirs in /workspace)
python3 -m venv "$AIPERF_VENV"
source "$AIPERF_VENV/bin/activate"
python3 -m pip install -q --upgrade pip
resolve_trace_source        # installs hf CLI into the venv
install_agentic_deps        # installs aiperf + deps into the venv
```

**2. Ordering differs by engine — and ordering alone does NOT fix vLLM.** SGLang installs deps **before** launching the server (global, in the same python). vLLM starts the server **first** (system python loads the image's web stack), then does the venv install. But starting vLLM first is *not* the fix on its own — the bad import is lazy (at request time, after the server is "ready"), so a global upgrade still breaks it. The venv (Gotcha 1) is the actual fix; the ordering just lets model-load overlap venv setup.

**3. Trace-count / load.** The corpus is 949 traces; `WEKA_NUM_DATASET_ENTRIES` caps it (default 949; `benchmark_lib.sh` line ~1423). For smokes set `export WEKA_NUM_DATASET_ENTRIES=64` before `build_replay_cmd` — plenty to exercise the path + measure prefix-cache at low conc, and loads faster. Note: aiperf's *dataset configuration* (load + reconstruct + inputs.json + mmap) takes **~145s even at 64 entries**, up to 4–14 min at full 949 — `build_replay_cmd` already bumps `AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800` to absorb it. A prior "runner dies ~16 min in, blank conclusion, no logs" was first blamed on host-RAM OOM from reconstructing all 949 trajectories — that theory was **refuted** (live RAM was 4/117 GB). The real culprits were the dep clash (Gotcha 1) **and a full `/mnt` disk** on the runner. Still, on a small-RAM / no-swap box, cap entries for smokes and don't assume full-corpus load is free.

**4. Runner `/mnt` disk fills up.** HF cache (`/mnt/hf_hub_cache`), curated models (`/mnt/models`), and docker image layers (`/mnt/containerd`) share `/dev/sdc`. A 100%-full `/mnt` makes the dataset/model download fail and the job die with a blank conclusion. Check `ssh h100 'df -h /mnt'`; reclaim with `sudo docker image prune -a -f` (stop the orphan job container **without removing it** first so its image stays referenced) — typically frees ~100 GB. HF cache and `/mnt/models/*` are large but shared infra — confirm with the user before deleting.

**5. (SGLang only) NaN server-metrics patch.** SGLang emits `sglang:fwd_occupancy=NaN` on an uninitialized gauge; orjson encodes NaN as `null`, failing aiperf's `ServerMetricsRecordMessage` validation and dropping the **entire** `/metrics` scrape (cache-hit rate included). The SGLang reference launcher applies `patches/aiperf-skip-nonfinite-server-metrics.patch` to the editable aiperf source at runtime (idempotent). Keep that block when copying the SGLang launcher. Not needed for vLLM.

## perf-changelog.yaml

Append-only, **exact whitespace**, `scenario-type: agentic-coding`:

```yaml
- config-keys:
    - <your-key>
  description:
    - "Agentx-weka (cc-traces-weka) on aiperf for <model> (<engine> <precision> TP<n>) on <runner>"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-coding
```

## Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/agentic/<script>.sh
git submodule status utils/aiperf   # MUST be the vngcloud fork, branch cjq/weka-live-assistant-responses
python3 utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml --model-prefix <model-prefix> --framework <fw>   # expect the conc list you set
git switch -c exp/<name> && git add -p && git commit && git push -u origin exp/<name>
```

Dispatch — **top-level `ref` MUST be the branch, not `main`** (GreenNode fork; the in-repo CLAUDE.md's `ref=main` example is the upstream convention — do not copy it):

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=exp/<name> -f 'inputs[ref]=exp/<name>' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --model-prefix <model-prefix> --framework <fw>' \
  -f 'inputs[test-name]=<label>' \
  -f 'inputs[duration-override]='
```

## Watch + confirm

```bash
RUN_ID=$(gh run list --repo vngcloud/InferenceX --workflow e2e-tests.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --repo vngcloud/InferenceX --json status,jobs -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```

- Job names should read `agentic / ... <model-prefix>_tp<n>_conc<c>_offloadnone ...` — confirms the agentic path (not a fixed-seq fallback).
- **Live sanity-check** on the runner: `ssh h100 'tail server.log'` should show `POST /v1/chat/completions ... 200 OK` (vLLM serving aiperf). `_IncludedRouter` / `TaskHandle` errors → Gotcha 1 (venv not isolating).
- **Source of truth = raw `profile_export_aiperf.json`** (per-conc artifact `agentic_<key>/trace_replay/`), not the log. Key fields: `output_token_throughput`, `total_token_throughput`, `output_token_throughput_per_user` (.avg/.p50/.p99), `time_to_first_token`, `inter_token_latency`, `request_count`, `error_summary` (empty = clean). **Prefix-cache hit %** is in the separate `server_metrics_export.json`.
```bash
jq -r '"reqs \(.request_count.avg) err \(.error_summary|length) | out \(.output_token_throughput.avg|floor) tot \(.total_token_throughput.avg|floor) tok/s/user \(.output_token_throughput_per_user.avg|floor) | TTFT \(.time_to_first_token.p50|floor)ms ITL \(.inter_token_latency.p50)ms"' .../profile_export_aiperf.json
```
