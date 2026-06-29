---
name: agentic-replay-run
description: Configure and dispatch an InferenceX agentic-replay benchmark for coding trace datasets, especially SemiAnalysis/public Weka datasets via utils/aiperf-mooncake and the AgentX scenario, plus internal MiniMax weka-v4. Use when the user wants to run, dispatch, smoke, or benchmark a model with agentic-replay, weka_trace, mooncake_trace, AIPerf trace replay, SemiAnalysis datasets, HF dataset links, or internal coding traces.
---

# Agentic-replay run

Flow: pick dataset + model/serving → write or reuse master-config entry + launch script → validate matrix → commit/push the requested branch → dispatch.

> **Inherits from the `bench-config` skill** — read it first for: script-name derivation rule, what-to-edit-where (sweepable `search-space` vs fixed serve flags), runner↔tp rule, exit 127 missing-script failure, and engine gotchas (pre-quantized fp8 → no `--quantization`).

## Intake (AskUserQuestion)

1. **Dataset** — prefer Weka coding datasets. Accept a SemiAnalysis/public HF dataset link or repo id, the default SemiAnalysis dataset, or internal MiniMax weka-v4.
2. **Model + serving config** — HF slug, engine + image, precision, TP/EP, serve flags. User typically pastes a launch line.
   - **Sanity-check `--model`/`--tokenizer`**: must be a plain HF slug (`namespace/repo`). Local paths (`/models/...`, `/mnt/...`) → strip prefix and confirm with user. Raw paths cause `OSError` on the runner.
3. **Runner** — `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100), `h200-greennode_01` (8×H200), `rtx5090-greennode_00` (1×RTX5090). Full list in `.github/configs/runners.yaml`. `search-space.tp` MUST match GPU count.
4. **Duration** — `900` is canonical AgentX; `300` is acceptable for smoke with `--unsafe-override` added by `benchmark_lib.sh`.
5. **Branch** — use the branch the user asks for. If unspecified, create `exp/<name>`. Do not switch branches when the user says to stay on `dev`.

## Datasets

| Dataset | Source | Type | AIPerf source | Scenario | How to configure |
|---|---|---|---|---|---|
| SemiAnalysis Weka default | `semianalysisai/cc-traces-weka-with-subagents-060826` | `weka_trace` | `utils/aiperf-mooncake` | `inferencex-agentx-mvp` | omit source; matrix defaults `public-dataset` |
| Any public SemiAnalysis/HF Weka | user link or repo id | `weka_trace` | `utils/aiperf-mooncake` | `inferencex-agentx-mvp` | map to `public-dataset` alias/repo id |
| Internal MiniMax weka-v4 | local repo dataset dir/file | `weka_trace` | `utils/aiperf-mooncake` | `inferencex-agentx-mvp` | set `input-file` |
| Agentic-coding | `agentic/datasets/agentic_coding_1variant_64k_150s.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | mooncake replay | set `input-file`, `no-fixed-schedule: true` |
| Gemma blend_prod | `agentic/datasets/gemma_blend_prod.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | mooncake replay | set `input-file`, `strip-trace-delays: true` |

For Weka, do not depend on committed datasets. Prefer `public-dataset` when the user gives a SemiAnalysis/HF dataset. If Weka has no `input-file` / `public-dataset`, it defaults to `semianalysis_cc_traces_weka_with_subagents_060826` (`semianalysisai/cc-traces-weka-with-subagents-060826` on HF). Weka must not set `no-fixed-schedule`, think-time, fixed-schedule, or warmup flags in the launcher; `benchmark_lib.sh` maps `custom-dataset-type: weka_trace` to `--scenario inferencex-agentx-mvp` and the AgentX flags. Archived: `minimax_claude_code_prod_v3.jsonl` — do not use unless explicitly requested.

## A) Master-config entry

Append to `.github/configs/nvidia-master.yaml`:

```yaml
<model-prefix>-<precision>-<hw>-<framework>[-<tag>]:
  image: lmsysorg/sglang:v0.5.12-cu130   # or vllm/vllm-openai:v0.21.0
  model: Namespace/ModelName             # HF slug
  model-prefix: short-name              # dashboard group; drives script name (see bench-config)
  precision: fp8
  framework: sglang                     # vllm | sglang
  runner: h200-greennode_01
  multinode: false
  scenarios:
    agentic-replay:
    - # input-file: benchmarks/single_node/agentic/datasets/<dataset>  # local/internal dataset only
      # public-dataset: semianalysis_cc_traces_weka_with_subagents_060826
      custom-dataset-type: weka_trace       # mooncake_trace only for non-Weka traces
      max-model-len: 131072
      benchmark-client: [aiperf]
      duration: 300                       # 900 for canonical AgentX; 300 smoke is OK
      # no-fixed-schedule: true           # mooncake_trace only; omit for weka_trace
      # strip-trace-delays: true           # Gemma blend_prod only
      # tokenizer: <hf-id>                 # only if served-model-name ≠ valid HF tokenizer
      search-space:
      - { tp: 8, ep: 8, conc-list: [4, 8, 16, 24, 32] }
```

For default SemiAnalysis Weka, omit both `input-file` and `public-dataset`. For a user-supplied HF dataset link/repo, set `public-dataset`. `ep:` required for MoE models; omit for dense.

## B) Launch script

Script path derived: `benchmarks/single_node/<model-prefix>_<precision>_<hw>[_<framework>].sh`.  
Reuse if serve flags match; otherwise copy closest template:
- **Mooncake** → `qwen3-4b-2507_bf16_h100_vllm.sh`
- **Weka vLLM** → `qwen3-4b-v4-weka_bf16_h200_vllm.sh`
- **Weka SGLang** → `minimaxm2.5-weka_fp8_h200_sglang.sh` or `glm5.2-ep8-deepep_fp8_h200_sglang.sh`

Change **only the serve block** and model-specific env requirements. Keep the public/local dataset source selection, `STOP_ARGS`, and `run_client_benchmark` call shape.

**MANDATORY — pin AIPerf fork** (right after `source ../benchmark_lib.sh`):
```bash
# Both mooncake_trace AND weka_trace use utils/aiperf-mooncake (thangquang09 fork,
# branch benchtool/agentx-weka). It carries the weka_trace loader AND the
# data_collector math.isfinite NaN filter, so SGLang's sglang:fwd_occupancy=NaN
# no longer drops the /metrics scrape — no runtime patch needed.
export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"
export AIPERF_VENV_DIR="${AIPERF_VENV_DIR:-/tmp/aiperf-mooncake-agentx-weka-venv}"
```
Without `AIPERF_SOURCE_DIR`, the run silently falls back to PyPI and fork patches are lost. Without a fork-specific `AIPERF_VENV_DIR`, self-hosted runners can reuse stale `/tmp/aiperf-venv` installs that predate new CLI flags. Do **not** use `utils/aiperf` (vngcloud fork) for weka any more — its `data_collector.py` uses `== float("inf")` which never catches NaN, so the SGLang runtime patch (`patches/aiperf-skip-nonfinite-server-metrics.patch`) would be required.

**MANDATORY — Weka AgentX wiring**: for `custom-dataset-type: weka_trace`, `benchmarks/benchmark_lib.sh` must add `--scenario inferencex-agentx-mvp` and AgentX flags. Do not pass Weka `--no-fixed-schedule`, `--use-think-time-only`, or warmup flags from the launcher.

SGLang serve block:
```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tp "$TP" --ep "$EP_SIZE" \
  --context-length "$MAX_MODEL_LEN" \
  <user flags> --trust-remote-code > "$SERVER_LOG" 2>&1 &
```
vLLM: use `vllm serve "$MODEL"` with `--tensor-parallel-size "$TP"` and `--max-model-len "$MAX_MODEL_LEN"`.

Add `EP_SIZE` to `check_env_vars` when using `$EP_SIZE`.

## Known-good Weka coding smoke references

Use these as patterns for future SemiAnalysis/public Weka coding runs:

| Model | Config key | Run | Notes |
|---|---|---|---|
| MiniMax-M2.5 | `minimaxm2.5-weka-fp8-h200-greennode-sglang-smoke` | https://github.com/vngcloud/InferenceX/actions/runs/28376099323 | default SemiAnalysis Weka, TP8/EP8, conc1, duration 300, ctx196608 |
| GLM-5.2-FP8 | `glm5.2-hicache-fp8-h200-sglang` / `glm5.2-weka-fp8-h200-greennode-sglang-smoke` | https://github.com/vngcloud/InferenceX/actions/runs/28279462408 | use SGLang `v0.5.14-cu130`; 0.5.12 fails GLM-5.2-FP8 weight loading |

## C) Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/<script>.sh
uv run python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml --config-keys <key>
# Expect: scenario-type=agentic-replay, ep/tp/conc correct

git add .github/configs/nvidia-master.yaml benchmarks/single_node/<script>.sh
git commit -m "feat(agentic): add <model> weka smoke"
git push origin <branch>
```

Do not touch `perf-changelog.yaml`, DCGM sidecars, or runner scripts for a one-off smoke unless the user explicitly asks. Add changelog only for a real sweep/PR trigger.

**Run naming** — `inputs[test-name]` must start with `yyyy/mm/dd`, followed by a short free-form label that identifies what's unique about this run at a glance. No fixed field order — include whatever dimensions matter: model, precision, GPU config, framework, dataset, context size, special flags, smoke vs full, etc.

```
yyyy/mm/dd  <whatever makes this run identifiable>
```

Examples:
```
2026/06/27 MiniMax-M2.5 fp8 8xH200 sglang cc-weka-v4
2026/06/27 MiniMax-M2.5 fp8 8xH200 sglang cc-weka-v4 ctx192k
2026/06/27 Gemma4-27B fp8 8xH200 vllm agentic-64k mtp smoke
2026/06/27 Qwen3-4B bf16 1xH100 vllm cc-weka-v4 gpu-mem0.9
```

Common shorthands: `cc-weka-v4` · `agentic-64k` · `gemma-blend` for the three integrated datasets.  
`NxHW` (e.g. `8xH200`) is usually worth including — strip "greennode" from the runner name.

Dispatch — **top-level `ref` MUST be the branch** (`ref=main` silently falls back to single-node lane and fails):

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=<branch> \
  -f 'inputs[ref]=<branch>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=yyyy/mm/dd <label>' \
  -f 'inputs[duration-override]=300'
```

## Watch

```bash
RUN_ID=$(gh run list --repo vngcloud/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --repo vngcloud/InferenceX --json status,jobs \
  -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```

**Confirm AIPerf fork** (not PyPI): job log must show:
```
[aiperf] CLI missing; installing from source: /workspace/utils/aiperf-mooncake
```
Both mooncake and weka now resolve to `/workspace/utils/aiperf-mooncake`. Seeing `installing aiperf==0.9.0 from PyPI` → `AIPERF_SOURCE_DIR` missing from script. Seeing `/workspace/utils/aiperf` → the script still pins the old vngcloud fork (missing the SGLang NaN fix). Seeing `Unknown parameter: --use-think-time-only` after source install → stale/shared venv or missing `benchmark_lib.sh` flag wiring; set a fork-specific `AIPERF_VENV_DIR` and verify the branch includes the benchmark_lib forwarding fix.

**grace-period**: `--benchmark-grace-period` (default 120s) = max in-flight drain after duration cutoff. 120s covers 64k–192k contexts; increase only if tail E2E latency exceeds ~120s.

**Prefix-cache hit %**: in `server_metrics_export.json` artifact (`prefix_cache_hits / prefix_cache_queries`), not `profile_export_aiperf.json`.
