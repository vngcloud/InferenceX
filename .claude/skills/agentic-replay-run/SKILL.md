---
name: agentic-replay-run
description: Configure and dispatch an InferenceX agentic-replay benchmark for integrated trace datasets: mooncake_trace datasets via utils/aiperf-mooncake (agentic-coding and Gemma blend_prod) and Weka-trace datasets via utils/aiperf (MiniMax Claude Code v4 Weka). Use when the user wants to run / dispatch / kick off an agentic replay, mooncake_trace, weka_trace, or AIPerf trace-replay benchmark, or asks to benchmark a model against one of those datasets.
---

# Agentic-replay run

Flow: pick dataset + model/serving → write master-config entry + launch script → add `perf-changelog.yaml` entry → add DCGM sidecar → commit on `exp/<name>` → dispatch.

> **Inherits from the `bench-config` skill** — read it first for: script-name derivation rule, what-to-edit-where (sweepable `search-space` vs fixed serve flags), runner↔tp rule, exit 127 missing-script failure, and engine gotchas (pre-quantized fp8 → no `--quantization`).

## Intake (AskUserQuestion)

1. **Dataset** — one of the integrated datasets below.
2. **Model + serving config** — HF slug, engine + image, precision, TP/EP, serve flags. User typically pastes a launch line.
   - **Sanity-check `--model`/`--tokenizer`**: must be a plain HF slug (`namespace/repo`). Local paths (`/models/...`, `/mnt/...`) → strip prefix and confirm with user. Raw paths cause `OSError` on the runner.
3. **Runner** — `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100), `h200-greennode_01` (8×H200), `rtx5090-greennode_00` (1×RTX5090). Full list in `.github/configs/runners.yaml`. `search-space.tp` MUST match GPU count.
4. **Duration** — `900` (full, warmup=20) or `90` (smoke, warmup=2). For smoke: set `--warmup-request-count "${WARMUP_REQUEST_COUNT:-2}"` in the script — `WARMUP_REQUEST_COUNT` is not in the launcher's `RUN_ENV` allowlist so it must be hardcoded.
5. **New branch?** — recommend `exp/<name>` (never dispatch from `main` — see dispatch section).

## Datasets

| Dataset | Path | Type | AIPerf source | Think-time | Extra flag |
|---|---|---|---|---|---|
| Agentic-coding | `agentic/datasets/agentic_coding_1variant_64k_150s.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | yes | — |
| Gemma blend_prod | `agentic/datasets/gemma_blend_prod.jsonl` | `mooncake_trace` | `utils/aiperf-mooncake` | no | `strip-trace-delays: true` |
| MiniMax CC v4 Weka | `agentic/datasets/minimax_cc_v4_weka/` | `weka_trace` | `utils/aiperf` | yes | dir input, cap inter-turn delays |

All: `no-fixed-schedule: true`. Archived: `minimax_claude_code_prod_v3.jsonl` — do not use unless explicitly requested.

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
    - input-file: benchmarks/single_node/agentic/datasets/<dataset>
      custom-dataset-type: mooncake_trace   # or weka_trace
      max-model-len: 131072
      benchmark-client: [aiperf]
      no-fixed-schedule: true
      # strip-trace-delays: true           # Gemma blend_prod only
      # tokenizer: <hf-id>                 # only if served-model-name ≠ valid HF tokenizer
      search-space:
      - { tp: 8, ep: 8, conc-list: [4, 8, 16, 24, 32] }
```

`duration` is omitted — overridden at dispatch. `ep:` required for MoE models; omit for dense.

## B) Launch script

Script path derived: `benchmarks/single_node/<model-prefix>_<precision>_<hw>[_<framework>].sh`.  
Reuse if serve flags match; otherwise copy closest template:
- **Mooncake** → `qwen3-4b-2507_bf16_h100_vllm.sh`
- **Weka** → `qwen3-4b-v4-weka_bf16_h200_vllm.sh`

Change **only the serve block**. Keep verbatim: `check_env_vars`, `STOP_ARGS`, `REPLAY_ARGS` block, `run_client_benchmark` call.

**MANDATORY — pin AIPerf fork** (right after `source ../benchmark_lib.sh`):
```bash
# mooncake_trace
export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"
# weka_trace
# export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf"
```
Without this, the run silently falls back to PyPI and fork patches are lost.

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

## C) DCGM sidecar (always on for GreenNode)

Edit `runners/launch_<hw>-greennode.sh` — paste **right before** the `docker run --rm \` line:

```bash
DCGM_IMAGE="${DCGM_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04}"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-greennode}"
docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" "$DCGM_IMAGE"
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT
```

Commit on the same `exp/<name>` branch. If port 9400 is already held (`ss -ltn | grep 9400`), surface the conflict — don't retry blindly.

## D) perf-changelog.yaml

Append-only, exact whitespace:

```yaml
- config-keys:
    - <your-key>
  description:
    - "Agentic-replay <dataset> for <model> (<engine> <precision> TP<n>) on <runner>"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-replay
```

## E) Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/<script>.sh
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml --config-keys <key>
# Expect: scenario-type=agentic-replay, ep/tp/conc correct

git switch -c exp/<name>
git add .github/configs/nvidia-master.yaml benchmarks/single_node/<script>.sh \
        runners/launch_<hw>-greennode.sh perf-changelog.yaml
git commit && git push -u origin exp/<name>
```

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
  -f ref=exp/<name> \
  -f 'inputs[ref]=exp/<name>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=yyyy/mm/dd <label>' \
  -f 'inputs[duration-override]=900'
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
Weka: expect `/workspace/utils/aiperf`. Seeing `installing aiperf==0.9.0 from PyPI` → `AIPERF_SOURCE_DIR` missing from script.

**grace-period**: `--benchmark-grace-period` (default 120s) = max in-flight drain after duration cutoff. 120s covers 64k–192k contexts; increase only if tail E2E latency exceeds ~120s.

**Prefix-cache hit %**: in `server_metrics_export.json` artifact (`prefix_cache_hits / prefix_cache_queries`), not `profile_export_aiperf.json`.
