---
name: agentic-replay-run
description: Configure and dispatch an InferenceX agentic-replay benchmark on GreenNode's pinned-v0.9.0 AIPerf fork (utils/aiperf-mooncake submodule) for one of the three integrated trace datasets (agentic-coding 64k/128k/167k, Claude-Code MiniMax production, or Gemma blend_prod), on any model + serving stack the user specifies. Use when the user wants to run / dispatch / kick off an agentic replay, mooncake_trace, or AIPerf trace-replay benchmark, or asks to benchmark a model against one of those datasets.
---

# Agentic-replay run

Flow: pick dataset + model/serving → write a master-config entry **and** its launch script → add a `perf-changelog.yaml` entry → commit on `exp/<name>` → dispatch `e2e-tests.yml`. The three datasets are AIPerf-integrated already; the work is config + (maybe) one launch script.

> **Inherits from the `bench-config` skill** — read it first. The generic mechanics live there: script-name **derivation rule**, *what-to-edit-where* (sweepable `search-space` knobs vs fixed serve flags hard-coded in the script), the `runner`↔`tp` rule, the **exit 127** missing-script failure, and the **engine gotchas** (pre-quantized fp8 → no `--quantization`; SGLang multimodal fp8 crash). This skill adds only the agentic-replay specifics: the 3 datasets, the `agentic-replay` scenario fields, `grace-period`, and the **`ref=branch`** dispatch.

## Ask the user first (AskUserQuestion)

1. **Dataset** — one of the three below.
2. **Model + serving config** — HF model id, engine (vLLM/SGLang) + image, precision, TP, and any special serve flags (gpu-mem-util, kv-dtype, quantization, etc.). The user typically pastes a `vllm serve …` / sglang launch line — that line becomes the serve block in the launch script (step B).
   - **Sanity-check the `--model` / `--tokenizer` value before using it.** A pasted launch line often carries a value that is correct on *the user's box* but wrong for InferenceX (the runner pulls from HF and validates the form). If the value is **not a plain HF slug** (`namespace/repo_name`) — e.g. an absolute filesystem path (`/models/...`, `/mnt/...`, `~/...`), a bare name with no namespace, or anything that looks copied from a local/patched setup — **do not pass it through.** Ask the user: *"`--model` is `<value>` — that's a local path; should I use the HF slug `<stripped value>` instead?"* Default/fallback to the HF-slug form (strip the leading dirs: `/models/RedHatAI/gemma-4-31B-it-FP8-block` → `RedHatAI/gemma-4-31B-it-FP8-block`). A raw path makes HF raise `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`. Same check for `--tokenizer`.
3. **Runner** — which box (`runner:` field). GreenNode options: `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100), `rtx5090-greennode_00` (1×RTX5090). The `runner:` value is the box label verbatim, and `search-space.tp` MUST match its GPU count. Full list in `.github/configs/runners.yaml`.
4. **Duration** — `900` (standard, recommended) or `90` (smoke). Passed as `duration-override`. **For a smoke (`90`), also drop warmup to 2 requests in the launch script (step B).** Warmup runs *before* the profiling window; the launcher default of 20 can eat the whole short run — observed: 20 warmup reqs = ~750 s on a 31B/131072-ctx model, leaving ~6 profiling reqs in a 90 s smoke.
5. **New branch?** — recommend **yes**, `exp/<name>`. Edit + commit + dispatch from it (never `main` — see gotcha).
6. **Bật DCGM không?** — mặc định **không**. Bật thì AIPerf sẽ thu thêm GPU telemetry phong phú hơn nhiều so với `gpu_metrics.csv` mặc định (`gpu_metrics.csv` chỉ có power/temp/util/clocks từ `nvidia-smi`; DCGM mở thêm `DCGM_FI_PROF_*`: SM/tensor-core activity, memory bandwidth, NVLink, …) vì AIPerf scrape trực tiếp endpoint DCGM. Nếu user cần → sửa launch script của runner để dựng container DCGM (xem **section DCGM** bên dưới); không cần → bỏ qua, giữ nguyên launcher.

## Datasets

| Dataset | File under `benchmarks/single_node/agentic/datasets/` | Think-time | Extra flag |
|---|---|---|---|
| Agentic-coding | `agentic_coding_1variant_64k_150s.jsonl` (64k tier committed; other tiers must be added to this dir first) | **yes** | — |
| Claude-Code MiniMax production | `minimax_claude_code_prod_v3.jsonl` | **yes** | — |
| Gemma blend_prod | `gemma_blend_prod.jsonl` | **no** (back-to-back) | `strip-trace-delays: true` |

All three: `custom-dataset-type: mooncake_trace`, `no-fixed-schedule: true`. Think-time datasets replay recorded inter-turn delays (capped). Gemma is single-turn with no `delay` field; `strip-trace-delays: true` makes the zero-think-time / pure-concurrency behaviour explicit.

## A) Master-config entry — `.github/configs/nvidia-master.yaml`

The entry is **declarative metadata only**. It does NOT contain the serve command. Append a new top-level key:

```yaml
<model-prefix>-<precision>-<hw>-<framework>[-<tag>]:   # KEY → used in --config-keys
  image: vllm/vllm-openai:v0.21.0      # engine + version the runner pulls
  model: Qwen/Qwen3-4B-Instruct-2507   # HF slug (must exist; AIPerf tokenizes with it)
  model-prefix: qwen3-4b-2507          # dashboard group AND first part of the script name (see B)
  precision: bf16                      # script-name part
  framework: vllm                      # vllm | sglang → script-name suffix
  runner: h100-greennode_00            # physical box (→ runners.yaml); set TP consistently
  multinode: false
  scenarios:
    agentic-replay:
    - input-file: benchmarks/single_node/agentic/datasets/<dataset>.jsonl
      custom-dataset-type: mooncake_trace
      max-model-len: 131072            # must cover the trace's longest turn
      benchmark-client: [aiperf]
      no-fixed-schedule: true
      # strip-trace-delays: true       # ONLY for Gemma blend_prod (back-to-back)
      # tokenizer: <hf-id>             # ONLY if served name != a valid HF tokenizer id; else omit (defaults to model)
      search-space:
      - { tp: 1, conc-list: [4] }      # tp must match the GPU count the runner provides
```

`duration` defaults to 1800 in the schema but is overridden at dispatch (`duration-override`), so leave it out. Sweepable in `search-space`: `tp`, `ep`, `dp-attn`, concurrency (`conc-list` **or** `conc-start`/`conc-end`). Fixed serve flags (gpu-mem-util, kv-dtype, quantization) go in the script — see bench-config's *what-to-edit-where* table.

## B) Launch script — holds the serve command

Script path is **derived** (bench-config rule): `benchmarks/single_node/<model-prefix>_<precision>_h100[_<framework>].sh`. **Reuse** if it exists and its serve flags match; otherwise **create** by copying the closest agentic-replay launcher — `qwen3-4b-2507_bf16_h100_vllm.sh` — adding the `AIPERF_SOURCE_DIR` export below, and changing **only the serve block** to the user's command. Keep everything else verbatim: `check_env_vars`, the trace-subset/`STRIP_TRACE_DELAYS` handling, `STOP_ARGS` (duration), the **`REPLAY_ARGS` block** (`no-fixed-schedule`, `grace-period`, sampling, warmup, tokenizer passthrough), and the `run_client_benchmark` call — these wire up the agentic-replay methodology and must not be dropped.

**MANDATORY — pin aiperf to our fork.** Right after `source ../benchmark_lib.sh`, the script MUST export `AIPERF_SOURCE_DIR` so `ensure_aiperf` installs from the `utils/aiperf-mooncake` submodule (clean fork pinned to `v0.9.0`, `thangquang09/aiperf`) into the isolated venv via `pip install <dir>` — instead of stock PyPI `aiperf==0.9.0`. All three datasets run through this path (ADR-0003). Without this export the run silently falls back to PyPI and any fork patch is lost.

```bash
source "$(dirname "$0")/../benchmark_lib.sh"

# Pin aiperf to the clean-v0.9.0 fork submodule (ADR-0003) instead of PyPI.
export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"

...
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$CONC" \
  <user's serve flags here> --trust-remote-code > "$SERVER_LOG" 2>&1 &
```

SGLang: swap for `python -m sglang.launch_server …`, keeping `$MODEL`/`$TP`/`$PORT`/`$MAX_MODEL_LEN`/`$CONC` on the same env vars.

**Smoke warmup.** For a smoke (Q4 = `90`), set warmup to **2** requests. `WARMUP_REQUEST_COUNT` is *not* in `launch_<hw>-greennode.sh`'s `RUN_ENV` allowlist, so it can't come from the config or dispatch — set it in the launch script itself: change the REPLAY_ARGS line to `--warmup-request-count "${WARMUP_REQUEST_COUNT:-2}"` (or `export WARMUP_REQUEST_COUNT=2` near the `AIPERF_SOURCE_DIR` export). Leave the default 20 for full (`900`) runs.

## perf-changelog.yaml

Append-only, **exact whitespace**, copy an existing agentic-replay entry at the tail and edit the key/description:

```yaml
- config-keys:
    - <your-key>
  description:
    - "Agentic-replay <dataset> on AIPerf for <model> (<engine> <precision> TP<n>) on <runner>"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-replay
```

## grace-period

`--benchmark-grace-period` (launcher default **120s**, env `BENCHMARK_GRACE_PERIOD`) only applies in duration mode: after the cutoff AIPerf stops sending new requests and waits up to this long for in-flight ones to finish (the rest drop). Size it to the **longest single request's E2E latency**, not the run duration — 120s is safe headroom for 64k/128k contexts. Leave it unless tail E2E exceeds ~120s.

## DCGM (optional — only if the user said yes in Q6)

DCGM is a **sidecar container on the runner**, not part of the config/script. The model server (vLLM/SGLang) already runs inside one `docker run` in the launcher with `--network host`; adding DCGM means starting a second container on the same host network so AIPerf (inside the model container) reaches `localhost:9400/metrics`. The default `gpu_metrics.csv` (nvidia-smi polling in `benchmark_lib.sh`) is untouched and keeps running in parallel.

Edit the launcher for the chosen runner — `runners/launch_<hw>-greennode.sh` (e.g. `launch_h100-greennode.sh`, `launch_rtx5090-greennode.sh`). Paste this block **right before** the model `docker run --rm \` line, then commit it on the same `exp/<name>` branch:

```bash
# DCGM exporter sidecar. Runs --network host so AIPerf inside the model
# container (also host network) reaches GPU telemetry at localhost:9400/metrics.
# SYS_ADMIN needed for DCGM_FI_PROF_* metrics; port 9400 must be free
# (conflicts with any host-level/k8s dcgm-exporter). Torn down on script exit.
DCGM_IMAGE="${DCGM_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04}"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-greennode}"
docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" "$DCGM_IMAGE"
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT
```

Notes for the agent:
- Copy verbatim. The only thing to vary is which `launch_<hw>-greennode.sh` file, matching the `runner:` chosen in Q3.
- Don't touch `benchmark_lib.sh` or the workflow — reachability on `localhost:9400` is all AIPerf needs; wiring the endpoint into the AIPerf config is the user's side.
- Reverting = delete the block (or `git checkout` the launcher).
- First-run check on the box: if a host-level/k8s dcgm-exporter already holds port 9400 (`docker ps | grep dcgm`, `ss -ltn | grep 9400`), the sidecar fails to bind — surface that instead of retrying.

## Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/<script>.sh
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml --config-keys <key>   # expect scenario-type=agentic-replay
git switch -c exp/<name> && git add -p && git commit && git push -u origin exp/<name>
```

Dispatch — **top-level `ref` MUST be the branch, not `main`** (agentic-replay routing is missing on `main`; `ref=main` silently falls back to the single-node lane and fails):

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=exp/<name> -f 'inputs[ref]=exp/<name>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=<label>' \
  -f 'inputs[duration-override]=<900|90>'
```

## Watch

```bash
RUN_ID=$(gh run list --repo vngcloud/InferenceX --workflow e2e-tests.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --repo vngcloud/InferenceX --json status,jobs -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```

**Confirm the fork was used** (not PyPI): the job log should show `ensure_aiperf` source-installing from the submodule —
```
[aiperf] CLI missing; installing from source: /workspace/utils/aiperf-mooncake
```
If you instead see `installing aiperf==0.9.0 from PyPI`, the `AIPERF_SOURCE_DIR` export is missing from the launch script (step B).

Prefix-cache hit % lives in the separate `server_metrics_export.json` artifact (`prefix_cache_hits / prefix_cache_queries`), not `profile_export_aiperf.json`.
