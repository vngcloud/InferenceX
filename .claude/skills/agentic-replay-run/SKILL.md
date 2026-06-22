---
name: agentic-replay-run
description: Configure and dispatch an InferenceX agentic-replay benchmark on official AIPerf for one of the three integrated trace datasets (agentic-coding 64k/128k/167k, Claude-Code MiniMax production, or Gemma blend_prod), on any model + serving stack the user specifies. Use when the user wants to run / dispatch / kick off an agentic replay, mooncake_trace, or AIPerf trace-replay benchmark, or asks to benchmark a model against one of those datasets.
---

# Agentic-replay run

Flow: pick dataset + model/serving → write a master-config entry **and** its launch script → add a `perf-changelog.yaml` entry → commit on `exp/<name>` → dispatch `e2e-tests.yml`. The three datasets are AIPerf-integrated already; the work is config + (maybe) one launch script.

> **Inherits from the `bench-config` skill** — read it first. The generic mechanics live there: script-name **derivation rule**, *what-to-edit-where* (sweepable `search-space` knobs vs fixed serve flags hard-coded in the script), the `runner`↔`tp` rule, the **exit 127** missing-script failure, and the **engine gotchas** (pre-quantized fp8 → no `--quantization`; SGLang multimodal fp8 crash). This skill adds only the agentic-replay specifics: the 3 datasets, the `agentic-replay` scenario fields, `grace-period`, and the **`ref=branch`** dispatch.

## Ask the user first (AskUserQuestion)

1. **Dataset** — one of the three below.
2. **Model + serving config** — HF model id, engine (vLLM/SGLang) + image, precision, TP, and any special serve flags (gpu-mem-util, kv-dtype, quantization, etc.). The user typically pastes a `vllm serve …` / sglang launch line — that line becomes the serve block in the launch script (step B).
   - **Sanity-check the `--model` / `--tokenizer` value before using it.** A pasted launch line often carries a value that is correct on *the user's box* but wrong for InferenceX (the runner pulls from HF and validates the form). If the value is **not a plain HF slug** (`namespace/repo_name`) — e.g. an absolute filesystem path (`/models/...`, `/mnt/...`, `~/...`), a bare name with no namespace, or anything that looks copied from a local/patched setup — **do not pass it through.** Ask the user: *"`--model` is `<value>` — that's a local path; should I use the HF slug `<stripped value>` instead?"* Default/fallback to the HF-slug form (strip the leading dirs: `/models/RedHatAI/gemma-4-31B-it-FP8-block` → `RedHatAI/gemma-4-31B-it-FP8-block`). A raw path makes HF raise `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`. Same check for `--tokenizer`.
3. **Runner** — which box (`runner:` field). GreenNode options: `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100), `rtx5090-greennode_00` (1×RTX5090). The `runner:` value is the box label verbatim, and `search-space.tp` MUST match its GPU count. Full list in `.github/configs/runners.yaml`.
4. **Duration** — `900` (standard, recommended) or `90` (smoke). Passed as `duration-override`.
5. **New branch?** — recommend **yes**, `exp/<name>`. Edit + commit + dispatch from it (never `main` — see gotcha).

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

Script path is **derived** (bench-config rule): `benchmarks/single_node/<model-prefix>_<precision>_h100[_<framework>].sh`. **Reuse** if it exists and its serve flags match; otherwise **create** by copying the closest agentic-replay launcher — `qwen3-4b-2507_bf16_h100_vllm.sh` — and changing **only the serve block** to the user's command. Keep everything else verbatim: `check_env_vars`, the trace-subset/`STRIP_TRACE_DELAYS` handling, `STOP_ARGS` (duration), the **`REPLAY_ARGS` block** (`no-fixed-schedule`, `grace-period`, sampling, warmup, tokenizer passthrough), and the `run_client_benchmark` call — these wire up the agentic-replay methodology and must not be dropped.

```bash
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$CONC" \
  <user's serve flags here> --trust-remote-code > "$SERVER_LOG" 2>&1 &
```

SGLang: swap for `python -m sglang.launch_server …`, keeping `$MODEL`/`$TP`/`$PORT`/`$MAX_MODEL_LEN`/`$CONC` on the same env vars.

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

Prefix-cache hit % lives in the separate `server_metrics_export.json` artifact (`prefix_cache_hits / prefix_cache_queries`), not `profile_export_aiperf.json`.
