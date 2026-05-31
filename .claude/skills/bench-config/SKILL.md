---
name: bench-config
description: "How to add or edit an InferenceX benchmark config and dispatch a sweep. Use whenever the user wants to benchmark a model, add/change a config in .github/configs/*-master.yaml, switch the serving engine (vLLM/SGLang/TRT), change sweep params (concurrency, seq-len, max-num-batched-tokens, speculative decoding), change the GPU/card count, add a new model, or kick off / debug an e2e benchmark run. Triggers on phrases like 'collect perf for <model>', 'add a bench config', 'run it on N cards', 'compare vLLM vs SGLang', 'dispatch the sweep', or an 'exit 127 / Launch job script' failure."
---

# InferenceX benchmark config flow

How a benchmark goes from a YAML entry to numbers, and exactly what to edit
for the common changes (different engine / params / card count / model).

## The 7 layers (dispatch → numbers)

```
1. .github/configs/<vendor>-master.yaml   ← the config entry (you edit this most)
       │  one key, e.g.  gemma4-fp8-h100-2x-sglang:
       ▼
2. utils/matrix_logic/generate_sweep_configs.py
       │  expands search-space into N jobs (one per tp × conc × seqlen × spec …)
       ▼
3. .github/workflows/e2e-tests.yml         ← dispatch entrypoint (workflow_dispatch)
       ▼
4. .github/workflows/benchmark-tmpl.yml    ← runs once PER job; exports every config
       │  field as an env var; picks the node from .github/configs/runners.yaml
       ▼
5. runners/launch_<runner-family>.sh       ← chooses the launch script + does docker run
       │  e.g. runners/launch_h100-greennode.sh
       ▼
6. benchmarks/single_node/<script>.sh      ← actual `vllm serve` / `sglang.launch_server`
       │  + run_benchmark_serving (+ optional eval)
       ▼
7. server.log + result JSON → uploaded → inferencex.com dashboard
```

Config files: `nvidia-master.yaml` (NVIDIA GPUs), `amd-master.yaml` (AMD).

## Anatomy of a config entry

```yaml
gemma4-fp8-h100-2x-sglang:               # KEY: label, used in --config-keys
  image: lmsysorg/sglang:v0.5.12-cu130   # which container == engine + version
  model: google/gemma-4-31B-it           # HF slug (verify it exists, see below)
  model-prefix: gemma4                    # dashboard grouping + 1st part of script name
  runner: h100-2x                         # which physical box (→ runners.yaml)
  precision: fp8                          # middle of script name
  framework: sglang                       # engine tag (→ script filename suffix)
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 1024                           # input length
      osl: 1024                           # output length
      search-space:                       # each row × each conc = one job
      - { tp: 2, conc-start: 4, conc-end: 16, spec-decoding: none }
```

### Launch-script filename is DERIVED, not named in the config

The runner launcher builds the path from config fields. The rule differs per
runner family — for the GreenNode H100 box (`runners/launch_h100-greennode.sh`):

```
benchmarks/single_node/{model-prefix}_{precision}_h100[_{framework}].sh
```

It prefers the framework-tagged name, then falls back to the engine-less name:
- `framework: sglang`, prefix `gemma4` → tries `gemma4_fp8_h100_sglang.sh`, else `gemma4_fp8_h100.sh`
- `framework: vllm`,   prefix `gemma4` → tries `gemma4_fp8_h100_vllm.sh` (absent) → `gemma4_fp8_h100.sh`

Other launchers (b200/b300) also append a `_mtp` spec suffix and a `_trt`
framework suffix — read the specific `runners/launch_<family>.sh` before
assuming a name. The h100-greennode launcher intentionally has **no** spec
suffix because scripts like `gemma4_fp8_h100.sh` branch internally on
`$SPEC_DECODING`.

## What to edit for each change

| Want to change… | Edit | Notes |
|---|---|---|
| **Engine** (vLLM↔SGLang↔TRT) | `framework:` + `image:` | Also WRITE a matching `{prefix}_{prec}_{hw}_{framework}.sh` launch script — the serve command differs per engine. Most work. |
| **Sweep params** (conc, seq-len, mnbt, spec N, tp) | `search-space` rows + `isl`/`osl` | Sweepable knobs live in search-space — **no script edit**: `tp`, `conc-start`/`conc-end`, `spec-decoding`, `num-speculative-tokens`, `max-num-batched-tokens`, `ep`, `dp-attn`. |
| **Fixed serve flag** (gpu-util, mem-fraction, kv-dtype) | the launch script | These are hard-coded in the `.sh`, not sweepable from YAML. |
| **Card count** | `runner:` **and** `tp:` together | `runner` = which box (h100-1x → 1-GPU node, h100-2x → 2-GPU node); `tp` = GPUs the engine uses. They must agree (`tp` ≤ GPUs on that node). Same script. |
| **New model** | `model:` + new `model-prefix:` | New prefix → new derived script filename → add a launch script too. |

`runner` → node mapping lives in `.github/configs/runners.yaml`
(e.g. `h100-1x: [h100-greennode_00]`, `h100-2x: [h100-greennode_01]`).

## Two rules that cause silent failures

1. **A new config almost always needs a new launch script.** YAML-only → the
   job dies at step "Launch job script" with **exit 127** (`...sh: not found`),
   because the filename is derived from prefix+precision+framework. Always
   create the matching `benchmarks/single_node/*.sh`.
2. **`runner` ≠ card count.** `runner` picks the box; `tp` sets GPUs used. Set
   both, and keep them consistent.

## Workflow: add a config the safe way

1. **Add the YAML entry** to `.github/configs/nvidia-master.yaml`.
2. **Add/verify the launch script** at the derived path. Model it on an
   existing same-engine script:
   - vLLM single-H100 baseline → `benchmarks/single_node/gemma4_fp8_h100.sh`
   - SGLang H100 → `benchmarks/single_node/qwen3.5_fp8_h100.sh` /
     `gemma4_fp8_h100_sglang.sh`
   Launch scripts read everything from env (`MODEL`, `TP`, `CONC`, `ISL`,
   `OSL`, `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `SPEC_DECODING`,
   `NUM_SPECULATIVE_TOKENS`, `RESULT_FILENAME`, …) and `source ../benchmark_lib.sh`.
3. **Verify the HF slug exists** (avoid a wasted GPU run):
   ```bash
   curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/<org>/<model>
   # 200 = exists
   ```
   For quant/arch surprises, fetch config.json (follow redirect with -L):
   ```bash
   curl -sL https://huggingface.co/<org>/<model>/resolve/main/config.json \
     | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('architectures'),d.get('model_type'),bool(d.get('quantization_config')))"
   ```
   For a pre-quantized FP8 checkpoint, do NOT pass `--quantization` — vLLM/SGLang
   auto-detect from `quantization_config`. Multimodal models (a `vision_config`)
   enforce a `max-num-batched-tokens` floor, so very small mnbt rows may fail.
4. **Validate locally before dispatch:**
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
   bash -n benchmarks/single_node/<script>.sh
   python3 utils/matrix_logic/generate_sweep_configs.py test-config \
     --config-files .github/configs/nvidia-master.yaml \
     --config-keys <your-config-key>          # prints the full job matrix as JSON
   ```
5. **Commit + push** the branch (config + script together).

## Dispatch a sweep

Target is **`vngcloud/InferenceX`** (NOT SemiAnalysisAI — those are upstream's
examples). The workflow definition is run from `main`; the repo under test is
the feature branch:

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=main \
  -f 'inputs[ref]=<your-branch>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <your-config-key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=<human label>'
```

`generate-cli-command` is passed verbatim to `generate_sweep_configs.py`;
subcommands: `test-config` (specific keys), `full-sweep` (filter by
`--model-prefix`/`--framework`/`--runner-type`/`--seq-lens`/`--min-conc`/`--max-conc`),
`runner-model-sweep` (validate a runner type across nodes).

## Watch / debug a run

```bash
gh run list  --repo vngcloud/InferenceX --workflow e2e-tests.yml --limit 5
gh run view <run-id> --repo vngcloud/InferenceX --json status,jobs \
  -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```

- The single-GPU box runs jobs **serially**; h100-1x and h100-2x are different
  nodes, so those runs proceed in parallel.
- **Live `server.log` is NOT readable while a step is in progress** — GitHub's
  log API returns `BlobNotFound` until the job completes; only then are step
  logs and the `server.log` artifact finalized.
- A fast fail (~20 s) at "Launch job script" = exit 127 = missing/misnamed
  launch script. A failure several minutes in = model load / OOM / arch
  unsupported — read `server.log` (grep `error|traceback|architectur|out of memory`).
- Per-step status:
  ```bash
  gh run view <run-id> --repo vngcloud/InferenceX --json jobs \
    -q '.jobs[] | select(.databaseId==<job-id>) | .steps[] | "\(.number) \(.conclusion // .status) \(.name)"'
  ```

## Known engine gotchas

- **SGLang + multimodal + on-the-fly fp8 crashes in the vision tower.** For a
  VLM (e.g. Gemma 4, arch `Gemma4ForConditionalGeneration`), `--quantization fp8`
  quantizes the vision encoder too and dies in `triton_scaled_mm` (a `scale_b`
  shape `AssertionError` in `gemma4_vision.py`) — the scheduler throws and the
  server never becomes healthy (~6 min in, not a fast fail). Fix: serve a
  **pre-quantized compressed-tensors checkpoint** whose `quantization_config.ignore`
  excludes the vision tower (e.g. `RedHatAI/gemma-4-31B-it-FP8-dynamic`,
  `ignore: ['re:.*vision.*', 'lm_head', 're:.*embed_tokens.*']`) and pass **no**
  `--quantization` flag — SGLang auto-detects compressed-tensors and honours the
  ignore list (LLM fp8, vision bf16). SGLang *does* support the Gemma 4 arch.
- **Pre-quantized fp8 checkpoints**: never pass `--quantization` — both vLLM and
  SGLang read `quantization_config` from the checkpoint. Forcing it can mismatch
  the on-disk scheme.

## Notes

- New sweepable knobs should be added as **typed search-space fields** in the
  schema, not smuggled through `model-prefix` variants.
- Each new search-space column that should reach the dashboard also needs a
  matching nullable column + lookup in the inferencex.com results DB.
- Don't run engine containers / large builds on the laptop — validate with
  `bash -n`, YAML parse, and the matrix generator, then run on the GPU runners.
