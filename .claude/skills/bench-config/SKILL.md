---
name: bench-config
description: "How to add or edit an InferenceX benchmark config and dispatch a sweep. Use whenever the user wants to benchmark a model, add/change a config in .github/configs/*-master.yaml, switch the serving engine (vLLM/SGLang/TRT), change sweep params (concurrency, seq-len, max-num-batched-tokens, speculative decoding), change the GPU/card count, add a new model, or kick off / debug an e2e benchmark run. Triggers on phrases like 'collect perf for <model>', 'add a bench config', 'run it on N cards', 'compare vLLM vs SGLang', 'dispatch the sweep', or an 'exit 127 / Launch job script' failure."
---

# InferenceX benchmark config

**Flow:** `.github/configs/<vendor>-master.yaml` → `generate_sweep_configs.py` → `e2e-tests.yml` → `benchmark-tmpl.yml` → `runners/launch_<family>.sh` → `benchmarks/single_node/<script>.sh` → dashboard. Config files: `nvidia-master.yaml`, `amd-master.yaml`.

## Config entry

```yaml
gemma4-fp8-h100-2x-sglang:               # KEY: used in --config-keys
  image: lmsysorg/sglang:v0.5.12-cu130   # engine + version
  model: google/gemma-4-31B-it           # HF slug
  model-prefix: gemma4                    # dashboard group + script-name part
  runner: h100-2x                         # physical box (→ runners.yaml)
  precision: fp8                          # script-name part
  framework: sglang                       # engine tag → script suffix
  multinode: false
  scenarios:
    fixed-seq-len:
    - { isl: 1024, osl: 1024, search-space: [ { tp: 2, conc-start: 4, conc-end: 16, spec-decoding: none } ] }
```

**Launch-script name is DERIVED, not in the config.** For h100-greennode:
`benchmarks/single_node/{model-prefix}_{precision}_h100[_{framework}].sh` — framework-tagged preferred, else engine-less fallback (`gemma4_fp8_h100_sglang.sh` → else `gemma4_fp8_h100.sh`). Other launchers (b200/b300) also add `_mtp`/`_trt`; read the specific `runners/launch_<family>.sh`. Scripts read all params from env and `source ../benchmark_lib.sh`.

## What to edit

| Change | Edit | Note |
|---|---|---|
| **Engine** (vLLM↔SGLang↔TRT) | `framework:` + `image:` | Also write a `{prefix}_{prec}_{hw}_{framework}.sh` script — serve command differs. |
| **Sweep params** (conc, seq-len, mnbt, spec N, tp) | `search-space` + `isl`/`osl` | No script edit. Sweepable: `tp`, `conc-start/end`, `spec-decoding`, `num-speculative-tokens`, `max-num-batched-tokens`, `ep`, `dp-attn`. Add new knobs as typed fields, not model-prefix variants. |
| **Fixed serve flag** (gpu-util, mem-fraction, kv-dtype) | the launch script | Hard-coded, not sweepable. |
| **Card count** | `runner:` **and** `tp:` together | `runner` = box (h100-1x→1 GPU, h100-2x→2 GPU); `tp` = GPUs used; keep consistent. Same script. |
| **New model** | `model:` + new `model-prefix:` | New prefix → new script name → add a script. |

`runner`→node map in `.github/configs/runners.yaml`.

## Two silent-failure rules
1. **A new config needs a matching launch script** — else the job dies at "Launch job script" with **exit 127** (`...sh: not found`).
2. **`runner` ≠ card count** — set `runner` + `tp` together.

## Validate, then dispatch
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/<script>.sh
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml --config-keys <key>   # prints job matrix
```
Verify the HF slug exists (and `config.json` arch/quant) before dispatch. Dispatch target is **`vngcloud/InferenceX`** (not SemiAnalysisAI); workflow runs from `main`, repo-under-test is the branch:
```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=main -f 'inputs[ref]=<branch>' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=<label>'
```

## Watch / debug
```bash
gh run view <run-id> --repo vngcloud/InferenceX --json status,jobs \
  -q '.jobs[] | "\(.status)/\(.conclusion // "-")  \(.name)"'
```
- ~20s fail at "Launch job script" = exit 127 = missing/misnamed script. Multi-minute fail = model load / OOM / arch — read `server.log` (`grep -iE 'error|traceback|architectur|out of memory'`).
- Live `server.log` is **not** readable mid-run (API returns `BlobNotFound`) — only after the job completes.

## Engine gotchas
- **SGLang + multimodal + on-the-fly fp8** crashes in the vision tower (`triton_scaled_mm` `scale_b` assert in `gemma4_vision.py`; server never healthy). Fix: serve a pre-quantized compressed-tensors checkpoint whose `quantization_config.ignore` excludes vision (e.g. `RedHatAI/gemma-4-31B-it-FP8-dynamic`) and pass **no** `--quantization`. SGLang *does* support the Gemma 4 arch.
- **Pre-quantized fp8 checkpoint** → never pass `--quantization`; vLLM/SGLang auto-detect from `quantization_config`.
