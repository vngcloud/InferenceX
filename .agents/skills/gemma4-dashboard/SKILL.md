---
name: gemma4-dashboard
description: Import finished gemma4 e2e runs into the Gemma Optimization Journey dashboard, for the SpeedBench scenarios (gemma4sba* recipes, aiperf ISL 8192/OSL 1024, high and low entropy) and the tau2 bench scenario (gemma4tau2 agentic recipes). Downloads the run artifacts, recomputes avg/p50/p90/p95/p99/max for TTFT, TPOT, ITL and E2E from the per-request exports, reads pool size, retracts and peak running slots from the server logs, stores everything in docs/gemma4-configs.json, and rebuilds docs/gemma-optimization-journey.html. Use when asked to add a run or config to the gemma dashboard, refresh it after new runs, change which configuration is shown as best, or compare a new run against the existing configs.
---

# Gemma4 dashboard import

`docs/gemma4-configs.json` is the single source of data: `meta` (units, `caveats` for SpeedBench, `caveats_t2` for tau2, `best`), `configs` (one per serving configuration) and `cells` (one per config × arm × cat × conc, with `stats`). `docs/gemma-optimization-journey.html` is generated from it plus `dashboard_template.html`. Do not edit the html; do not recompute stats by hand.

The page has three scenario tabs: SpeedBench high entropy 8k (`cat` hi), SpeedBench low entropy 8k (`cat` lo) and tau2 bench (`cat` t2). A config with `"scenario": "tau2"` belongs to the tau2 tab (MTP only, conc 1/8/16/32, `arm` mtp); every other config is a SpeedBench config with 16 cells. Each tab opens with a "best configuration" card driven by `meta.best.<hi|lo|t2>` = `{cfg, arm, ref, why}`: `cfg` is the config shown, `ref` the reference it is compared with, `why` the justification. Pick `cfg` from the data (summed throughput over the concs, plus retracts), and re-check it after every import.

Script (stdlib only, needs a logged-in `gh`, run with `-I`):

```bash
S=.agents/skills/gemma4-dashboard/gemma4_dash.py
python3 -I $S list                                   # configs and how many of the 16 cells each has
python3 -I $S new-config --id r015 --copy-from B --set 'short=SGL 256k·c8k r0.15' --set ratio=0.15 --set commit=6ecff88c
python3 -I $S import --config r015 [--key t15] <run-id>...
python3 -I $S new-config --id t2x --set scenario=tau2 --set short=... --set label=... --set engine=sglang --set context=262144 --set radix=true --set mem=0.92 --set chunk=8192 --set ratio=0.2 --set evict=32 --set mtp_max_running=32 --set commit=<sha> --set image=<image> --set note=...
python3 -I $S import-tau2 --config t2x <run-id>...   # tau2 agentic runs: conc from the aggregate json, stats from the profiling phase
python3 -I $S build --copy-to /home/phucnlt2/InferenceX/docs
```

## Steps

1. **Wait for the runs** (`gh run view <id> --repo vngcloud/InferenceX`). Artifacts expire, so import soon after a run finishes; imported stats stay in the json.
2. **Config.** If the run is a new serving configuration, `new-config --copy-from <similar>` and set what differs. `--set` takes JSON or a plain string. Fields the build uses: `id short axis label engine(sglang|vllm) context radix(bool) ratio mem chunk evict extra image image_base mtp_max_running commit node note`. `radix` false means `--disable-radix-cache` (SGLang) or no prefix cache (vLLM); the serve command shown in the tooltips is generated from these fields, so keep them true to the recipe.
3. **Import.** `import --config <id> <run>...`. Cells are keyed by arm/cat/conc taken from the artifacts (conc from `result.json`, because artifact names are truncated and sometimes read `conc6-<hash>`). A run that holds several configs (e.g. the ratio sweep with exp-name tokens `t15` and `t25`) needs `--key` = the text after `gemma4sba` in the artifact name; the error lists the keys. Failed cells are skipped (`--include-failed` to keep), existing cells are not overwritten (`--replace`). Read the table it prints and the `NOTE:` lines (missing server log, expired artifact).
4. **Check the numbers** against the baseline of the same arm/cat/conc (usually B or vllm256). Single runs are n=1: differences under ~3% are noise. Pool size, retracts and peak running (cell fields `full swa retract peak`) explain throughput gaps, so say so when you report.
5. **Build** with `--copy-to` the checkout the user actually opens (`/home/phucnlt2/InferenceX/docs`; a worktree copy alone goes stale). Run `node --check` on the inline script if you changed the template.
6. **Report** (Vietnamese to this user): what was imported, table vs baseline, where the html is. Add a diary entry in `docs/gemma4-optimization-diary.md` only if asked.

## tau2 runs

- Artifacts per conc: `agentic_gemma4tau2_*` (aiperf files), `bmk_agentic_gemma4tau2_*` (aggregate json: output/total throughput, cache hit, pool, power, accepted/errored counts) and `server_logs_gemma4tau2_*`; they share a 20-hex suffix, which is how `import-tau2` groups them. One run can hold several concs; use one tau2 config per recipe (engine + SGLang ratio), not per run.
- Cell fields beyond SpeedBench: `ttot` (total tok/s), `hit` (prefix-cache hit), `ok`, `err` (errored requests in the profiling phase, counted from the jsonl), `werr` (errored requests in the warmup phase; the harness aggregate `records_error_dropped` adds both, and almost all of it is warmup, so do not report it as the error rate), `power` (W). `tput` is `output_tput_tps`. Latency stats come from `profile_export.jsonl` records with `benchmark_phase == "profiling"`; the import prints a cross-check of TTFT and E2E p90 against the harness aggregate, which must match.
- The tau2 serve command shown on the page is built by `serve_cmd_tau2` from the config fields; `ratio_flag=false` means the recipe did not pass `--swa-full-tokens-ratio` (engine default 0.8).
- tau2 recipes are not SpeedBench recipes: `gemma4tau2_fp8_h200_mtp.sh` (vLLM) and `gemma4tau2_fp8_h200_sglang_mtp.sh` (SGLang); commit of each run is in the config, not the run's `headSha`.

## How the numbers are computed

- Source: `profile_export.jsonl` of the `speedbench_aiperf_*` artifact (aiperf per-request records), cancelled requests dropped.
- TTFT and E2E: per request, seconds. TPOT: per request, `(e2e - ttft) / (osl - 1)`, ms. ITL: all inter-chunk gaps of all requests pooled, ms, so its max is one worst stall and with MTP a chunk carries several tokens (compare base and MTP on TPOT, not ITL). Percentiles are linear-interpolated, same as numpy.
- `tput` is `output_throughput` of `result.json` (one GPU). `acc` is `acceptance_length` of `result.json` (vLLM) or the mean of `accept len:` in the SGLang log.
- Server log: SGLang `Full/SWA KV Cache ... #tokens`, `Retract requests`, max `#running-req`; vLLM `GPU KV cache size`, `preempt`, max `Running: N reqs` (sampled every ~10 s, so peak is a lower bound).

## Gotchas

- Serve commands are reconstructed from config fields, not read from the run. For a new config, compare them with `server_args` in the server log once.
- Configs with fewer than 16 cells (e.g. MTP c64-only sweeps) leave gaps in the charts; only the JS syntax of this case was checked, not a browser render.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` in the vLLM command text is ignored by vLLM v0.30.0 (the comment line says so); it is kept as in the recipes.
