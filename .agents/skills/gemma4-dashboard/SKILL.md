---
name: gemma4-dashboard
description: Import one or more finished gemma4 SpeedBench e2e runs (gemma4sba* recipes, aiperf ISL 8192/OSL 1024) into the gemma4 dashboard. Downloads the run artifacts, recomputes avg/p50/p90/p95/p99/max for TTFT, TPOT, ITL and E2E from the per-request exports, reads pool size, retracts and peak running slots from the server logs, stores everything in docs/gemma4-configs.json, and rebuilds docs/gemma4-dashboard.html. Use when asked to add a run or config to the gemma4 dashboard, refresh it after new runs, or compare a new run against the existing configs.
---

# Gemma4 dashboard import

`docs/gemma4-configs.json` is the single source of data: `meta` (units, caveats), `configs` (one per serving configuration) and `cells` (one per config × arm × entropy × conc, with `stats`). `docs/gemma4-dashboard.html` is generated from it plus `dashboard_template.html`. Do not edit the html; do not recompute stats by hand.

Script (stdlib only, needs a logged-in `gh`, run with `-I`):

```bash
S=.agents/skills/gemma4-dashboard/gemma4_dash.py
python3 -I $S list                                   # configs and how many of the 16 cells each has
python3 -I $S new-config --id r015 --copy-from B --set 'short=SGL 256k·c8k r0.15' --set ratio=0.15 --set commit=6ecff88c
python3 -I $S import --config r015 [--key t15] <run-id>...
python3 -I $S build --copy-to /home/phucnlt2/InferenceX/docs
```

## Steps

1. **Wait for the runs** (`gh run view <id> --repo vngcloud/InferenceX`). Artifacts expire, so import soon after a run finishes; imported stats stay in the json.
2. **Config.** If the run is a new serving configuration, `new-config --copy-from <similar>` and set what differs. `--set` takes JSON or a plain string. Fields the build uses: `id short axis label engine(sglang|vllm) context radix(bool) ratio mem chunk evict extra image image_base mtp_max_running commit node note`. `radix` false means `--disable-radix-cache` (SGLang) or no prefix cache (vLLM); the serve command shown in the tooltips is generated from these fields, so keep them true to the recipe.
3. **Import.** `import --config <id> <run>...`. Cells are keyed by arm/cat/conc taken from the artifacts (conc from `result.json`, because artifact names are truncated and sometimes read `conc6-<hash>`). A run that holds several configs (e.g. the ratio sweep with exp-name tokens `t15` and `t25`) needs `--key` = the text after `gemma4sba` in the artifact name; the error lists the keys. Failed cells are skipped (`--include-failed` to keep), existing cells are not overwritten (`--replace`). Read the table it prints and the `NOTE:` lines (missing server log, expired artifact).
4. **Check the numbers** against the baseline of the same arm/cat/conc (usually B or vllm256). Single runs are n=1: differences under ~3% are noise. Pool size, retracts and peak running (cell fields `full swa retract peak`) explain throughput gaps, so say so when you report.
5. **Build** with `--copy-to` the checkout the user actually opens (`/home/phucnlt2/InferenceX/docs`; a worktree copy alone goes stale). Run `node --check` on the inline script if you changed the template.
6. **Report** (Vietnamese to this user): what was imported, table vs baseline, where the html is. Add a diary entry in `docs/gemma4-optimization-diary.md` only if asked.

## How the numbers are computed

- Source: `profile_export.jsonl` of the `speedbench_aiperf_*` artifact (aiperf per-request records), cancelled requests dropped.
- TTFT and E2E: per request, seconds. TPOT: per request, `(e2e - ttft) / (osl - 1)`, ms. ITL: all inter-chunk gaps of all requests pooled, ms, so its max is one worst stall and with MTP a chunk carries several tokens (compare base and MTP on TPOT, not ITL). Percentiles are linear-interpolated, same as numpy.
- `tput` is `output_throughput` of `result.json` (one GPU). `acc` is `acceptance_length` of `result.json` (vLLM) or the mean of `accept len:` in the SGLang log.
- Server log: SGLang `Full/SWA KV Cache ... #tokens`, `Retract requests`, max `#running-req`; vLLM `GPU KV cache size`, `preempt`, max `Running: N reqs` (sampled every ~10 s, so peak is a lower bound).

## Gotchas

- Serve commands are reconstructed from config fields, not read from the run. For a new config, compare them with `server_args` in the server log once.
- Configs with fewer than 16 cells (e.g. MTP c64-only sweeps) leave gaps in the charts; only the JS syntax of this case was checked, not a browser render.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` in the vLLM command text is ignored by vLLM v0.30.0 (the comment line says so); it is kept as in the recipes.
