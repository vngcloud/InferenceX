---
name: inferencex-agentic-dispatch
description: Configure, validate, preview, and dispatch single-node InferenceX agentic coding benchmarks from a project branch. Use when a user wants to turn an SGLang or vLLM serving command into InferenceX config and recipe files, choose smoke/full/custom duration, select the full or 256k-capped SemiAnalysis coding dataset, choose an exact runner, define a CCU sweep, require DCGM and server metrics, prevent automatic ingest by default, or safely retry a failed manual benchmark.
---

# InferenceX Agentic Dispatch

Build one reproducible agentic-only benchmark, show the exact generated matrix and server commands, and dispatch only after explicit user approval.

## 1. Inspect before asking

Read `AGENTS.md`, the current branch/status, the relevant master config, runner launcher, closest agentic recipe, and `.github/workflows/e2e-tests.yml`. Preserve unrelated changes. Do not use an installed global benchmark skill as a substitute for this project workflow.

Read [references/project-map.md](references/project-map.md) before editing. Use `scripts/preflight.py` after editing and before presenting the preview.

## 2. Gather the benchmark contract

Ask for missing choices in compact groups. Do not ask again for values already supplied.

1. Serving config: full launch command, framework/image, model identity, precision/quantization, topology, local weight path or HF repo, served model name, and any required cache/scheduler/parser flags.
2. Run mode:
   - smoke: `90` seconds
   - full: `3600` seconds
   - custom: a positive integer supplied by the user
3. Dataset:
   - full: `semianalysis_cc_traces_weka_062126`
   - cap 256k: `semianalysis_cc_traces_weka_062126_256k`
4. Exact runner node and its `cluster:*` pool. For local weights, also obtain an SSH target or another read-only way to verify the host path.
5. Ordered CCU list, for example `2` or `8,12,16`.

Default automatic ingest to **off**. Enable it only when the user explicitly requests production ingest.

## 3. Map inputs into project files

Prefer a dedicated master-config key and recipe over changing unrelated benchmarks.

- Add or update the entry in `configs/nvidia-master.yaml` or `configs/amd-master.yaml`.
- Set `multinode: false` and only `scenarios.agentic-coding`.
- Use an exact `cluster:*` runner pool and explicit `conc-list`.
- For HiCache, set `kv-offloading: dram` and `kv-offload-backend: { name: hicache }`.
- Create or update the matching `benchmarks/single_node/agentic/*.sh` recipe.
- Preserve the supplied server command. InferenceX may set `max-running-requests = 2 * CCU` and CUDA graph max batch size to `min(2 * CCU, 64)` when requested by the user or established by the existing recipe.
- Always enable server metrics and cache reporting. Always configure AIPerf server metrics and DCGM telemetry URLs.
- Ensure the selected runner launcher starts DCGM and forwards `KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`, and `KV_OFFLOAD_BACKEND_METADATA`.
- If weights are local, mount the host model root read-only at the container model root. A host path existing outside the benchmark container is insufficient.

Use `apply_patch` for edits. Do not commit yet.

## 4. Preflight deterministically

Run:

```bash
python3 .agents/skills/inferencex-agentic-dispatch/scripts/preflight.py \
  --repo . \
  --config-file configs/nvidia-master.yaml \
  --config-key <key> \
  --recipe <recipe> \
  --launcher <launcher> \
  --runner-node <exact-node> \
  --dataset <full|cap-256k> \
  --duration <seconds> \
  --ccu <comma-list> \
  [--model-container-path <path>] \
  [--model-host-path <path> --model-host-root <root> --model-container-root <root> \
   --ssh-target <user@host> --ssh-port <port>]
```

Treat any failure as blocking. In addition, run `git diff --check`, `bash -n` on changed recipes/launchers, and relevant matrix tests when available.

The preflight must establish all of these:

- exact agentic matrix and CCUs, with no eval/fixed-sequence jobs;
- correct duration override;
- dataset loader choice;
- metrics, DCGM, HiCache flags, and forwarded environment variables;
- local model visibility inside the same image/container mount used by the job;
- complete local weights with config/index and no incomplete files;
- current non-default branch can be pushed and checked out by the workflow.

## 5. Present a verification gate

Print a compact preview containing:

- branch and files changed;
- model path, served name, image, precision, runner, dataset, duration, and CCUs;
- effective server command for every CCU;
- generated matrix rows;
- telemetry status: server metrics and DCGM enabled;
- model mount/preflight result;
- exact `gh workflow run` command;
- `skip-agentic-ingest=true` prominently.

Ask the user to approve this preview. Do not commit, push, or dispatch before an explicit yes.

## 6. Commit, push, and dispatch after approval

Re-run validation, commit only the intended files, and push the current branch. Follow repository commit-message rules.

Dispatch `.github/workflows/e2e-tests.yml` from the same pushed branch and also pass that branch through the workflow `ref` input. Always include:

```text
--scenario-type agentic-coding --single-node --no-evals
duration-override=<chosen-duration>
skip-agentic-ingest=true
```

Unless the user explicitly opted into ingest, never omit or negate `skip-agentic-ingest=true`.

After dispatch, verify the run SHA equals the pushed SHA, `get-jobs` succeeds, and the job names contain exactly the requested CCUs, runner, precision, and agentic scenario. Report the run URL. Do not cancel or stop a run unless the user explicitly asks.

## 7. Failure rules

- A local path reported as an invalid Hugging Face repo ID usually means the path is absent inside the container. Verify the mount before changing the model ID.
- `KV_OFFLOAD_BACKEND is required` means the launcher did not forward the generated backend metadata. Inspect the live container environment when possible.
- A successful host download is not enough: verify the index and weight files from inside the benchmark image.
- Never silently switch dataset variants, duration, runner, image, model path, CCUs, or ingest policy.
- Do not use a successful smoke result as a substitute for the user's approval of the full-run preview.
