---
name: aiperf-remote-benchmark
description: Prepare, verify, commit, dispatch, and inspect remote AIPerf benchmarks in InferenceX for Claude Code Weka v4 CCU sweeps, public SemiAnalysis Weka CCU sweeps, and the 2026-07-09 fixed-schedule simulation replay. Use when benchmarking any already-hosted OpenAI-compatible model endpoint through the benchmark-client GitHub Actions runner, including the default GreenNode GLM-5.2 endpoint.
---

# AIPerf remote benchmark

Use the existing `agentic-replay` remote path. Do not start a model server or choose a GPU runner: the provider already serves the model and GitHub Actions runs AIPerf on `benchmark-client`.

## Intake

Ask in this order. Do not dispatch until every applicable answer and the final settings are confirmed.

1. Ask for the exact model server launch command. Explain that it is optional metadata: if omitted, warn that the run cannot record the serving configuration, which weakens later comparisons. Never infer a command from the remote endpoint.
2. Ask for one scenario:
   - Claude Code Weka v4 CCU sweep.
   - Public SemiAnalysis Weka CCU sweep. Recommend this for a general CCU sweep.
   - Fixed-schedule replay of the 2026-07-09 simulation.
3. For either CCU sweep, ask for the exact CCU ladder. Do not ask for CCU for simulation; keep matrix `conc=1` as identity metadata and let trace timestamps schedule arrivals.
4. Ask for smoke or full:
   - Smoke: `60s` for every scenario.
   - Full CCU sweep: `900s`.
   - Full simulation: `3000s`.
5. Ask whether the target is the default GreenNode GLM-5.2 service:
   - URL: `https://maas-llm-aiplatform-hcm.api.vngcloud.vn`
   - API model: `z-ai/glm-5.2`
   - tokenizer: `zai-org/GLM-5.2`
   - GitHub secret: `GREENNODE_API_KEY`
   If not, ask for base URL, API model name, tokenizer, API key, and GitHub secret name. Never put an API key in YAML, logs, artifacts, a commit, or a command shown back to the user.
6. Warn that a shared endpoint must have exactly one remote benchmark job hitting it at a time. State these two rules before dispatch:
   - Within one run, CCU fan-out must be serialized. `test-sweep-agentic-replay` should use `max-parallel: 1` so `conc-4`, `conc-12`, and similar matrix entries do not overlap.
   - Across runs, do not dispatch on top of another run that targets the same endpoint. GitHub `concurrency` is not a reliable queue for this case.
7. Ask whether the user knows the endpoint is idle and not already used by another remote run.
   - If yes, record that confirmation and continue.
   - If no or unsure, propose that you check GitHub Actions for in-progress or pending remote runs targeting the same endpoint before submitting anything.
8. Suggest a title and ask the user to confirm it. Use `YYYY/MM/DD <model> <provider> <scenario> <CCU ladder if any> <Smoke|Full>`, adding serving details only when useful.

## Verify the endpoint

Smoke-test every provider, including GreenNode, before editing or dispatching. A repository secret cannot be read back; obtain the key from an existing local environment variable or ask the user for a temporary value.

Use `curl` with `Authorization: Bearer`, first against `/v1/models`, then send a minimal streamed chat completion to `/v1/chat/completions` with the selected API model and one output token. Adapt paths only if the provider documents a different OpenAI-compatible layout. Keep the key in an environment variable and disable shell tracing. Confirm that the requested model is accepted, not merely that the host returns HTTP 200.

Stop on authentication errors, an absent/rejected model, non-OpenAI-compatible responses, or repeated transport failures. Report the failure without exposing response headers or secrets. Do not spend a runner slot on an unreachable endpoint.

## Verify no overlapping remote run

Before dispatch, treat shared-endpoint exclusivity as a hard gate: one endpoint, one active remote benchmark job.

- If the user already confirmed the endpoint is idle, still repeat the warning in the final pre-dispatch confirmation.
- If the user is unsure, inspect GitHub Actions before submitting. Check the relevant workflow runs and jobs for any in-progress or pending remote AIPerf work that targets the same endpoint or same remote config family.
- Do not rely on GitHub `concurrency` to queue more than one waiting run for the shared endpoint. It can leave one run pending and cancel a later one instead of forming a true FIFO queue.
- If another matching run is active or pending, stop and tell the user which run must finish first. Do not dispatch a second overlapping run.
- If no overlapping run is found, say that the check was performed and proceed.

## Configure the run

Create an `exp/aiperf-remote-<date>-<slug>` branch from the user's current base. Preserve unrelated working-tree changes and stage only benchmark files.

Treat `(provider, API model, scenario/dataset)` as the stable config identity. Search `.github/configs/nvidia-master.yaml` before editing:

- If that identity already has a config, reuse its config key and `model-prefix`; edit it only on the experiment branch. This applies equally to GreenNode GLM-5.2, an existing DeepSeek config, or any other existing model.
- If no matching identity exists, create one config by copying the closest template. Give it a stable provider/model/scenario key and `model-prefix` suitable for later runs.
- Never repurpose a config belonging to another provider, model, or scenario. For example, a new DeepSeek benchmark must not mutate a GLM-5.2 config.

Use these as known templates:

| Scenario | Template config |
|---|---|
| Public SemiAnalysis CCU sweep | `glm5-2-greennode-bench-client-remote` or the HF-Weka smoke template `glm5-2-greennode-weka-hf-062126-remote-smoke` |
| Claude Code Weka v4 CCU sweep | `glm5-2-greennode-claude-code-weka-v4-remote-smoke` |
| Simulation smoke | `glm5-2-greennode-historical-fixed-remote-smoke` |
| Simulation full | `glm5-2-greennode-historical-fixed-remote` |

Edit only the reused or newly created config. Keep these invariants:

- `image: docker.io/thangquang09/aiperf:weka` (prebuilt AIPerf image; runner pulls it and `install_agentic_deps` skips the pip install. Rebuild with `make docker-push` in `utils/aiperf-mooncake` after aiperf code changes.)
- `runner: benchmark-client`
- `framework: api`
- `benchmark-client: [aiperf]`
- `custom-dataset-type: weka_trace`
- Public sweep dataset default: use the generic HuggingFace Weka loader with:
  ```yaml
  public-dataset: weka_hf
  hf-weka-repo: semianalysisai/cc-traces-weka-062126
  ```
- Weka v4 dataset: `input-file: benchmarks/single_node/agentic/datasets/minimax_cc_v4_weka`
- Simulation dataset: `input-file: benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions` with `fixed-schedule: true`
- Simulation search space: `{ tp: 1, ep: 1, conc-list: [1] }`
- Sweep search space: `{ tp: 1, ep: 1, conc-list: [<confirmed ladder>] }`
- Put the confirmed duration in YAML. Leave `duration-override` empty during dispatch so committed config and executed config agree.
- If supplied, store the launch command under `remote.server-command: |`. If omitted, remove that field and repeat the metadata warning in the confirmation.
- Set `remote.url`, `model`, `tokenizer`, and `api-key-secret-name` to the confirmed provider values.

For a newly published compatible Weka HuggingFace corpus, do **not** put the
HF repo directly in `public-dataset`. Use `public-dataset: weka_hf` plus
`hf-weka-repo: <org/repo>`. The matrix generator forwards this as
`PUBLIC_DATASET=weka_hf` and `HF_WEKA_REPO=<org/repo>`, and `_remote_replay.sh`
must build:

```bash
--public-dataset weka_hf --hf-weka-repo <org/repo>
```

If a run fails during dataset configuration with
`--public-dataset weka_hf requires --hf-weka-repo`, the repo field was dropped
somewhere in the workflow plumbing. Check all of these paths before dispatching
again: `.github/workflows/e2e-tests.yml`, `.github/workflows/benchmark-tmpl.yml`,
`runners/launch_remote.sh`, `benchmarks/single_node/agentic/_remote_replay.sh`,
and `benchmarks/benchmark_lib.sh` for the docker replay path. This path was
validated on 2026-07-17 in
[run 29588955806](https://github.com/vngcloud/InferenceX/actions/runs/29588955806):
the log showed `Loading HuggingFace dataset 'semianalysisai/cc-traces-weka-062126'`
at revision `23f152f6f0f9399a85901b89a6458def0ef16729`, and the workflow
completed successfully.

Do not add client-side context filtering unless the user explicitly requests it. A server command's context length is metadata, not proof of live server state.

## Validate and confirm

Run:

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/agentic/_remote_replay.sh
uv run python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys <selected-config-key>
```

Inspect the generated matrix and show the user: config key, image, runner, provider URL, API model, tokenizer, dataset (`public-dataset` plus `hf-weka-repo` when present, or `input-file`), fixed schedule status, CCU ladder, duration, secret name, whether server-command metadata is present, and the shared-endpoint warning state:

- whether `test-sweep-agentic-replay` is serialized with `max-parallel: 1`
- whether another run against the same endpoint was checked
- whether the endpoint is clear for dispatch

Show the suggested Actions title. Require explicit confirmation before commit, push, or dispatch.

## Commit, push, and dispatch

Use a conventional commit title. The commit body must contain:

- `Scenario:` with dataset, CCUs/fixed schedule, and duration.
- `Server command:` followed by the exact supplied command, or `Not provided`.

Never include the API key. Run GitNexus change detection when required by repository instructions, stage only the selected config and any directly required remote replay change, commit, and push the `exp/...` branch.

Dispatch `.github/workflows/e2e-tests.yml` with both the workflow ref and tested ref set to that committed branch. Using `main` for either ref is incorrect for this workflow.

```bash
gh api -X POST \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='<exp-branch>' \
  -f 'inputs[ref]=<exp-branch>' \
  -f 'inputs[test-name]=<confirmed-title>' \
  -f 'inputs[generate-cli-command]=test-config --config-files .github/configs/nvidia-master.yaml --config-keys <selected-config-key>' \
  -f 'inputs[duration-override]='
```

Find the run by title and branch rather than assuming the newest repository run. Wait only until `get-jobs` finishes and the expected matrix jobs appear. Confirm their CCUs or fixed-schedule identity, runner type, run head SHA, and queued/in-progress status. Then stop polling and tell the user: `Run đã chạy tại: <url>`.

If the endpoint is shared, do not dispatch a second run until the previous matching run finishes. Sequential submission is the safe default.

Do not wait for benchmark completion in the dispatch turn. Analyze results only after the user later reports that the run is finished or explicitly asks for status/results.

## Audit a completed run

When the user reports completion, verify that every matrix and collection job finished, then inspect logs and artifacts. A run is valid only when:

- Every expected CCU/fixed-schedule job exists and uses `benchmark-client`.
- The checked-out SHA equals the pushed commit.
- Profiling completes with usable records; report warmup errors, profiling errors, cancellations, and grace-period timeouts separately.
- Raw `agentic_*` artifacts contain AIPerf exports such as `benchmark_command.txt`, `benchmark.log`, `profile_export_aiperf.json`, CSV/timeslices, and record JSONL.
- `bmk_agentic_*` contains the normalized InferenceX result and `results_bmk/agg_bmk.json` contains the aggregate.
- When a server command was supplied, normalized results contain that exact command with `server_config_source: user-provided`.

Treat duration-boundary cancellations separately from request errors. A successful smoke validates plumbing, dataset loading, and endpoint behavior; do not present its throughput as a production performance conclusion.
