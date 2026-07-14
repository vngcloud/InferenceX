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
6. Suggest a title and ask the user to confirm it. Use `YYYY/MM/DD <model> <provider> <scenario> <CCU ladder if any> <Smoke|Full>`, adding serving details only when useful.

## Verify the endpoint

Smoke-test every provider, including GreenNode, before editing or dispatching. A repository secret cannot be read back; obtain the key from an existing local environment variable or ask the user for a temporary value.

Use `curl` with `Authorization: Bearer`, first against `/v1/models`, then send a minimal streamed chat completion to `/v1/chat/completions` with the selected API model and one output token. Adapt paths only if the provider documents a different OpenAI-compatible layout. Keep the key in an environment variable and disable shell tracing. Confirm that the requested model is accepted, not merely that the host returns HTTP 200.

Stop on authentication errors, an absent/rejected model, non-OpenAI-compatible responses, or repeated transport failures. Report the failure without exposing response headers or secrets. Do not spend a runner slot on an unreachable endpoint.

## Configure the run

Create an `exp/aiperf-remote-<date>-<slug>` branch from the user's current base. Preserve unrelated working-tree changes and stage only benchmark files.

Reuse the closest entry in `.github/configs/nvidia-master.yaml`:

| Scenario | Template config |
|---|---|
| Public SemiAnalysis CCU sweep | `glm5-2-greennode-bench-client-remote` |
| Claude Code Weka v4 CCU sweep | `glm5-2-greennode-claude-code-weka-v4-remote-smoke` |
| Simulation smoke | `glm5-2-greennode-historical-fixed-remote-smoke` |
| Simulation full | `glm5-2-greennode-historical-fixed-remote` |

Edit only the selected config. Keep these invariants:

- `image: python:3.12-bookworm`
- `runner: benchmark-client`
- `framework: api`
- `benchmark-client: [aiperf]`
- `custom-dataset-type: weka_trace`
- Public sweep dataset: `public-dataset: semianalysis_cc_traces_weka_with_subagents_060826`
- Weka v4 dataset: `input-file: benchmarks/single_node/agentic/datasets/minimax_cc_v4_weka`
- Simulation dataset: `input-file: benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions` with `fixed-schedule: true`
- Simulation search space: `{ tp: 1, ep: 1, conc-list: [1] }`
- Sweep search space: `{ tp: 1, ep: 1, conc-list: [<confirmed ladder>] }`
- Put the confirmed duration in YAML. Leave `duration-override` empty during dispatch so committed config and executed config agree.
- If supplied, store the launch command under `remote.server-command: |`. If omitted, remove that field and repeat the metadata warning in the confirmation.
- Set `remote.url`, `model`, `tokenizer`, and `api-key-secret-name` to the confirmed provider values.

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

Inspect the generated matrix and show the user: config key, image, runner, provider URL, API model, tokenizer, dataset, fixed schedule status, CCU ladder, duration, secret name, and whether server-command metadata is present. Show the suggested Actions title. Require explicit confirmation before commit, push, or dispatch.

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

Find the run by title and branch rather than assuming the newest repository run. Report its URL and head SHA.

## Monitor and audit

Wait for every matrix job and collection job to finish. A smoke run is valid only when:

- Every expected CCU/fixed-schedule job exists and uses `benchmark-client`.
- The checked-out SHA equals the pushed commit.
- Profiling completes with usable records; report warmup errors, profiling errors, cancellations, and grace-period timeouts separately.
- Raw `agentic_*` artifacts contain AIPerf exports such as `benchmark_command.txt`, `benchmark.log`, `profile_export_aiperf.json`, CSV/timeslices, and record JSONL.
- `bmk_agentic_*` contains the normalized InferenceX result and `results_bmk/agg_bmk.json` contains the aggregate.
- When a server command was supplied, normalized results contain that exact command with `server_config_source: user-provided`.

Treat duration-boundary cancellations separately from request errors. A successful smoke validates plumbing, dataset loading, and endpoint behavior; do not present its throughput as a production performance conclusion.
