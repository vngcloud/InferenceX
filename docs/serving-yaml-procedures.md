# Serving-YAML Benchmark Procedures

<div align="center">

**English** | [中文](./serving-yaml-procedures_zh.md)

</div>

Use this page to benchmark your own serving config (a `vllm serve` / `sglang` command, or a Dynamo frontend + `dynamo.sglang` pair) on a VNG GPU runner without writing a recipe or editing `configs/`. One YAML file goes in, one GitHub Actions run comes out. The schema in [`infx/serving/schema.py`](../infx/serving/schema.py) and the agent skill [`.agents/skills/serving-bench/SKILL.md`](../.agents/skills/serving-bench/SKILL.md) are authoritative; this page explains who can run it and what the infrastructure provides.

## Procedure index

| Procedure | Use when |
| --- | --- |
| [Check prerequisites](#check-prerequisites) | Running a serving YAML for the first time |
| [Know what the runner provides](#know-what-the-runner-provides) | Deciding whether your image and model will work |
| [Write and dispatch a serving YAML](#write-and-dispatch-a-serving-yaml) | Benchmarking a config |
| [Read results and failures](#read-results-and-failures) | A run finished or failed |
| [Escalate to infra](#escalate-to-infra) | The blocker is on the node, not in your YAML |

## Check prerequisites

| Requirement | How to check |
| --- | --- |
| Write access to `vngcloud/InferenceX` | `gh repo view vngcloud/InferenceX` succeeds and you can push a branch |
| `gh` authenticated | `gh auth status` |
| `uv` installed | `uv --version` (the dispatch script validates the YAML through `uv run`) |
| Your branch is pushed and local `HEAD` equals `origin/<branch>` | `utils/serving/dispatch.sh` refuses to dispatch otherwise, because CI runs the pushed code |

You do not need a Slurm account, SSH access to the GPU nodes, or registry credentials. Every job runs on the node as the runner's service user (`stackops`), whoever dispatched it.

## Know what the runner provides

| Resource | Provided by | What it means for you |
| --- | --- | --- |
| Container image pull | The runner user's enroot credentials (`~/.config/enroot/.credentials` of `stackops`) | Public images work. `vcr.vngcloud.vn` images work only if the VCR account configured on the runner can read that repository. Any other private registry fails at `enroot import`; see [Escalate to infra](#escalate-to-infra). |
| Model weights | `serve_generic.sh` runs `hf download <model>` before starting the engine | The model does not need to be on the node. If it is already in the node's HF cache the download is a no-op; otherwise it is fetched. `model:` must be an HF repo id; local paths are rejected. |
| HF cache location | The launcher, per runner pool (see table below) | You cannot set `HF_HUB_CACHE` in `env:` (reserved). |
| Gated or private models | `HF_TOKEN` from the repository secret `INFERENCEX_OFFICIAL_RO_HF_TOKEN` | The HF account behind that token must have access to the model (accepted license). |
| Lock-pinned scenario datasets (tau2) | `GREENLAB_HF_TOKEN` in the runner's `.env` | Nothing to set; never put tokens in the YAML. |
| Dynamo discovery, ports, metrics wiring | `runners/launch_hardware-hcm.sh` | Leave out `DYN_*`, `ETCD_ENDPOINTS` and NATS settings. |

| Runner pool | `runner:` values | GPUs | Host HF cache | Router (`router:`) |
| --- | --- | --- | --- | --- |
| `cluster:h200-greennode-slurm` | `h200-greennode-slurm-1x_*` | 1 | `/data/inferencex-<uid>/hf`, or `IX_HF_CACHE_HOST` from the runner's `.env` | No |
| `cluster:hardware-hcm` | `hardware-hcm-8x_01`, `hardware-hcm-8x_02` | 8 | `/mnt/hf_hub_cache` | Yes |

The first run of a large model spends extra time downloading it and uses node disk (for example, ~400 GB for GLM-5.3 W4AFP8). Plan the first run accordingly.

## Write and dispatch a serving YAML

1. Create `serving/<name>.yaml`. Field reference and translation rules (model path → HF id, env vars, router split, context length) are in [`.agents/skills/serving-bench/SKILL.md`](../.agents/skills/serving-bench/SKILL.md). Working examples: [`serving/qwen38-bf16-tp1.yaml`](../serving/qwen38-bf16-tp1.yaml) (engine only), [`serving/glm53-w4afp8-dynamo.yaml`](../serving/glm53-w4afp8-dynamo.yaml) (Dynamo router + engine) and [`serving/glm53-w4afp8-sglang-router.yaml`](../serving/glm53-w4afp8-sglang-router.yaml) (sglang-router + SGLang engine).
2. Pick a scenario from `scenarios/` (`agentx-cc-062126`, `agentx-cc-062126-smoke`, `custom-agentic-tau2`, `custom-agentic-tau2-smoke`). Start with a `-smoke` scenario for a new config.
3. Validate and dispatch:

   ```bash
   utils/serving/dispatch.sh serving/<name>.yaml
   ```

   The script runs the same validation CI runs. On `ERROR <field>: <fix>` lines nothing was dispatched: fix the YAML and rerun. On success it prints the run URL. Use `--ref <branch>` to dispatch a branch other than the current one.
4. Do not redispatch if the script reports that the run did not show up; check `gh run list --repo vngcloud/InferenceX -w e2e-tests.yml -L 5` first.

### Replicas

Several copies of one engine (e.g. 2× TP1): set `replicas: N` and `gpus` = N × per-engine GPUs; the router command lists the engines as `{{engine_urls}}` (space-separated URLs). Not with the Dynamo frontend.

```yaml
gpus: 2
replicas: 2
router:
  image: vllm/vllm-openai:v0.25.1
  metrics_port: 29000
  serve: >-
    bash -c "python3 -m pip install -q vllm-router==0.1.14
    && exec vllm-router --host 0.0.0.0 --port {{router_port}}
    --worker-urls {{engine_urls}} --policy cache_aware
    --prometheus-host 127.0.0.1 --prometheus-port 29000
    --request-timeout-secs 14400 --disable-retries"
```

## Read results and failures

- Status: `gh run view <run-id> --repo vngcloud/InferenceX`. One job per `conc` value. Expect about 20 minutes of image pull, model load and warmup on top of the scenario duration.
- Artifacts (`gh run download <run-id> --repo vngcloud/InferenceX -n <artifact>`) start with `<name>_c<conc>`; `results/` holds `serving.yaml`, `server.log`, `benchmark.log`, `router.log` (router path) and `aiperf_artifacts/`.
- On failure, read `server.log` first, then `benchmark.log`. For the router path, read `router.log` first when the job fails before aiperf starts. `gh run view <run-id> --repo vngcloud/InferenceX --log-failed | tail -100` shows the launcher side (Slurm, enroot).
- An engine crash describes the config as given. Changing flags is the config owner's decision, not the benchmark's.

## Escalate to infra

Ask the runner owners (the cluster channel) instead of changing the YAML when:

- `enroot import` fails with 401/403 for your image: the runner user needs credentials for that registry, or the image must be pre-imported as a squash (`/mnt/containers` on `hardware-hcm`).
- The model download fails with 401/403: the HF account behind `INFERENCEX_OFFICIAL_RO_HF_TOKEN` needs access to the model.
- The download fails for lack of disk, or the job waits in the Slurm queue for a long time (`ssh slurm-client squeue -p hardware-hcm` if you have access).

## Source-of-truth links

| Source | Owns |
| --- | --- |
| [`infx/serving/schema.py`](../infx/serving/schema.py) | YAML fields, validation rules, reserved env vars, allowed runners |
| [`infx/serving/to_matrix.py`](../infx/serving/to_matrix.py) | YAML → benchmark matrix and run title |
| [`utils/serving/dispatch.sh`](../utils/serving/dispatch.sh) | Local validation, push check, dispatch |
| [`benchmarks/single_node/agentic/serve_generic.sh`](../benchmarks/single_node/agentic/serve_generic.sh) | Model pre-download, engine start, readiness |
| [`runners/launch_h200-greennode-slurm-1x.sh`](../runners/launch_h200-greennode-slurm-1x.sh), [`runners/launch_hardware-hcm.sh`](../runners/launch_hardware-hcm.sh) | Allocation, image import, mounts, router sidecar |
| [`docs/superpowers/specs/`](./superpowers/specs/) | Design decisions (serving-YAML dispatch, tau2, Dynamo router) |
