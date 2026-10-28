---
name: serving-bench
description: Benchmark an LLM serving config with the AgentX agentic-coding scenario or the tau2 customer-service scenario (custom-agentic-tau2) on a VNG GPU runner, starting from a vllm/sglang/trtllm-serve launch command, a Dynamo frontend + engine pair, a docker-compose file or a k8s manifest. Use when the user says "benchmark agentx với config…", "chạy agentx cho config này", "bench agentx", "bench tau2", "chạy tau2 cho config…", or pastes a serve command and asks for a benchmark. Writes one serving YAML and dispatches it with one command.
---

# serving-bench

You turn the user's serving config into **one YAML file** and run **one command**. Do not write recipes, edit configs, or call `gh workflow run` yourself.

## 1. Write `serving/<name>.yaml`

```yaml
name: qwen38-bf16-tp1               # [a-z0-9.-], no "_"; becomes the artifact name prefix
runner: h200-greennode-slurm-1x_01  # see "Runner" below
image: vllm/vllm-openai:v0.29.0     # required
model: Qwen/Qwen3.8-27B             # HF repo id, never a local path
gpus: 1                             # GPUs the command needs
replicas: 1                         # optional: N engine copies, gpus/N GPUs each; N > 1 needs router: with {{engine_urls}}
precision: bf16                     # optional (bf16 default), naming only
env:                                # optional, the engine's env vars; quote booleans
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
serve: |                            # the launch command, as given
  vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --max-model-len 262144
  --default-chat-template-kwargs '{"enable_thinking": true}'
engine-args:                         # optional: trtllm-serve only; written to engine_args.yaml and
  attn_backend: FLASHINFER           # passed as --config (no such flag in serve:)
router:                             # optional: any router in front of the engine (hardware-hcm runner)
  image: <router image>
  env: {}                           # the router's env vars
  metrics_port: 18888               # optional: the router's /metrics port; default = the router port
  serve: |                          # {{router_port}} = the port the benchmark calls, {{engine_url}} = the engine
    python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}}
    --worker-urls={{engine_url}} --policy=cache_aware
scenario: agentx-cc-062126-smoke    # a file name in scenarios/ (without .yaml); or a list — one job per scenario in one run (needs exactly one conc)
conc: [4, 8]                        # one benchmark job per value
```

Full working examples: `serving/qwen38-bf16-tp1.yaml` (engine only), `serving/glm53-w4afp8-dynamo.yaml` (Dynamo router + engine), `serving/glm53-w4afp8-sglang-router.yaml` (sglang-router + SGLang engine) and `serving/gemma4-31b-fp8-trtllm-tp2-livekit-v1.yaml` (trtllm-serve, node-local .sqsh image, engine-args). Format references only: never dispatch them or copy their values for a user's request.

Translation rules:
- **Model path → HF id.** `--model-path /models/<org>/<model>` becomes `--model-path <org>/<model>` and `model: <org>/<model>`. If you cannot tell the HF id from the path, ask.
- **Host/port:** leave them or drop them; they are replaced by the benchmark's own.
- **trtllm-serve** (`trtllm-serve serve`): the 1.4.x CLI takes the MODEL as a positional argument and spells its flags snake_case (`--tensor_parallel_size`, `--served_model_name`, `--tool_parser`, `--reasoning_parser`) — translate pasted kebab-case flags (`--tensor-parallel-size` → `--tensor_parallel_size`, `--dyn-tool-call-parser` → `--tool_parser`, `--dyn-reasoning-parser` → `--reasoning_parser`); `--kv-block-size N` → `kv_cache_config.tokens_per_block: N` and `--publish-kv-events` → `kv_cache_config.kv_events_config.enable_kv_cache_events: true` go under `engine-args:`. A pasted `--extra-engine-args <file.yaml>` / `--config <file.yaml>` does not survive (the file is not on the runner): move the file's keys into an `engine-args:` block; the benchmark writes them to `engine_args.yaml` next to the run and appends `--config engine_args.yaml` itself.
- **Router:** a router part goes under `router:` with its own `image`; the engine stays in `serve`. Engine env → `env:`, router env → `router.env:`.
  - **Replicas** (several copies of one engine, e.g. 2× TP1): set `replicas: N` and `gpus` = N × per-engine GPUs; the router command lists the engines as `{{engine_urls}}` (space-separated URLs). Not with the Dynamo frontend.
  - **Dynamo** (`python3 -m dynamo.frontend` + `python3 -m dynamo.sglang`): as before. Leave out discovery settings (`DYN_*`, `ETCD_ENDPOINTS`, NATS) and `--http-host/--http-port`: the benchmark wires them.
  - **Any other router** (sglang-router and others): keep the user's flags verbatim except the wiring. Listen host/port flags → `--host=0.0.0.0 --port={{router_port}}`. A worker/backend URL (k8s service, hostname) → `{{engine_url}}` (or `{{engine_port}}` where only a port fits). A literal metrics/prometheus port flag → also `router.metrics_port: <port>`. The engine's own `--host/--port` are dropped, as for any engine.
  - **Router command missing** (only flags + image): infer it from the flag family and say so in the reply so the user can correct it. `--worker-urls`, `--policy=cache_aware`, `--dp-aware`, `--prometheus-port` → `python3 -m sglang_router.launch_router`. Unknown family → ask. The command must match the image: slim/compiled images (no python3) run a binary path (e.g. `/work/target/release/smg launch`), not `python3 -m …`.
  - Placeholders exist only in `router.serve` / `router.env`. No `$VAR` there either.
- **Env vars** (lines like `NCCL_P2P_LEVEL=NVL`, compose `environment:`, k8s `env:`) go under `env:`.
- No secrets in `env:` (the YAML is shown in the run inputs and saved as an artifact). Dataset access uses the runner's own `GREENLAB_HF_TOKEN`; never set it in the YAML.
- Images: public ones and `vcr.vngcloud.vn` ones the runner's VCR account can read need nothing extra (the runner holds the pull credentials). Another private registry fails at `enroot import`; tell the user it needs infra (credentials on the runner user or a pre-imported squash), see `docs/serving-yaml-procedures.md`. Never put credentials in the YAML. A `.sqsh` file already on the node runs as is: use its absolute path as `image:` and pin the runner to the node that holds it (`hardware-hcm-8x_*` or `hcm-rtx6000pro_*` only).
- **gpus:** the number of GPUs the command needs, e.g. tensor-parallel size × pipeline-parallel size. If the flags do not make it clear, ask.
- **Runner:** `gpus: 1` → `h200-greennode-slurm-1x_*`; `gpus: 8` → `hardware-hcm-8x_01` or `_02` (the two 8×H200 boxes, same env convention). Run `ssh slurm-client squeue -p hardware-hcm`, pick the free box; if both are held, pick one and tell the user the run waits in the Slurm queue. Never cancel another user's job. A router (`router:`) needs a `hardware-hcm-*` runner.
- **docker-compose / k8s:** `image` from `image:`; `serve` from `command:` + `args:` joined; `env` from `environment:` / `env:`.
- No shell variables (`$X`) in `serve`: write literal values.
- **Drop `--numa-node …`** (with all its values). Benchmark convention.
- **DeepEP on `hardware-hcm-*`** (`--moe-a2a-backend deepep`): add `NVSHMEM_REMOTE_TRANSPORT: none` to `env:`. The container sees InfiniBand devices and single-node NVSHMEM would segfault on IBGDA/IBRC.
- **JSON in a flag must be single-quoted:** `--kv-events-config='{"publisher":"zmq",…}'`. A pasted `--x={"a":"b"}` loses its quotes.
- **Context length:** agentx scenarios need 262144 context (`maxContextLength: 262144`; `--max-model-len 262144` / `--context-length 262144`), ask if the config cannot; tau2 scenarios set none (turns p95 ~25k tokens), the model default is fine unless below 32k.

## 2. Ask the user (do not guess) when

Asking means: stop, write nothing, dispatch nothing, and end your reply with the question. This holds even when you run non-interactively — then your reply is the question, and the user answers in the next turn. Never replace a missing answer with an assumption.

- the image is not given and no version either → ask for the exact image tag. Only the user's message counts: never take the image from `serving/*.yaml`, the example above or past runs, and never reuse an existing `serving/` file for a new request — write a new one;
- a **version** is given but no image (e.g. "SGLang 0.5.18") → resolve a public image yourself, do not ask: SGLang → `lmsysorg/sglang:v<ver>`, vLLM → `vllm/vllm-openai:v<ver>`. Verify the tag exists before writing the YAML: `curl -s "https://hub.docker.com/v2/repositories/<repo>/tags?name=<tag>&page_size=50"` and check the exact tag is in `results[].name`. Variants (`-cu129`, `-cu130`, `-runtime`) exist → take the bare tag and name it in the reply. Tag not found → stop and ask. A router image is never resolved this way (it is the user's own build): use it as given; a bare name with no registry prefix → ask for the full reference;
- the scenario is not given → run `ls scenarios/*.yaml`, list the names, ask which: `agentx-cc-062126` (Weka coding traces, 3600 s), `agentx-cc-062126-smoke` (180 s), `custom-agentic-tau2` (tau2 customer-service agent traces, 1200 s), `custom-agentic-tau2-smoke` (180 s). `scenarios/datasets/` is not a scenario;
- `gpus` is neither 1 nor 8 → ask which runner.
- a Dynamo router paired with a non-Dynamo engine, or the reverse → the validator rejects it; tell the user the Dynamo frontend only works with `dynamo.sglang`, and ask.

## 3. Dispatch

```bash
utils/serving/dispatch.sh serving/<name>.yaml
```

It validates first. On `ERROR <field>: <fix>` lines, fix the YAML and rerun; nothing was dispatched. On success it prints the run URL. The current git branch must be pushed (the script says so if not); do not push without the user's OK.

## 4. Results

- Status: `gh run view <run-id> --repo vngcloud/InferenceX`. One job per `conc`. Expect roughly 20 min of image pull, model load and warmup on top of the scenario duration (180 s smoke ≈ 25 min).
- Artifacts (`gh run download <run-id> --repo vngcloud/InferenceX -n <artifact>`), names start with `<name>_c<conc>`:
  - `agentic_<name>_c<conc>_…`: `results/` with `serving.yaml`, `router_command.txt`, `router.log`, `server_command.txt`, `server.log`, `benchmark_command.txt`, `benchmark.log`, `aiperf_artifacts/`
  - `bmk_agentic_<name>_c<conc>_…`: aggregate result JSON
  - `server_logs_<name>_c<conc>_…`: server logs
- Dataset errors appear as `ERROR dataset <name>: …` lines in the job log (`gh run view <id> --repo vngcloud/InferenceX --log-failed`); the first tau2 run downloads ~1.1 GB.
- A failed job: read `server.log` first (engine flags, OOM, parser errors), then `benchmark.log` (aiperf). Router path: read `router.log` first when the job fails before aiperf starts. `gh run view <run-id> --repo vngcloud/InferenceX --log-failed | tail -100` shows the launcher side (slurm, enroot).
- If the engine crashes or the job fails, report the error and the config as given. Do not change flags or env, or redispatch a modified config, on your own: the config is the user's, a change is their decision.
- Report to the user: run URL, per-conc success/failure, and where the artifacts are.
