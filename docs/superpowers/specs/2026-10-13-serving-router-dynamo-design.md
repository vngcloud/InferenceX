# Serving-YAML dispatch V2: Dynamo router + engine on hgx-h200 (design)

**Date:** 2026-10-13 · **Branch:** `feat/serving-config-dispatch` · **Author:** Ly Quang Thang
**Builds on:** `2026-10-12-serving-yaml-dispatch-design.md` (V1) and its tau2 amendments. Everything V1 says still holds unless this document overrides it.

## Goal

A user pastes a two-part Dynamo config (frontend/router + engine, each with its own image, images on the private VCR registry). The agent turns it into one serving YAML and dispatches it with `utils/serving/dispatch.sh`. The run boots both processes on the 8×H200 box `hgx-h200-02` (Slurm partition `hardware-hcm`) and benchmarks through the router.

Success: the GLM-5.3 W4AFP8 example below runs green on `hardware-hcm-8x_02` with `agentx-cc-062126-smoke`. The artifacts hold `router.log`, `server.log`, `serving.yaml` and aiperf results that include `sglang:*` server metrics.

## Decisions (agreed in brainstorming)

1. **Benchmark the paste as given.** Prod (han-glm k8s) carries extra pieces that the paste does not include: an etcd sidecar, `DYN_*` env, three sglang hotfix files mounted over the image, and a frontend sampling-defaults patch. V2 ignores all of them. Matching prod is not a goal.
2. **Dynamo only.** `router:` supports `python3 -m dynamo.frontend` in front of `python3 -m dynamo.sglang`. Other routers (smg/sglang-router) are out of scope. The key does not prevent adding them later.
3. **No `--numa-node`.** Benchmark convention: the agent drops it and the validator rejects it. It is also wrong on hgx, which has 4 NUMA zones. `0 0 0 0 1 1 1 1` got the hicache pin OOM-killed (see `../docker-compose.glm53-w4afp8-8xh200.yaml`).
4. **Slurm-native, no Docker.** Every process runs in a pyxis/enroot container started with `srun` inside one `salloc`, the way `runners/launch_hardware-hcm.sh` already runs the DCGM sidecar next to the bench step.
5. **Private images use VCR account `60108-khanghd` by default.** The creds live in the runner user's enroot credentials file on the box, never in the YAML or in GitHub.
6. **The user does not wire anything.** The launcher adds whatever lets frontend and engine find each other. aiperf only talks to the router.

## Facts checked on 2026-10-13

- `hgx-h200-02`: 8×H200, 192 CPU, ~2 TB RAM, partition `hardware-hcm`, account `dev`. GitHub runners `hardware-hcm-{1,2,4,8}x_02` are online. The node is often held by manual jobs (job 2985 of `ngoctg2` held all 8 GPUs at check time), so dispatched jobs wait in the Slurm queue.
- The model `PhalaCloud/GLM-5.3-W4AFP8` (~400 GB, public on HF) is already in `/mnt/hf_hub_cache` on hgx-h200-02. `launch_hardware-hcm.sh` mounts that directory as `HF_HUB_CACHE`.
- VCR with `60108-khanghd`: `GET` manifest returns 200 for `60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1`, `108942-public69/dynamo-stack/sglang-runtime:1.5.0` and `108942-public69/dynamo-stack/etcd:v3.5.17`. (A `HEAD` request returned 404 for the first one; enroot uses `GET`.)
- Dynamo ≥ 1.4 has `--discovery-backend` / `DYN_DISCOVERY_BACKEND` with values `kubernetes | etcd | file | mem`. The default is `etcd` at `localhost:2379`. `file` stores under `DYN_FILE_KV`, and the docs list it for single-machine use with the event plane on zmq (no NATS).

## Overview

```
serving/<name>.yaml (engine + router:)
  → dispatch.sh → e2e-tests.yml get-jobs → infx.serving.to_matrix
      serving object += router-image, router-cmd, router-env
  → benchmark-tmpl.yml: env ROUTER_IMAGE / ROUTER_CMD / ROUTER_ENV (+ V1 SERVE_*)
  → runners/launch_hardware-hcm.sh
      salloc -p hardware-hcm -A dev --nodelist=hgx-h200-02 --gres=gpu:8
      srun: enroot import engine, router, dcgm  → /mnt/containers/*.sqsh (flock, cached)
      srun --overlap  dcgm.sqsh    dcgm-exporter &
      srun --overlap  router.sqsh  dynamo.frontend --http-port $PORT …  > results/router.log &
      srun            engine.sqsh  bash serve_generic.sh
                        ├─ dynamo.sglang …  > results/server.log &
                        ├─ wait: frontend /v1/models lists the served model
                        └─ aiperf → http://localhost:$PORT
```

pyxis shares the host network namespace, so all processes reach each other on `localhost`. Both containers mount the job workspace at `/workspace`, so they share the discovery directory.

## §1 Serving YAML

New optional block `router:`. The rest of the V1 schema is unchanged.

```yaml
name: glm53-w4afp8-dynamo
runner: hardware-hcm-8x_02
image: vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0   # engine image
model: PhalaCloud/GLM-5.3-W4AFP8
gpus: 8
precision: w4afp8
env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  SGLANG_DP_USE_GATHERV: "1"
  NCCL_P2P_LEVEL: NVL
  HF_HUB_OFFLINE: "1"
  SGLANG_ENABLE_METRICS_DP_ATTENTION: "1"
  SGLANG_ENABLE_METRICS_DEVICE_TIMER: "1"
serve: |
  python3 -m dynamo.sglang
  --model-path=PhalaCloud/GLM-5.3-W4AFP8
  --served-model-name=zai-org/GLM-5.3
  --quantization=w4afp8 --tp=8 --dp=8 --ep=8
  --enable-dp-attention --enable-dp-attention-local-control-broadcast --enable-dp-lm-head
  --dist-init-addr=127.0.0.1:10888
  --moe-a2a-backend=deepep --chunked-prefill-size=32768
  --tool-call-parser=glm47 --reasoning-parser=glm45
  --mem-fraction-static=0.75 --max-running-requests=256 --context-length=300000
  --kv-cache-dtype=fp8_e4m3 --dsa-prefill-backend=flashmla_sparse_q8 --allow-auto-truncate
  --enable-metrics --enable-metrics-for-all-schedulers --enable-cache-report
  --enable-hierarchical-cache --hicache-size=128 --hicache-io-backend=direct --hicache-write-policy=write_back
  --speculative-algorithm=EAGLE --speculative-num-steps=3 --speculative-eagle-topk=1 --speculative-num-draft-tokens=4
  --schedule-policy=dfs-weight --enable-prefill-delayer
  --kv-events-config='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'
router:
  image: vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1
  env: {}                       # optional, same rules as the top-level env
  serve: |
    python3 -m dynamo.frontend
    --http-host=0.0.0.0 --http-port=8888
    --router-mode=kv --router-prefill-load-scale=16 --router-min-initial-workers=1
    --dyn-chat-processor=sglang --dyn-preprocess-workers=8
    --tool-call-parser=glm47 --reasoning-parser=glm45
scenario: agentx-cc-062126-smoke
conc: [8]
```

`router:` is a nested pydantic model with `extra="forbid"` and the keys `image` (required, `IMAGE_PATTERN`), `serve` (required) and `env` (optional). The V1 rejection `router: not supported in V1` goes away.

### Validation (one line per problem, as in V1)

- `router.serve` must start with `python3 -m dynamo.frontend` (or `python -m`). Anything else: `router.serve: only dynamo.frontend is supported`.
- The engine with a router must be `python3 -m dynamo.sglang`. An engine of `dynamo.sglang` without `router:` is an error too, because a Dynamo engine has no client-facing HTTP port.
- `router:` requires a `hardware-hcm-*` runner (only that launcher starts the sidecar).
- The serving runner allowlist grows from the `cluster:h200-greennode-slurm` pool to that pool plus `hardware-hcm-8x_*`. `gpus` is checked against the matching pool's `gpus-per-node` (8 for `cluster:hardware-hcm`). Engine-only YAMLs (`sglang.launch_server`, `vllm serve`) also work on `hardware-hcm-8x_*`.
- `--numa-node` anywhere in `serve`: `serve: drop --numa-node (benchmark convention)`.
- A JSON-valued flag (`--kv-events-config`, and any value that starts with `{`) must parse as JSON after shell-style splitting. Otherwise: `serve: --kv-events-config is not valid JSON after shell quoting; wrap it in single quotes: --kv-events-config='{"publisher":…}'`. This catches the paste as written (`={"publisher":"zmq",…}` loses its double quotes in a shell).
- V1 rules apply to both `serve` and `router.serve`: no local model paths, no `$VAR`, and no reserved keys in `env` / `router.env`. `RESERVED_ENV` adds `DYN_DISCOVERY_BACKEND`, `DYN_FILE_KV`, `DYN_SYSTEM_ENABLED`, `DYN_SYSTEM_PORT`, `ROUTER_IMAGE`, `ROUTER_CMD`, `ROUTER_ENV`.

### Framework

`infer_framework` maps `python3 -m dynamo.sglang` to `sglang`. Metric prefix and naming stay those of sglang. The router's presence is carried by `router-cmd` being non-empty, not by a new framework value. The generator's matrix schema stays untouched.

## §2 Render (`infx/serving/render.py`)

- Engine (`dynamo.sglang`): split, strip nothing besides V1's `--host/--port` (the Dynamo engine has no HTTP port). Do not append `--enable-metrics` when it is already present (V1 rule unchanged).
- Router: split `router.serve`, strip `--http-host X`, `--http-host=X`, `--http-port N`, `--http-port=N`. `serve_generic.sh` / the launcher append `--http-host 0.0.0.0 --http-port $PORT`.
- The `serving` object (V1 amendment 1) gains:
  - `router-image`: `router.image`, or `''`
  - `router-cmd`: `shlex.join` of the router argv without host/port, or `''`
  - `router-env`: `KEY=VALUE` lines from `router.env` plus any `KEY=VALUE` prefix on `router.serve`
- `tp` for the matrix entry: V1 reads `--tp`/`--tp-size`/`--tensor-parallel-size`. `--tp=8` (the `=` form) must parse too. Add a test.

## §3 CI

- `e2e-tests.yml`: no change beyond what `to_matrix` emits (the `serving` output already travels as one JSON string).
- `benchmark-tmpl.yml`: add env `ROUTER_IMAGE`, `ROUTER_CMD`, `ROUTER_ENV` from `fromJSON(inputs.serving)`, with the same `inputs.serving != ''` guard as the V1 lines.

## §4 Launcher `runners/launch_hardware-hcm.sh`

Changes, all inside the existing flow:

1. **Recipe choice:** after the filename formula, `SERVE_CMD` non-empty → `BENCH_SCRIPT=benchmarks/single_node/agentic/serve_generic.sh` (same as the 1x launcher).
2. **Image import:** factor the existing `enroot import` + flock + `unsquashfs -l` check into `import_squash <image>` → prints the `.sqsh` path. Call it for `IMAGE`, `DCGM_IMAGE`, and `ROUTER_IMAGE` when set. `enroot import` reads VCR creds from the runner user's `~/.config/enroot/.credentials` (§6).
3. **Ports:** `PORT` stays `free_tcp_port`. New `DYN_SYSTEM_PORT="$(free_tcp_port)"`. Fixed ports inside the user's command (`--dist-init-addr …:10888`, kv-events `tcp://*:8900`) are kept as given. An 8-GPU job has the node to itself, so only a stray host process could collide.
4. **Discovery env**, exported before the srun steps and passed to both containers:
   `DYN_DISCOVERY_BACKEND=file`, `DYN_FILE_KV=/workspace/results/.dynkv`, `DYN_SYSTEM_ENABLED=true`, `DYN_SYSTEM_PORT`.
   `rm -rf "$GITHUB_WORKSPACE/results/.dynkv"` before the router starts, so a previous run's registrations cannot leak in. `.dynkv` is deleted after the run and not uploaded.
5. **Router sidecar**, when `ROUTER_CMD` is set, started after DCGM and before the bench step:
   ```
   srun --jobid="$JOB_ID" --overlap \
       --container-image="$ROUTER_SQUASH" \
       --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
       --no-container-mount-home --container-remap-root --container-writable \
       --container-workdir=/workspace/ --no-container-entrypoint \
       --export=ALL,PORT,DYN_DISCOVERY_BACKEND,DYN_FILE_KV,HF_HUB_CACHE \
       bash benchmarks/single_node/agentic/serve_router.sh > "$GITHUB_WORKSPACE/results/router.log" 2>&1 &
   ROUTER_SRUN_PID=$!
   ```
   `serve_router.sh` (≈15 lines) exports `ROUTER_ENV` lines, writes `results/router_command.txt`, and `exec`s `ROUTER_CMD --http-host 0.0.0.0 --http-port $PORT`. The EXIT trap kills `ROUTER_SRUN_PID` next to `DCGM_SRUN_PID`. The router step sees the allocation's GPUs but allocates none.
6. The bench step's `--export` adds `DYN_DISCOVERY_BACKEND,DYN_FILE_KV,DYN_SYSTEM_ENABLED,DYN_SYSTEM_PORT,ROUTER_CMD`.

## §5 `serve_generic.sh`

When `ROUTER_CMD` is non-empty (the Dynamo path):

- Start the engine argv as is: no `--host/--port` appended. Log to `results/server.log`, command to `results/server_command.txt`.
- Readiness: poll `http://localhost:$PORT/v1/models` until it lists `${SERVED_MODEL_NAME:-$MODEL}`. While polling, fail fast if the engine PID exits or `results/router.log` shows the frontend process ended. The timeout is the one `wait_for_server_ready` uses today.
- `AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics,http://localhost:$DYN_SYSTEM_PORT/metrics"`, `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:`.
- aiperf targets `http://localhost:$PORT` exactly as in V1.

Without `ROUTER_CMD` the script behaves exactly as V1.

## §6 One-time setup on hgx-h200-02

- `~/.config/enroot/.credentials` (mode 600) for the user that runs `hardware-hcm-8x_02`:
  `machine vcr.vngcloud.vn login 60108-khanghd password <from ~/.secrets/vcr-registry-auth.json>`.
  Never commit or echo it. Rotation means editing that line, with no runner restart (enroot reads it per import).
- Nothing else: `/mnt/hf_hub_cache` already holds the model and `/mnt/containers` is the squash cache.

## §7 Skill `serving-bench`

- Replace "a router is part of the config → say V1 benchmarks the engine only" with how to fill `router:` for a Dynamo frontend + `dynamo.sglang` pair.
- Translation rules to add:
  - drop `--numa-node …`;
  - JSON-valued flags go in single quotes;
  - `/models/<org>/<name>` → HF id, in `serve` and `model`;
  - k8s args or a pasted command with `=` forms stay as they are;
  - engine env vars go under `env:`, frontend env vars under `router.env:`;
  - 8 GPUs → runner `hardware-hcm-8x_02` (`hardware-hcm-8x_01` if the user asks for box 01);
  - VCR images need nothing extra.
- Before dispatching to `hardware-hcm-*`, run `ssh slurm-client squeue -p hardware-hcm`. If a manual job holds the node, tell the user the run will queue. Never cancel someone else's job.
- Results: add `router.log` and `router_command.txt` to the artifact list. A router failure shows there first.

## Testing

1. **Unit** (`utils/test_serving_*.py`): router block accepted and rejected (unknown key, non-frontend command, engine/router pairing both ways, non-hcm runner); `--numa-node` rejected; unquoted `--kv-events-config` rejected, single-quoted accepted; `--http-port=` / `--http-host` stripping; `--tp=8` parsed; `serving` object carries the three router fields and is empty-string for V1 YAMLs; matrix entries still pass `benchmark_schema.validate_matrix`.
2. `bash -n` on `launch_hardware-hcm.sh`, `serve_generic.sh`, `serve_router.sh`; `actionlint` on `benchmark-tmpl.yml` if available.
3. **Spike** (manual, before the launcher work, once hgx-h200-02 is free; an `salloc` with the same flags as the launcher):
   a. `enroot import` of both VCR images with the khanghd credentials file;
   b. frontend + `dynamo.sglang` with `DYN_DISCOVERY_BACKEND=file`: frontend lists the model and the KV router logs received zmq KV events;
   c. `sglang:*` metrics appear on `DYN_SYSTEM_PORT/metrics`; aiperf accepts a comma list in `AIPERF_SERVER_METRICS_URLS`.
   Fallbacks: (b) fails → etcd sidecar (`108942-public69/dynamo-stack/etcd:v3.5.17`) as a third `srun --overlap` step and `DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS=http://localhost:<free port>`. (c) fails → engine metrics come from wherever the spike finds them, or the frontend URL only, with the required prefix relaxed for the Dynamo path. Record the outcome as an amendment here.
4. **Smoke:** dispatch the §1 YAML to `hardware-hcm-8x_02`, `agentx-cc-062126-smoke`, conc `[8]`. Pass = run success; `router.log`, `server.log`, `serving.yaml`, aiperf artifacts present; aiperf server metrics include `sglang:*`.
5. **Regression:** `utils/serving/dispatch.sh --check`-equivalent (`to_matrix --check`) passes on every existing `serving/*.yaml`. One V1 smoke on `h200-greennode-slurm-1x_01` only if the shared files (`serve_generic.sh`, `render.py`) changed behavior for the engine-only path.

## Risks

- **hgx-h200-02 is shared and often held by manual jobs.** Dispatched runs pend until it is free. `salloc` has no timeout of its own, so the GitHub job waits too.
- **File discovery on 1.5.0** is documented but untested here. Covered by spike 3b and the etcd fallback.
- **Model load time:** ~400 GB W4AFP8 from local NVMe plus DeepEP/CUDA-graph warmup. The readiness timeout may need raising for this path. Measure in the spike.
- **Fixed ports in the user's command** (10888, 8900) can collide with stray processes on the host. Kept as given. Revisit only if it happens.
- **The paste differs from prod** (no hotfix files, no sampling patch). Results describe the pasted config, not the prod deployment (decision 1).

## Future direction: a shared benchmark standard behind an MCP (prepared for, not built here)

Later, a separate sub-project with its own spec, started after the V2 smoke is green: an MCP server that any team member uses to benchmark a serving config.

- **Contract:** the serving YAML. The MCP is a thin layer over what exists: `validate` (`to_matrix --check`), `dispatch`, `status`, `results`.
- **The only user credential is their own GitHub token.** `workflow_dispatch` needs `actions:write` on `vngcloud/InferenceX`, so GitHub decides who may run. The MCP runs locally (stdio) with that token and holds no shared secret.
- **The MCP pins the ref** to `vng-benchmark`/`main` and sends only the YAML. The YAML is data, not code. A free `--ref` would let anyone with write access run arbitrary branch code on GPU nodes that hold registry and HF credentials.
- **We own everything that is not the config under test:** scenarios (`scenarios/*.yaml` + `scenarios/datasets/lock.yaml`), infra env (`RESERVED_ENV`, `DYN_*`, ports, caches), registry/HF/dataset credentials. The user owns `image`, `serve`, `env` (engine env is part of what is measured), `router`, `conc`, and the scenario choice by name.
- **One Slurm service user per cluster** (greennode `test` and research `hardware-hcm`), owning every runner. Its home holds the single enroot `.credentials` and the runner `.env`, and it uses one fixed directory layout per node (e.g. `<root>/ix/{hf,containers,datasets,scratch}`). That removes the per-uid workarounds in the 1x launcher (`NODE_ROOT=…-$(id -u)`, `umask 0000`, ACLs, a runner uid with no passwd entry). The cluster admins have to create it on both clusters.
- **Model/dataset prefetch:** with a fixed layout the MCP can tell whether a model is already on the node before a job holds 8 GPUs (a 400 GB download inside the allocation wastes the node).

### Constraints V2 keeps so this stays possible

1. **Credentials live only on the runner side** (`.env`, enroot `.credentials`). The launcher and the code never name an account. `60108-khanghd` is what §6 writes into the file today, not a value in code.
2. **No new hardcoded host paths.** V2 reuses the paths `launch_hardware-hcm.sh` already has (`/mnt/hf_hub_cache`, `/mnt/containers`), and any new path is read from an `IX_*` variable with today's value as the default (V1 §4 pattern).
3. **The YAML carries no infra details:** no cache paths, ports, discovery settings or credentials. The validator rejects them through `RESERVED_ENV`.
4. **`dispatch.sh` keeps `--ref`** for development, documented as a developer-only option the MCP will not expose.

## Out of scope

The MCP server and the Slurm service user (see Future direction); smg/sglang-router; disaggregated prefill/decode; multi-node; prod hotfix/patch mounts; `hardware-hcm` tiers below 8x for the router path; ingest/dashboard import.

## Amendments (2026-10-13, spike)

Spike run on **hgx-h200-02** (Slurm job 3003, account `dev`, 2026-10-13 08:38–09:50, allocation 120 min). Logs kept at `/shared/thanglq5/dynspike/` on the submit host (`run.sh`, `engine.log`, `frontend.log`, `engine.crash1-3.log`, `import.log`).

**Box conventions (step 2, both boxes).** One convention, identical on both; `hgx-h200-01`'s local hostname is `hgx-h200-001` (the launcher maps it).

| | hgx-h200-01 | hgx-h200-02 |
|---|---|---|
| runner user | `stackops` (uid 1001) | `stackops` (uid 1001) |
| enroot credentials for runner user | unknown, `/home/stackops` is 750 (unreadable) | unknown, same |
| model in `/mnt/hf_hub_cache` | yes | yes |
| free disk `/mnt` | 13T | 12T |
| V2 squashes in `/mnt/containers` | none | none |
| writable by `thanglq5` | `/mnt/hf_hub_cache`, `/tmp` (662G), `/shared/thanglq5` (NFS) | same |

The runner user is `stackops`, not the submit user `thanglq5` (uid 1002). `/home/thanglq5` does not exist on the nodes, `/mnt/containers` and `/mnt/tmp` are stackops-owned (not writable by other users), so **§6's one-time setup cannot be performed by `thanglq5`**: the `machine vcr.vngcloud.vn …` line has to go into `/home/stackops/.config/enroot/.credentials` (mode 600) on both boxes by whoever can write that home, or the two squashes have to be pre-imported into `/mnt/containers`. Pending a human decision; the smoke cannot pull the `60108-backend-worker` frontend image without it.

**(a) enroot import — PASS.** Both images import with the `60108-khanghd` credentials read from `<HOME>/.config/enroot/.credentials` of the invoking user (the spike used a scratch HOME under `/tmp`; `enroot import` logs "Authentication succeeded"). ~7 min for both (~25.3 GB per squash).

**(b) file discovery + KV events — PASS, with one engine flag change.** `DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=<dir>` works: the frontend lists `zai-org/GLM-5.3` on its HTTP port, the KV router selects workers (`router_mode=kv`), the zmq event plane connects (5 single-consumer streams), and the frontend KV indexer logs "Using direct-ZMQ KV event ingestion". Chat completions round-trip through the router (first request 54.9 s for 16 tokens — cold, no CUDA graph; warm 16.9 s for 64 tokens).

**(SUPERSEDED by the run 6 amendment below: no `--deepep-mode` flag is needed; `NVSHMEM_REMOTE_TRANSPORT=none` fixes the NVSHMEM init crash.) The spike found the §1 engine command needed `--deepep-mode normal` on hgx.** As pasted (`--moe-a2a-backend=deepep` with no mode → sglang resolves `auto` to `low_latency`), the engine segfaults during `get_deepep_buffer`: NVSHMEM cannot init its transports on these nodes — IBGDA never (no `/dev/ibgda` on host or container), IBRC fails (`srq creation failed`, `ep_create failed` status 7). All 16 RoCE HCAs are `f0`=ACTIVE / `f1`=DOWN. `NVSHMEM_IBGDA_SUPPORT=0`, `NVSHMEM_IBRC_SUPPORT=0` (build-feature flags, not runtime switches) and `NVSHMEM_HCA_LIST=<active f0 ports>` did not change the crash. `--deepep-mode normal` boots and serves. Cost: sglang disables CUDA graphs in normal mode (its own warning), so decode latency is higher than prod's low-latency mode — accepted under decision 1 (results describe the pasted config on this hardware). `/dev/infiniband` is identical host vs container (51 entries) and memlock is ~1 GB in both, so this is not a container-passthrough problem.

**(c) engine metrics — PASS.** `DYN_SYSTEM_PORT`/metrics serves 766 `sglang:` series; the frontend's `/metrics` serves 235 `dynamo_frontend` series. `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:` is satisfiable with the two-URL metrics list. `DYN_SYSTEM_ENABLED` is deprecated in this Dynamo build (the port alone controls the metrics server) — kept as a harmless explicit setting.

**Load time:** engine start → model listed on the frontend ≈ **3 minutes** (~400 GB W4AFP8 from local NVMe, `deepep-mode normal`, CUDA graphs off).

**Offline download:** `HF_HUB_OFFLINE=1 hf download …` inside the engine container **fails — the image has no bare `hf` CLI** (`command not found`). The real path (`AIPERF_VENV/bin/hf`, created by `install_agentic_deps`) was not exercised by the spike, so the pre-download in `serve_generic.sh` runs with `HF_HUB_OFFLINE=` unset for that one command (plan Task 5 fallback, applied).

**Tools in the engine image:** `uv` MISSING, `pip` OK, `curl` OK. (`install_agentic_deps` is shared V1 code and unchanged; `curl` present is the hard requirement.)

**Spike scratch deviation:** the plan's `/mnt/tmp/dynspike-thanglq5` scratch was moved to `/tmp/dynspike-thanglq5` (`/mnt/tmp` is stackops-owned). The spike squashes could not seed `/mnt/containers` for the smoke (not writable by the spike user) — the smoke imports its own as the runner user, which is what makes the credentials item above load-bearing.

## Amendments (2026-10-13, implementation)

- **Row `tp` comes from `gpus`** (V1 behaviour), so "`--tp=8` must parse" in §2 is moot.
- **`IX_SQUASH_DIR`** overrides the squash cache dir in `runners/launch_hardware-hcm.sh` (`SQUASH_CACHE_DIR="${IX_SQUASH_DIR:-/mnt/containers}"`); default stays `/mnt/containers`. A runner's `.env` may set it (tests point it at a temp dir).
- **`.router_exited` marker:** the router sidecar subshell writes it on exit, so `wait_for_frontend_model` fails fast instead of waiting out its timeout.
- **Smoke (runner `hardware-hcm-8x_02`, `hgx-h200-02`), 5 runs.** Runs 1-4 each found and fixed a real bug:
  1. Dataset pre-download blocked by YAML env `HF_HUB_OFFLINE=1` → `HF_HUB_OFFLINE=` prefix on the dataset pre-download (`benchmark_lib.sh`).
  2. `DYN_SYSTEM_PORT` from `free_tcp_port` exceeded i16 → bounded 20000-32767 helper in `launch_hardware-hcm.sh`.
  3. aiperf telemetry CSV path pointed inside the DCGM container → `/workspace/$GPU_METRICS_CSV`.
  4. aiperf client inherited `HF_HUB_OFFLINE=1` → `serve_generic.sh` unsets it after the server is ready (engine keeps it).
  
  Runs: https://github.com/vngcloud/InferenceX/actions/runs/38385771112 , .../38387761468 , .../38389108473 , .../38390486040 , .../38391898903 (run 5).
- **Run 5 outcome:** every V2 plumbing step worked, but the sglang scheduler crashed: `AttributeError: 'DeepseekSparseAttnBackend' object has no attribute 'forward_metadata'` (`dsa_indexer.get_indexer_metadata`) under `--deepep-mode normal` (which disables CUDA graphs). 0 successful requests, FAILED at the metrics gate. `--deepep-mode normal` was only a workaround for the NVSHMEM crash below, and it triggered this one. Superseded by run 6.
- **Root cause of the hgx failures:** the hgx container sees 51 InfiniBand devices, so NVSHMEM tries IBGDA/IBRC and segfaults in `get_deepep_buffer` (`init failed for transport: IBGDA` is still logged but is non-fatal). The han prod pod has no `/dev/infiniband`, so it never tries.
- **Fix (run 6, PASS):** remove `--deepep-mode normal` and add env `NVSHMEM_REMOTE_TRANSPORT: none` (deepep auto resolves to low_latency, CUDA graph on, speculative decoding on). Run https://github.com/vngcloud/InferenceX/actions/runs/38432875488 (`hardware-hcm-8x_02`) completed successfully: 143 requests total / 53 successful, TTFT p50 1.73 s, ITL p50 11.6 ms, 149 `sglang:*` series in `server_metrics_export.json`. The open item (choose a config that runs on hgx) is closed; no repo plumbing change was needed. **Not isolated:** run 6 changed two things at once (dropped `--deepep-mode normal`, added `NVSHMEM_REMOTE_TRANSPORT=none`). The env var is what stops the NVSHMEM IB-init segfault; whether the `forward_metadata` crash came from `normal` itself or from CUDA graphs being off is untested (`normal` + the env var alone was not run).
- **Prod comparison (han cluster, `glm53-w4afp8-dynamo-engine`):** same flags, etcd discovery, CUDA graph on, no IB devices in the pod. It also mounts a `glm53-sglang-hotfix` ConfigMap over `dsa_backend.py` / `forward_mha.py` / `index_topk_share.py`; that hotfix is not needed on our run.
- **V1 note:** `launch_h200-greennode-slurm.sh` is not affected: it has no serving path, and no other recipe ships a `gpu_metrics.csv`.
