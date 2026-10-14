# Serving-YAML V2 (Dynamo router + engine on hgx-h200) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A serving YAML with a `router:` block (Dynamo frontend + `dynamo.sglang` engine, two private VCR images) validates, dispatches, and runs on either 8×H200 box of Slurm partition `hardware-hcm`: `hardware-hcm-8x_01` (hgx-h200-01) or `hardware-hcm-8x_02` (hgx-h200-02). Both boxes follow one env convention, so a YAML runs the same on either.

**Architecture:** The V1 pipeline stays as it is (`serving/<name>.yaml` → `utils/serving/dispatch.sh` → `e2e-tests.yml` → `infx.serving.to_matrix` → `benchmark-tmpl.yml` → launcher → `serve_generic.sh`). V2 adds three router fields to the rendered `serving` object and three `ROUTER_*` env vars in `benchmark-tmpl.yml`. It teaches `runners/launch_hardware-hcm.sh` to run the serving path and a router sidecar (`srun --overlap` pyxis container, the same pattern as the DCGM sidecar). It adds a Dynamo branch to `serve_generic.sh`. Frontend and engine find each other through Dynamo's file discovery backend in the shared `/workspace/results/.dynkv`.

**Tech Stack:** Python 3.12 + pydantic + PyYAML (`infx/serving`), bash, Slurm + pyxis/enroot, GitHub Actions, NVIDIA Dynamo 1.5.0 (sglang runtime), aiperf.

**Execution (chosen by the user 2026-10-13):** superpowers:subagent-driven-development, one fresh subagent per task plus a reviewer before the next task. Task 1 and Task 8 touch the cluster, so the controller (or the user) runs them, not a coding subagent.

**Spec:** `docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md` (V2). It builds on `docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md` (V1, read its "Amendments" section). Read both before Task 1.

---

## Read first: context in 5 minutes

**Where things are (branch `feat/serving-config-dispatch`):**

| Piece | File | What it does today |
|---|---|---|
| Schema | `infx/serving/schema.py` | pydantic `ServingSpec`, `load_serving(text, repo)`. Every error is one line `<field>: <fix>`. Rejects `router:` ("not supported in V1"). Runner allowlist = pool `cluster:h200-greennode-slurm` only. |
| Render | `infx/serving/render.py` | `render(spec) -> dict` with keys `framework`, `serve-cmd` (shlex-joined argv without `--host/--port`), `serve-env`, `served-model-name`. `infer_framework(argv)` knows only vllm/sglang. |
| Matrix | `infx/serving/to_matrix.py` | `build(text) -> {"matrix": [...], "serving": {...}, "title": str}`. Matrix rows must keep exactly the generator keys (schema `extra="forbid"`). Row `tp` = `spec.gpus`, so nothing parses `--tp`. |
| Workflow | `.github/workflows/benchmark-tmpl.yml` ~line 221 | Maps `fromJSON(inputs.serving).<key>` to env `SERVE_CMD`, `SERVE_ENV`, `SERVED_MODEL_NAME`, `SERVING_SCENARIO`, `SERVING_LOADER`, `SERVING_YAML`. Line ~422 routes `hardware-hcm-*` runners to `runners/launch_hardware-hcm.sh`. `HF_HUB_CACHE=/mnt/hf_hub_cache/`, `RESULT_DIR=/workspace/results`. |
| 1x launcher | `runners/launch_h200-greennode-slurm-1x.sh` | V1 launcher. `SERVE_CMD` set → `BENCH_SCRIPT=benchmarks/single_node/agentic/serve_generic.sh`. Do not change it in V2. |
| hgx launcher | `runners/launch_hardware-hcm.sh` | pyxis path: `salloc` → `srun enroot import` into `/mnt/containers/*.sqsh` → DCGM sidecar `srun --overlap … &` → bench `srun --container-image=… bash $BENCH_SCRIPT`. It has **no** serving path yet. |
| Recipe | `benchmarks/single_node/agentic/serve_generic.sh` | Exports `SERVE_ENV`, fetches dataset + model, starts `SERVE_CMD --host 0.0.0.0 --port $PORT`, `wait_for_server_ready`, `build_replay_cmd_yaml`, `run_agentic_replay_and_write_outputs`. |
| Bash helpers | `benchmarks/benchmark_lib.sh` | `wait_for_server_ready` (~line 625, polls `/health`, then `server_watch capture`), `write_command`, `build_replay_cmd_yaml` (~3474, turns comma-separated `AIPERF_SERVER_METRICS_URLS` into several `--server-metrics` values), `validate_required_agentic_server_metrics` (~3640, greps `"<prefix>` in the export JSON). |
| Skill | `.agents/skills/serving-bench/SKILL.md` (`.claude/skills/serving-bench` is a symlink) | What the user-facing agent follows. It still says "router → V1 engine only". |
| Tests | `utils/test_serving_*.py` | Run from `utils/` (pytest.ini: `pythonpath = ..`). Launcher tests stub `salloc/srun/squeue/scancel/hostname` as PATH scripts that log argv (see `utils/test_serving_launcher.py`). |

**Run the tests** (the system python has no pytest. Use uv, baseline = 80 passed):
```bash
cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_*.py
```

**Facts already verified (2026-10-13), do not re-check them unless something fails:**
- Images, all pullable with VCR account `60108-khanghd` (GET manifest 200):
  - engine `vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0`
  - frontend `vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1`
  - etcd fallback `vcr.vngcloud.vn/108942-public69/dynamo-stack/etcd:v3.5.17`

  The credentials are in `~/.secrets/vcr-registry-auth.json` on the workstation. Read that file with the **Read tool**: `cat *secret*` is denied by policy. Never print the password.
- Model `PhalaCloud/GLM-5.3-W4AFP8` (~400 GB) is already in `/mnt/hf_hub_cache` on hgx-h200-02. hgx-h200-01 is **not checked yet**: someone was running GLM-5.3 there, so it may be on that box under another path (Task 1 checks).
- Slurm access: `ssh slurm-client` (submit host). hgx nodes refuse direct ssh, so reach them only through `srun --jobid=<id> --overlap …`. Partition `hardware-hcm`, account `dev`, Slurm nodes `hgx-h200-01` / `hgx-h200-02` (runners `hardware-hcm-8x_01` / `_02`; the local hostname of box 01 is `hgx-h200-001`, which the launcher maps).
- **Use either box. The user does not pin node 02.** Pick whichever is free at the time. Both boxes are often held by someone's manual job (`ssh slurm-client squeue -p hardware-hcm`). **Never `scancel` another user's job.** Wait, or ask the user.
- Dynamo ≥1.4: `DYN_DISCOVERY_BACKEND` ∈ `kubernetes|etcd|file|mem`, default `etcd` at `localhost:2379`. `file` keeps state under `DYN_FILE_KV`. The event plane defaults to zmq for file discovery.
- The user's paste has `--kv-events-config={"publisher":"zmq",…}` **unquoted**. A shell strips the double quotes, so it must be single-quoted in the YAML.

**Rules from the user (global CLAUDE.md):**
- Commit as `git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit …`.
- `git add` exact files only, never `-A` or `.`, never `.zcodeignore` or `.serena/`.
- **Ask the user before any `git push`** and before any `dispatch.sh` (it needs the branch pushed).
- `export X=…` as a bash command is denied by policy. Assign inline (`X=… cmd`) when you run things yourself. Inside scripts `export` is fine.

## Global Constraints

- Every validation error is one line `<field>: <problem and fix>` (V1 style). `to_matrix` prints them as `ERROR <line>` and exits 2.
- Matrix rows keep exactly the V1 key set. Router data travels only in the `serving` object (`router-image`, `router-cmd`, `router-env`). For engine-only YAMLs all three are `""`.
- Only `dynamo.frontend` (router) + `dynamo.sglang` (engine) are supported as a router pair. Either one without the other is an error.
- `router:` requires a `hardware-hcm-*` runner. Serving runners = `h200-greennode-slurm-1x_*` (pool `cluster:h200-greennode-slurm`) + `hardware-hcm-8x_*` (pool `cluster:hardware-hcm`).
- `--numa-node` anywhere in `serve`/`router.serve` → `serve: drop --numa-node (benchmark convention)` (field name `router.serve` when it is in the router).
- Credentials only on the runner side. No account name in code. No new hardcoded host path: a new path knob is an `IX_*` variable whose default is today's value.
- Launcher-owned env (`RESERVED_ENV` additions): `DYN_DISCOVERY_BACKEND`, `DYN_FILE_KV`, `DYN_SYSTEM_ENABLED`, `DYN_SYSTEM_PORT`, `ROUTER_IMAGE`, `ROUTER_CMD`, `ROUTER_ENV`.
- Discovery values: `DYN_DISCOVERY_BACKEND=file`, `DYN_FILE_KV=/workspace/results/.dynkv`, `DYN_SYSTEM_ENABLED=true`, `DYN_SYSTEM_PORT=$(free_tcp_port)`.
- **One env convention on both hgx boxes:**
  - same runner-side setup: enroot `~/.config/enroot/.credentials` for each box's runner user, model under `/mnt/hf_hub_cache` (HF layout), squash cache `/mnt/containers`, `/mnt/tmp`;
  - no per-box branches in the launcher besides the existing hostname → Slurm-node map;
  - a gap found on one box is fixed on that box (setup), not worked around in code.
- V1 behaviour (engine-only, 1x runner) must not change. All 80 existing tests stay green, with the edits this plan names.

## Review Focus

1. **The paste as typed:** an unquoted `--kv-events-config={"publisher":…}` gives a one-line error that says to single-quote it, not a crash or a silent run. → test in Task 2.
2. **`HF_HUB_OFFLINE=1` in the user's `env:`.** `serve_generic.sh` exports `SERVE_ENV` *before* `"$AIPERF_HF_CLI" download "$MODEL"`. With a complete cache this is a no-op, but with a partial cache it errors. The spike (Task 1 step 6) records whether it passes on hgx-02. If it fails, Task 5 must run the download with `HF_HUB_OFFLINE` unset for that one command. → Task 1 / Task 5.
3. **A router or engine that dies during load** must fail the job within seconds, not hang until the 180-min salloc ends. → `wait_for_frontend_model` tests in Task 5 (engine dead, router-exited marker).
4. **Stale discovery state:** a `.dynkv` left by an earlier run must not register a ghost worker. → Task 6 asserts the launcher removes it before the router starts.
5. **Engine-only YAMLs on `hardware-hcm-8x_*`** (plain `sglang.launch_server`) must work too: serve_generic path, no router sidecar, no DYN env. → Task 6 parametrized launcher test.

---

## File map

| File | Change | Task |
|---|---|---|
| `docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md` | Amendments section (spike outcome, tp note) | 1, 8 |
| `infx/serving/schema.py` | `RouterSpec`, runner pools, numa/`$`/local-path/env checks for router | 2 |
| `infx/serving/render.py` | dynamo framework, router pairing, JSON-flag check, router render | 2 |
| `infx/serving/to_matrix.py` | `serving` object + router fields | 3 |
| `.github/workflows/benchmark-tmpl.yml` | `ROUTER_IMAGE/CMD/ENV` env | 3 |
| `benchmarks/benchmark_lib.sh` | `wait_for_frontend_model` | 4 |
| `benchmarks/single_node/agentic/serve_router.sh` | new: frontend entrypoint | 4 |
| `benchmarks/single_node/agentic/serve_generic.sh` | Dynamo branch | 5 |
| `runners/launch_hardware-hcm.sh` | serving path, `import_squash`, router sidecar, DYN env | 6 |
| `serving/glm53-w4afp8-dynamo.yaml` | new example | 7 |
| `.agents/skills/serving-bench/SKILL.md` | router rules | 7 |
| tests `utils/test_serving_{schema,render,to_matrix,workflows,replay}.py`, new `utils/test_serving_launcher_hcm.py`, new `utils/test_serving_router.py` | | 2–7 |

Task order: **Task 1 (spike) needs one hgx box free** (either one). Its setup steps 2–3 run on **both** boxes, because a 0-GPU allocation fits even next to a busy job. If it is busy, do Tasks 2–5 first (they do not depend on the spike), then come back. Tasks 6–8 depend on the spike outcome.

---

### Task 1: Spike on a free hgx box + make both boxes follow one convention (manual, throwaway, no repo code)

Answers three questions before the launcher is written: (a) pyxis/enroot imports both VCR images with the khanghd credentials, (b) frontend + `dynamo.sglang` find each other with `DYN_DISCOVERY_BACKEND=file`, and the KV router gets zmq events, (c) `sglang:*` metrics show up on `DYN_SYSTEM_PORT/metrics`. aiperf already accepts several `--server-metrics` URLs (`build_replay_cmd_yaml` joins them), so that part is not a question.

**Files:** none in the repo, except the spec amendment in step 9.

- [ ] **Step 1: Check the node is free.**
  Run: `ssh slurm-client 'squeue -p hardware-hcm -o "%i %u %j %T %b %M %N"'`
  Pick the free box: `NODE=hgx-h200-01` or `NODE=hgx-h200-02` (no job holding its 8 GPUs). Steps 4–8 use `$NODE`. If both are held, still do steps 2–3 (0-GPU allocations), tell the user, and continue with Task 2.
  The commands below are written for box 02. For box 01 substitute `--nodelist=hgx-h200-01`.

- [ ] **Step 2: Convention check on BOTH boxes.** Run this once per box (`--nodelist=hgx-h200-01`, then `hgx-h200-02`). It shows the runner user (whose `~/.config/enroot/.credentials` enroot reads), whether the model is in `/mnt/hf_hub_cache`, and what is in the squash cache.
  ```bash
  ssh slurm-client 'salloc -p hardware-hcm -A dev --nodelist=hgx-h200-02 --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=20 --no-shell --job-name=spike-probe-thanglq5 && J=$(squeue -u $USER -n spike-probe-thanglq5 -h -o %A) && srun --jobid=$J bash -c "ps -eo user,cmd | grep -E \"Runner.Listener|run.sh\" | grep -v grep; id; echo HOME=\$HOME; ls -la ~/.config/enroot 2>&1; ls -d /mnt/hf_hub_cache/models--PhalaCloud--GLM-5.3-W4AFP8 2>&1; df -h /mnt/hf_hub_cache /mnt/containers | tail -2; ls /mnt/containers | grep -iE \"dynamo|dcgm\""; scancel $J'
  ```
  Write a small table of both boxes: runner user, credentials present, model present, free disk. Any gap is fixed **as setup on that box**: credentials via step 3; a missing model by `hf download PhalaCloud/GLM-5.3-W4AFP8` into `/mnt/hf_hub_cache` in a 0-GPU allocation, **after asking the user** (400 GB). Do not add code that branches per box.
  Record the runner user(s) for `hardware-hcm-8x_01` and `hardware-hcm-8x_02` and whether it is the same user as the `srun` user (`id`). If the runner user is not `thanglq5` and you cannot write its home, **stop and ask the user** who can install the credentials file (spec §6).

- [ ] **Step 3: Install the enroot credentials for the runner user on each box that lacks them** (one-time setup, spec §6; run once per box with its `--nodelist`).
  Read `~/.secrets/vcr-registry-auth.json` with the Read tool. `auths["vcr.vngcloud.vn"].auth` is base64 of `60108-khanghd:<password>`. Write the line through a pipe so the password never shows on a command line or in output:
  ```bash
  python3 -c 'import json,base64,os;a=json.load(open(os.path.expanduser("~/.secrets/vcr-registry-auth.json")))["auths"]["vcr.vngcloud.vn"]["auth"];u,p=base64.b64decode(a).decode().split(":",1);print(f"machine vcr.vngcloud.vn login {u} password {p}")' \
    | ssh slurm-client 'salloc -p hardware-hcm -A dev --nodelist=hgx-h200-02 --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=10 --no-shell --job-name=spike-cred-thanglq5 >/dev/null && J=$(squeue -u $USER -n spike-cred-thanglq5 -h -o %A) && srun --jobid=$J bash -c "umask 077; mkdir -p ~/.config/enroot; cat > ~/.config/enroot/.credentials; wc -l ~/.config/enroot/.credentials"; scancel $J'
  ```
  Expected: `1 …/.credentials`. This only works if the srun user is the runner user (step 2). Otherwise the user decides how to do it.

- [ ] **Step 4: Allocate the full node for the spike** (keep the allocation until step 8):
  ```bash
  ssh slurm-client 'salloc -p hardware-hcm -A dev --nodelist=hgx-h200-02 --gres=gpu:8 --cpus-per-task=192 --mem=1900G --time=120 --no-shell --job-name=spike-dynamo-thanglq5; squeue -u $USER -n spike-dynamo-thanglq5 -h -o %A'
  ```
  Record the job id as `J`.

- [ ] **Step 5: Import both images into the shared squash cache**, with the same file names the launcher uses, so the later smoke reuses them:
  ```bash
  ssh slurm-client 'J=<id>; for IMG in vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0 vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1; do SQ=/mnt/containers/$(echo $IMG | sed "s/[\/:@#]/_/g").sqsh; srun --jobid=$J --overlap bash -c "export ENROOT_CACHE_PATH=\$HOME/.cache/enroot TMPDIR=/mnt/tmp; mkdir -p \$ENROOT_CACHE_PATH \$TMPDIR; unsquashfs -l $SQ >/dev/null 2>&1 || enroot import -o $SQ docker://vcr.vngcloud.vn#${IMG#vcr.vngcloud.vn/}"; done'
  ```
  Expected: two `.sqsh` files, no `401`. → answers (a).

- [ ] **Step 6: Start frontend + engine with file discovery.** Use a scratch dir `/mnt/tmp/dynspike-thanglq5` as the shared dir, ports 18888 (frontend) and 18889 (system). Write a spike script on `/shared/thanglq5/dynspike/run.sh` (NFS, visible from the node). It runs two `srun --overlap --container-image=…` steps in the background, like the launcher will:
  - frontend: `python3 -m dynamo.frontend --http-host=0.0.0.0 --http-port=18888 --router-mode=kv --router-prefill-load-scale=16 --router-min-initial-workers=1 --dyn-chat-processor=sglang --dyn-preprocess-workers=8 --tool-call-parser=glm47 --reasoning-parser=glm45`
  - engine: the §1 `serve` of the spec, verbatim (no `--numa-node`, HF id model path, single-quoted `--kv-events-config`), env from §1 plus `HF_HUB_CACHE=/mnt/hf_hub_cache/`
  - both with env `DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/dynkv DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=18889`, mounts `/mnt/tmp/dynspike-thanglq5:/dynkv,/mnt/hf_hub_cache/:/mnt/hf_hub_cache/`, flags `--no-container-mount-home --container-remap-root --container-writable --no-container-entrypoint`, logs to `/shared/thanglq5/dynspike/{frontend,engine}.log`

  Also run, inside the engine container before the engine starts: `HF_HUB_OFFLINE=1 hf download PhalaCloud/GLM-5.3-W4AFP8 >/dev/null && echo OFFLINE_DOWNLOAD_OK` (Review Focus 2), and `which uv pip curl`. `install_agentic_deps` needs them.

  Record the time from engine start to the model showing in `/v1/models`.

- [ ] **Step 7: Probe.**
  ```bash
  ssh slurm-client 'J=<id>; srun --jobid=$J --overlap bash -c "
    curl -s localhost:18888/v1/models; echo;
    curl -s localhost:18888/v1/chat/completions -H content-type:application/json -d \"{\\\"model\\\":\\\"zai-org/GLM-5.3\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"hi\\\"}],\\\"max_tokens\\\":16}\" | head -c 400; echo;
    curl -s localhost:18889/metrics | grep -c ^sglang: ;
    curl -s localhost:18888/metrics | grep -c ^dynamo_frontend"'
  grep -iE "kv.?event|indexer|subscrib" /shared/thanglq5/dynspike/frontend.log | tail -5   # run on slurm-client
  ```
  Expected: the model id is listed, a chat completion comes back, `sglang:` count > 0 on 18889, `dynamo_frontend` count > 0, and the frontend log shows KV events or indexer activity after the request. → answers (b) and (c).

- [ ] **Step 8: Tear down.** `ssh slurm-client 'scancel <id>; rm -rf /mnt/tmp/dynspike-thanglq5'` (the rm through `srun` before the scancel if `/mnt/tmp` is node-local).

- [ ] **Step 9: Record the outcome** as an "Amendments (2026-10-13, spike)" section at the end of the spec: which box ran the spike, the step 2 table for both boxes, (a)/(b)/(c) pass or fail, the load time, `OFFLINE_DOWNLOAD_OK` or not, `uv`/`pip`/`curl` presence, the runner user. Decide using the spec's fallbacks:
  - (b) fails → Task 6 adds an etcd sidecar (`108942-public69/dynamo-stack/etcd:v3.5.17`, `etcd --listen-client-urls http://0.0.0.0:$ETCD_PORT --advertise-client-urls http://localhost:$ETCD_PORT --data-dir /workspace/results/.etcd`) and sets `DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS=http://localhost:$ETCD_PORT` instead of the file vars. Every other task stays the same.
  - (c) fails → write in the amendment where the engine's `sglang:*` metrics are. Task 5 uses that URL, or drops `AIPERF_REQUIRED_SERVER_METRIC_PREFIX` on the Dynamo path if there is none.
  - offline download fails → Task 5 step 3 runs the download as `HF_HUB_OFFLINE= "$AIPERF_HF_CLI" download "$MODEL"`.

  ```bash
  git add docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "docs(spec): V2 spike outcome and hgx box conventions"
  ```

---

### Task 2: Schema + render accept a Dynamo router pair

**Files:**
- Modify: `infx/serving/schema.py`
- Modify: `infx/serving/render.py`
- Test: `utils/test_serving_schema.py`, `utils/test_serving_render.py`

**Interfaces:**
- Produces: `RouterSpec(image: str, serve: str, env: dict[str, str])`, `ServingSpec.router: RouterSpec | None`; `render(spec)` returns the V1 keys plus `"router-image"`, `"router-cmd"`, `"router-env"` (all `""` without a router); `split_serve(serve: str, field: str = "serve") -> list[str]`.

- [ ] **Step 1: Write failing schema tests.** In `utils/test_serving_schema.py`:

  Replace `test_router_rejected_with_v1_message` with:
  ```python
  ROUTER = textwrap.dedent("""\
      router:
        image: vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1
        serve: python3 -m dynamo.frontend --http-port=8888 --router-mode=kv
  """)


  def dynamo(text: str = GOOD) -> str:
      return (text.replace("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_02")
                  .replace("gpus: 1", "gpus: 8")
                  .replace("serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144",
                           "serve: python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --tp=8")
              + ROUTER)


  def test_router_block_loads():
      spec = load_serving(dynamo())
      assert spec.router.image.endswith(":1.5.0-kvoverlap1")
      assert spec.router.env == {}


  def test_router_unknown_key_rejected():
      assert "router.port: unknown key" in errors(dynamo() + "  port: 1\n")


  def test_router_needs_hardware_hcm_runner():
      text = dynamo().replace("runner: hardware-hcm-8x_02", "runner: h200-greennode-slurm-1x_01").replace("gpus: 8", "gpus: 1")
      assert "router: needs a hardware-hcm-8x_* runner (only launch_hardware-hcm.sh starts the router)" in errors(text)


  @pytest.mark.parametrize("field, text", [
      ("serve", GOOD.replace("--max-model-len 262144", "--max-model-len 262144 --numa-node 0 0 1 1")),
      ("router.serve", dynamo().replace("--router-mode=kv", "--router-mode=kv --numa-node 0")),
  ])
  def test_numa_node_rejected(field, text):
      assert f"{field}: drop --numa-node (benchmark convention)" in errors(text)


  def test_router_serve_gets_v1_command_checks():
      text = dynamo().replace("--router-mode=kv", "--router-mode=kv --model-path=/models/x --x=$Y")
      out = errors(text)
      assert "router.serve: local path containing '/models/'" in out
      assert "router.serve: shell variables ($...) are not expanded" in out


  def test_router_env_rules():
      text = dynamo() + "  env: {DYN_FILE_KV: /x, 'bad-name': '1'}\n"
      out = errors(text)
      assert "router.env.DYN_FILE_KV: reserved, the benchmark recipe sets it; remove it from env" in out
      assert "router.env.bad-name: not a valid environment variable name" in out


  def test_router_image_pattern():
      assert "router.image: only letters, digits and ._/:@- are allowed" in errors(dynamo().replace("1.5.0-kvoverlap1", "x;rm"))
  ```

  Update `test_runner_outside_serving_pool_rejected`. Add a 4x runner to the fixture: `cluster:hardware-hcm: [hardware-hcm-8x_01, hardware-hcm-4x_01]`. Then:
  ```python
  @pytest.mark.parametrize("runner", ["hardware-hcm-4x_01", "nope"])
  def test_runner_outside_serving_pool_rejected(repo, runner):
      text = with_line("runner: h200-greennode-slurm-1x_01", f"runner: {runner}").replace("gpus: 1", "gpus: 8")
      with pytest.raises(ServingError) as info:
          load_serving(text, repo)
      assert str(info.value) == (
          f"runner: {runner} cannot run a serving YAML; use one of: "
          "h200-greennode-slurm-1x_01, h200-greennode-slurm-1x_02, hardware-hcm-8x_01"
      )


  def test_hardware_hcm_8x_accepts_8_gpus(repo):
      text = with_line("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01").replace("gpus: 1", "gpus: 8")
      assert load_serving(text, repo).gpus == 8
  ```
  (`test_unknown_key_names_allowed_keys` keeps working: `router` becomes an allowed key name in the message.)

- [ ] **Step 2: Run and see them fail.**
  Run: `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_schema.py`
  Expected: the new tests FAIL (router rejected with the V1 message, old runner message).

- [ ] **Step 3: Implement the schema.** In `infx/serving/schema.py`:
  - Replace `SERVING_RUNNER_POOL` with:
    ```python
    # Pools whose launcher runs serve_generic.sh on SERVE_CMD, and which runner names in
    # each pool do (hardware-hcm: only the full-node 8x identities; smaller tiers share the box).
    SERVING_RUNNERS = {
        "cluster:h200-greennode-slurm": re.compile(r"h200-greennode-slurm-1x_\d+"),
        "cluster:hardware-hcm": re.compile(r"hardware-hcm-8x_\d+"),
    }
    # Only launch_hardware-hcm.sh starts the router sidecar.
    ROUTER_RUNNER_PREFIX = "hardware-hcm-"
    ```
  - Extend `RESERVED_ENV` with `"DYN_DISCOVERY_BACKEND", "DYN_FILE_KV", "DYN_SYSTEM_ENABLED", "DYN_SYSTEM_PORT", "ROUTER_IMAGE", "ROUTER_CMD", "ROUTER_ENV",` and add a comment line: `# Dynamo discovery and the router sidecar: set by runners/launch_hardware-hcm.sh.`
  - Move the `env` coercion into a module function and add `RouterSpec` above `ServingSpec`:
    ```python
    def _coerce_env(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            if isinstance(item, bool):
                raise ValueError(f'{key}: quote boolean values, e.g. "1" or "true"')
            out[key] = str(item) if isinstance(item, (int, float)) else item
        return out


    class RouterSpec(BaseModel):
        """Dynamo frontend in front of a dynamo.sglang engine (spec 2026-10-13 §1)."""

        model_config = ConfigDict(extra="forbid")

        image: str = Field(min_length=1)
        serve: str = Field(min_length=1)
        env: dict[str, str] = Field(default_factory=dict)

        @field_validator("env", mode="before")
        @classmethod
        def _env_strings(cls, value: Any) -> Any:
            return _coerce_env(value)
    ```
    In `ServingSpec` add `router: RouterSpec | None = None` after `serve`, and make its `_env_strings` validator `return _coerce_env(value)`.
  - Replace `_gpus_per_node`:
    ```python
    def _serving_runners(repo: Path) -> dict[str, int]:
        """Runner name -> gpus-per-node, for every runner that can run a serving YAML."""
        runners = yaml.safe_load((repo / "configs" / "runners.yaml").read_text())
        out: dict[str, int] = {}
        for pool, pattern in SERVING_RUNNERS.items():
            names = [name for name in (runners.get("labels") or {}).get(pool) or [] if pattern.fullmatch(name)]
            if not names:
                continue
            gpus = ((runners.get("hardware") or {}).get(pool) or {}).get("gpus-per-node")
            if gpus is None:
                raise ServingError(f"runner: pool {pool} has no gpus-per-node in configs/runners.yaml")
            out.update({name: int(gpus) for name in names})
        return out


    def _gpus_per_node(runner: str, repo: Path) -> int:
        allowed = _serving_runners(repo)
        if runner not in allowed:
            raise ServingError(f"runner: {runner} cannot run a serving YAML; use one of: {', '.join(allowed)}")
        return allowed[runner]
    ```
  - In `load_serving`, delete the `if "router" in raw:` block. In `_format_validation_error` the allowed-key list comes from the model of the failing location: for `loc[0] == "router"` use `RouterSpec.model_fields`:
    ```python
    if item["type"] == "extra_forbidden":
        model = RouterSpec if item["loc"][:1] == ("router",) else ServingSpec
        message = "unknown key (allowed: " + ", ".join(model.model_fields) + ")"
    ```
  - Replace the serve/env checks in `load_serving` with loops that cover both blocks:
    ```python
    commands = [("serve", spec.serve)] + ([("router.serve", spec.router.serve)] if spec.router else [])
    for field, command in commands:
        for marker in LOCAL_PATH_MARKERS:
            if marker in command:
                problems.append(
                    f"{field}: local path containing {marker!r}; replace it with the HF repo id from model"
                )
                break
        if "$" in command:
            problems.append(f"{field}: shell variables ($...) are not expanded; write literal values")
        if "--numa-node" in command:
            problems.append(f"{field}: drop --numa-node (benchmark convention)")
    envs = [("env", spec.env)] + ([("router.env", spec.router.env)] if spec.router else [])
    for prefix, env in envs:
        for key, value in env.items():
            if not ENV_NAME.fullmatch(key):
                problems.append(f"{prefix}.{key}: not a valid environment variable name")
            elif key in RESERVED_ENV:
                problems.append(f"{prefix}.{key}: reserved, the benchmark recipe sets it; remove it from env")
            if "\n" in value:
                problems.append(f"{prefix}.{key}: value must be a single line")
    if spec.router:
        if not IMAGE_PATTERN.fullmatch(spec.router.image):
            problems.append("router.image: only letters, digits and ._/:@- are allowed")
        if not spec.runner.startswith(ROUTER_RUNNER_PREFIX):
            problems.append("router: needs a hardware-hcm-8x_* runner (only launch_hardware-hcm.sh starts the router)")
    ```
    Keep the existing `image:` message text. Update the module docstring to say V1+V2 and name both specs.

- [ ] **Step 4: Run the schema tests.** Same command. Expected: all pass.

- [ ] **Step 5: Write failing render tests** in `utils/test_serving_render.py`:
  ```python
  from infx.serving.schema import RouterSpec

  PASTE_KV = '--kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'
  QUOTED_KV = """--kv-events-config='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'"""


  def dyn(engine: str = "python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --tp=8",
          router: str = "python3 -m dynamo.frontend --http-host=0.0.0.0 --http-port=8888 --router-mode=kv",
          **router_extra) -> ServingSpec:
      return spec(engine, runner="hardware-hcm-8x_02", gpus=8,
                  router=RouterSpec(image="img-r", serve=router, **router_extra))


  def test_dynamo_pair_renders_router_fields():
      out = render(dyn(router="FOO=1 python3 -m dynamo.frontend --http-host 0.0.0.0 --http-port=8888 --router-mode=kv",
                       env={"BAR": "2"}))
      assert out["framework"] == "sglang"
      assert shlex.split(out["serve-cmd"])[:3] == ["python3", "-m", "dynamo.sglang"]
      assert out["router-image"] == "img-r"
      assert shlex.split(out["router-cmd"]) == ["python3", "-m", "dynamo.frontend", "--router-mode=kv"]
      assert out["router-env"].splitlines() == ["FOO=1", "BAR=2"]


  def test_engine_only_has_empty_router_fields():
      out = render(spec("vllm serve Qwen/Qwen3.8-27B"))
      assert (out["router-image"], out["router-cmd"], out["router-env"]) == ("", "", "")


  def test_dynamo_engine_without_router_rejected():
      with pytest.raises(ServingError, match=r"^router: required for python3 -m dynamo.sglang"):
          render(spec("python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B"))


  def test_router_with_plain_engine_rejected():
      with pytest.raises(ServingError, match=r"^serve: with a router block the engine must be python3 -m dynamo.sglang"):
          render(dyn(engine="python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B"))


  def test_router_must_be_dynamo_frontend():
      with pytest.raises(ServingError, match=r"^router.serve: only python3 -m dynamo.frontend is supported"):
          render(dyn(router="smg launch --port 8888"))


  def test_unquoted_json_flag_from_the_paste_rejected():
      with pytest.raises(ServingError) as info:
          render(dyn(engine=f"python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B {PASTE_KV}"))
      assert str(info.value) == (
          "serve: --kv-events-config is not valid JSON after shell quoting; "
          "wrap it in single quotes: --kv-events-config='{...}'"
      )


  def test_single_quoted_json_flag_kept_verbatim():
      out = render(dyn(engine=f"python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B {QUOTED_KV}"))
      assert shlex.split(out["serve-cmd"])[-1] == (
          '--kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'
      )
  ```
  Add `(["python3", "-m", "dynamo.sglang", "--model-path=m"], "sglang")` to the `test_infer_framework` parametrize list.

- [ ] **Step 6: Run and see them fail.**
  Run: `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_render.py`
  Expected: FAIL (KeyError `router-image`, unknown engine for dynamo.sglang, …).

- [ ] **Step 7: Implement render.** In `infx/serving/render.py`:
  ```python
  import json

  _ROUTER_OWNED_FLAGS = ("--http-host", "--http-port")


  def split_serve(serve: str, field: str = "serve") -> list[str]:
      joined = re.sub(r"\\[ \t]*\r?\n", " ", serve)
      try:
          return shlex.split(joined, comments=True)
      except ValueError as error:
          raise ServingError(f"{field}: cannot parse the command ({error}); check the quotes") from error


  def _python_module(argv: list[str]) -> str | None:
      if len(argv) >= 3 and argv[0] in ("python", "python3") and argv[1] == "-m":
          return argv[2]
      return None


  def infer_framework(argv: list[str]) -> str:
      if argv[:2] == ["vllm", "serve"]:
          return "vllm"
      if argv[:2] == ["sglang", "serve"] or "sglang.launch_server" in argv or _python_module(argv) == "dynamo.sglang":
          return "sglang"
      raise ServingError(
          "serve: unknown engine; the command must start with `vllm serve`, "
          "`python3 -m sglang.launch_server`, `sglang serve` or `python3 -m dynamo.sglang`"
      )


  def _strip_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
      out: list[str] = []
      skip_value = False
      for arg in argv:
          if skip_value:
              skip_value = False
              continue
          if arg in flags:
              skip_value = True
              continue
          if arg.startswith(tuple(f"{flag}=" for flag in flags)):
              continue
          out.append(arg)
      return out


  def _check_json_values(field: str, argv: list[str]) -> None:
      """A flag value that starts with "{" must still be JSON once the shell removed its quotes."""
      for index, arg in enumerate(argv):
          if arg.startswith("--") and "=" in arg:
              flag, value = arg.split("=", 1)
          elif index and argv[index - 1].startswith("--") and "=" not in argv[index - 1]:
              flag, value = argv[index - 1], arg
          else:
              continue
          if not value.startswith("{"):
              continue
          try:
              json.loads(value)
          except ValueError:
              raise ServingError(
                  f"{field}: {flag} is not valid JSON after shell quoting; "
                  f"wrap it in single quotes: {flag}='{{...}}'"
              ) from None


  def _render_router(spec: ServingSpec, engine_argv: list[str]) -> dict[str, str]:
      is_dynamo_engine = _python_module(engine_argv) == "dynamo.sglang"
      if spec.router is None:
          if is_dynamo_engine:
              raise ServingError(
                  "router: required for python3 -m dynamo.sglang (a Dynamo engine has no HTTP port); "
                  "add a router block running python3 -m dynamo.frontend"
              )
          return {"router-image": "", "router-cmd": "", "router-env": ""}
      if not is_dynamo_engine:
          raise ServingError("serve: with a router block the engine must be python3 -m dynamo.sglang")
      prefix_env, argv = _pop_env_prefix(split_serve(spec.router.serve, "router.serve"))
      if _python_module(argv) != "dynamo.frontend":
          raise ServingError("router.serve: only python3 -m dynamo.frontend is supported")
      _check_json_values("router.serve", argv)
      env = {**prefix_env, **spec.router.env}
      return {
          "router-image": spec.router.image,
          "router-cmd": shlex.join(_strip_flags(argv, _ROUTER_OWNED_FLAGS)),
          "router-env": "\n".join(f"{key}={value}" for key, value in env.items()),
      }
  ```
  In `render`: after `framework = infer_framework(argv)`, call `_check_json_values("serve", argv)`, replace `_strip_owned_flags(argv)` with `_strip_flags(argv, _OWNED_FLAGS)` (delete `_strip_owned_flags`), compute `router = _render_router(spec, argv)` **before** stripping, and return `{..., **router}`. Update the module docstring: the router argv loses `--http-host/--http-port`, and `serve_router.sh` appends them.

- [ ] **Step 8: Run all serving tests.**
  Run: `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_*.py`
  Expected: all pass. If `test_serving_to_matrix.py::test_serving_object` fails only because of the 3 new keys, that is Task 3. Leave it red and note it.

- [ ] **Step 9: Commit.**
  ```bash
  git add infx/serving/schema.py infx/serving/render.py utils/test_serving_schema.py utils/test_serving_render.py
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): router block for a Dynamo frontend + dynamo.sglang pair; hardware-hcm-8x runners; numa and JSON-flag checks"
  ```

---

### Task 3: Router fields travel to the job env

**Files:**
- Modify: `infx/serving/to_matrix.py` (`build`)
- Modify: `.github/workflows/benchmark-tmpl.yml` (env block ~line 221)
- Test: `utils/test_serving_to_matrix.py`, `utils/test_serving_workflows.py`

**Interfaces:**
- Consumes: `render(spec)["router-image" | "router-cmd" | "router-env"]` (Task 2).
- Produces: `build(text)["serving"]` has keys `serve-cmd, serve-env, served-model-name, scenario, loader, yaml, router-image, router-cmd, router-env`. Job env `ROUTER_IMAGE`, `ROUTER_CMD`, `ROUTER_ENV` (empty strings on the engine-only path).

- [ ] **Step 1: Failing tests.** In `utils/test_serving_to_matrix.py`, extend the expected dict in `test_serving_object` with `"router-image": "", "router-cmd": "", "router-env": ""`, and add:
  ```python
  DYNAMO = textwrap.dedent("""\
      name: glm53-dyn
      runner: hardware-hcm-8x_02
      image: vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0
      model: PhalaCloud/GLM-5.3-W4AFP8
      gpus: 8
      precision: w4afp8
      serve: python3 -m dynamo.sglang --model-path=PhalaCloud/GLM-5.3-W4AFP8 --served-model-name=zai-org/GLM-5.3 --tp=8
      router:
        image: vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1
        serve: python3 -m dynamo.frontend --http-port=8888 --router-mode=kv
      scenario: agentx-cc-062126-smoke
      conc: [8]
  """)


  def test_dynamo_serving_object_and_rows():
      result = build(DYNAMO)
      serving = result["serving"]
      assert serving["router-image"].endswith(":1.5.0-kvoverlap1")
      assert serving["router-cmd"] == "python3 -m dynamo.frontend --router-mode=kv"
      assert serving["served-model-name"] == "zai-org/GLM-5.3"
      validate_matrix(result["matrix"])
      assert {row["runner"] for row in result["matrix"]} == {"hardware-hcm-8x_02"}
      assert result["matrix"][0]["tp"] == 8


  @pytest.mark.parametrize("path", sorted((REPO / "serving").glob("*.yaml")), ids=lambda p: p.name)
  def test_every_checked_in_serving_yaml_builds(path):
      build(path.read_text())
  ```
  Use the same `validate_matrix` call style as `test_rows_pass_the_workflow_schema` in that file. If it takes extra args, copy them.

  In `utils/test_serving_workflows.py` add:
  ```python
  def test_benchmark_tmpl_maps_router_fields():
      env = load("benchmark-tmpl.yml")["env"]
      for var, key in [("ROUTER_IMAGE", "router-image"), ("ROUTER_CMD", "router-cmd"), ("ROUTER_ENV", "router-env")]:
          assert env[var] == f"${{{{ inputs.serving != '' && fromJSON(inputs.serving).{key} || '' }}}}"
  ```
  (If `env` is not top-level in `benchmark-tmpl.yml`, find the mapping that holds `SERVE_CMD` and assert on that one.)

- [ ] **Step 2: Run and see them fail.** `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_to_matrix.py test_serving_workflows.py`. Expected: FAIL on the missing keys and env.

- [ ] **Step 3: Implement.** In `build()`, add to the `"serving"` dict:
  ```python
  "router-image": rendered["router-image"],
  "router-cmd": rendered["router-cmd"],
  "router-env": rendered["router-env"],
  ```
  In `benchmark-tmpl.yml`, under `SERVING_YAML:` add (same comment block, one line noting the hgx launcher):
  ```yaml
  # Router sidecar (Dynamo frontend); only runners/launch_hardware-hcm.sh reads these.
  ROUTER_IMAGE: ${{ inputs.serving != '' && fromJSON(inputs.serving).router-image || '' }}
  ROUTER_CMD: ${{ inputs.serving != '' && fromJSON(inputs.serving).router-cmd || '' }}
  ROUTER_ENV: ${{ inputs.serving != '' && fromJSON(inputs.serving).router-env || '' }}
  ```

- [ ] **Step 4: Run all serving tests.** Expected: all pass. If `actionlint` is installed, also run `actionlint .github/workflows/benchmark-tmpl.yml`.

- [ ] **Step 5: Commit.**
  ```bash
  git add infx/serving/to_matrix.py .github/workflows/benchmark-tmpl.yml utils/test_serving_to_matrix.py utils/test_serving_workflows.py
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): router-image/cmd/env in the serving object and ROUTER_* job env"
  ```

---

### Task 4: `serve_router.sh` and `wait_for_frontend_model`

**Files:**
- Create: `benchmarks/single_node/agentic/serve_router.sh`
- Modify: `benchmarks/benchmark_lib.sh` (add `wait_for_frontend_model` right after `wait_for_server_ready`, ~line 655)
- Test: create `utils/test_serving_router.py`

**Interfaces:**
- Produces: `serve_router.sh` reads `ROUTER_CMD`, `ROUTER_ENV`, `PORT`, `RESULT_DIR`, writes `$RESULT_DIR/router_command.txt`, and `exec`s `ROUTER_CMD --http-host 0.0.0.0 --http-port $PORT`.
- Produces: `wait_for_frontend_model --port P --model M --server-log F --server-pid PID --router-exited FILE [--sleep-interval S]`. Returns 0 when `http://localhost:P/v1/models` lists id `M`, and sets `INFERENCEX_SERVER_STATE` / `INFERENCEX_SERVER_PID` like `wait_for_server_ready`. Returns 1 when PID is gone or FILE exists.

- [ ] **Step 1: Failing tests** in a new `utils/test_serving_router.py`:
  ```python
  import json
  import subprocess
  import threading
  from http.server import BaseHTTPRequestHandler, HTTPServer
  from pathlib import Path

  import pytest

  REPO = Path(__file__).resolve().parents[1]
  ROUTER = REPO / "benchmarks/single_node/agentic/serve_router.sh"


  def test_serve_router_exports_env_appends_port_and_records_command(tmp_path):
      cmd = "python3 -c 'import os,sys; print(os.environ[\"FOO\"], *sys.argv[1:])' --router-mode=kv"
      out = subprocess.run(["bash", str(ROUTER)], cwd=REPO, capture_output=True, text=True, env={
          "PATH": "/usr/bin:/bin", "ROUTER_CMD": cmd, "ROUTER_ENV": "FOO=bar\n", "PORT": "9999",
          "RESULT_DIR": str(tmp_path), "DYN_DISCOVERY_BACKEND": "file", "DYN_FILE_KV": str(tmp_path / ".dynkv"),
      })
      assert out.returncode == 0, out.stderr
      assert out.stdout.splitlines()[-1] == "bar --router-mode=kv --http-host 0.0.0.0 --http-port 9999"
      assert "--http-port 9999" in (tmp_path / "router_command.txt").read_text()


  @pytest.fixture
  def frontend():
      models = {"data": []}

      class Handler(BaseHTTPRequestHandler):
          def do_GET(self):
              body = json.dumps(models).encode()
              self.send_response(200 if self.path == "/v1/models" else 404)
              self.end_headers()
              self.wfile.write(body)

          def log_message(self, *args):
              pass

      server = HTTPServer(("127.0.0.1", 0), Handler)
      threading.Thread(target=server.serve_forever, daemon=True).start()
      yield server.server_address[1], models
      server.shutdown()


  def wait(port: int, tmp_path: Path, pid_expr: str) -> subprocess.CompletedProcess:
      body = f"""
          source benchmarks/benchmark_lib.sh
          {pid_expr}
          wait_for_frontend_model --port {port} --model zai-org/GLM-5.3 --server-log {tmp_path}/server.log \\
              --server-pid "$pid" --router-exited {tmp_path}/router.exited --sleep-interval 0.2
      """
      (tmp_path / "server.log").write_text("engine log line\n")
      return subprocess.run(["bash", "-c", body], cwd=REPO, capture_output=True, text=True, timeout=60)


  def test_ready_when_model_listed(frontend, tmp_path):
      port, models = frontend
      models["data"] = [{"id": "zai-org/GLM-5.3"}]
      out = wait(port, tmp_path, "sleep 30 & pid=$!")
      assert out.returncode == 0, out.stderr


  def test_fails_fast_when_engine_exits(frontend, tmp_path):
      port, _ = frontend
      out = wait(port, tmp_path, "true & pid=$!; wait $pid")
      assert out.returncode == 1
      assert "engine exited before the frontend listed zai-org/GLM-5.3" in out.stderr
      assert "engine log line" in out.stderr


  def test_fails_fast_when_router_step_exited(frontend, tmp_path):
      port, _ = frontend
      (tmp_path / "router.exited").write_text("exit 1\n")
      out = wait(port, tmp_path, "sleep 30 & pid=$!")
      assert out.returncode == 1
      assert "router step exited (exit 1)" in out.stderr
  ```

- [ ] **Step 2: Run and see them fail.** `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_router.py`. Expected: FAIL (no script, unknown function).

- [ ] **Step 3: Implement `serve_router.sh`:**
  ```bash
  #!/usr/bin/env bash
  set -eo pipefail
  set -x

  # Dynamo frontend for the serving-YAML router path
  # (docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md §4).
  # runners/launch_hardware-hcm.sh starts it as an srun --overlap sidecar next to
  # the bench step; aiperf sends every request to $PORT.
  #   ROUTER_CMD  shell-quoted frontend argv, without --http-host/--http-port
  #   ROUTER_ENV  KEY=VALUE lines exported before the frontend starts
  # Discovery (DYN_DISCOVERY_BACKEND, DYN_FILE_KV) comes from the launcher.

  source "$(dirname "$0")/../../benchmark_lib.sh"

  check_env_vars ROUTER_CMD PORT RESULT_DIR DYN_DISCOVERY_BACKEND DYN_FILE_KV

  mkdir -p "$RESULT_DIR"
  # xtrace off: ROUTER_ENV values may hold tokens; log only the names.
  { set +x; } 2>/dev/null
  if [[ -n "${ROUTER_ENV:-}" ]]; then
      while IFS= read -r assignment; do
          if [[ -n "$assignment" ]]; then
              export "$assignment"
              echo "router env: ${assignment%%=*}"
          fi
      done <<< "$ROUTER_ENV"
  fi
  set -x

  eval "set -- $ROUTER_CMD"
  ROUTER_ARGV=("$@" --http-host 0.0.0.0 --http-port "$PORT")
  write_command "$RESULT_DIR/router_command.txt" "${ROUTER_ARGV[@]}"
  exec "${ROUTER_ARGV[@]}"
  ```
  `chmod +x` it. `write_command` tees to stdout. The test reads the **last** stdout line, which is the program's own output.

- [ ] **Step 4: Implement `wait_for_frontend_model`** in `benchmarks/benchmark_lib.sh`, directly after `wait_for_server_ready`:
  ```bash
  # Serving-YAML router path: the Dynamo engine has no HTTP port, so readiness is the
  # frontend on $port listing <model> (it only does once the engine registered). Fails
  # fast when the engine process exits or the launcher wrote <router-exited> (router step ended).
  wait_for_frontend_model() {
      local port="" model="" server_log="" server_pid="" router_exited="" sleep_interval=5
      while [[ $# -gt 0 ]]; do
          case $1 in
              --port) port="$2"; shift 2 ;;
              --model) model="$2"; shift 2 ;;
              --server-log) server_log="$2"; shift 2 ;;
              --server-pid) server_pid="$2"; shift 2 ;;
              --router-exited) router_exited="$2"; shift 2 ;;
              --sleep-interval) sleep_interval="$2"; shift 2 ;;
              *) echo "Unknown parameter: $1" >&2; return 1 ;;
          esac
      done
      if [[ -z "$port" || -z "$model" || -z "$server_log" || -z "$server_pid" || -z "$router_exited" ]]; then
          echo "Error: --port, --model, --server-log, --server-pid and --router-exited are required" >&2
          return 1
      fi
      while true; do
          if ! kill -0 "$server_pid" 2>/dev/null; then
              echo "ERROR: engine exited before the frontend listed $model; last lines of $server_log:" >&2
              tail -n 50 "$server_log" >&2
              return 1
          fi
          if [[ -f "$router_exited" ]]; then
              echo "ERROR: router step exited ($(cat "$router_exited")) before it listed $model; see results/router.log" >&2
              return 1
          fi
          if python3 - "$port" "$model" <<'PY' 2>/dev/null
  import json, sys, urllib.request
  port, model = sys.argv[1], sys.argv[2]
  with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as response:
      ids = [entry.get("id") for entry in json.load(response).get("data", [])]
  sys.exit(0 if model in ids else 1)
  PY
          then
              break
          fi
          sleep "$sleep_interval"
      done
      INFERENCEX_SERVER_STATE=$(mktemp /tmp/inferencex-server-state.XXXXXX) || return 1
      PYTHONPATH="$INFERENCEX_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench_serving.server_watch capture --pid "$server_pid" \
          > "$INFERENCEX_SERVER_STATE" || return 1
      INFERENCEX_SERVER_PID="$server_pid"
  }
  ```
  If `server_watch capture` fails on a `sleep` process in `test_ready_when_model_listed`, read `infx/bench_serving/server_watch.py`. Make the test PID something capture accepts (e.g. `python3 -c 'import time; time.sleep(30)' & pid=$!`). Do not weaken the function.

- [ ] **Step 5: Run.** `… test_serving_router.py test_serving_replay.py` → all pass. Also `bash -n benchmarks/benchmark_lib.sh benchmarks/single_node/agentic/serve_router.sh`.

- [ ] **Step 6: Commit.**
  ```bash
  git add benchmarks/single_node/agentic/serve_router.sh benchmarks/benchmark_lib.sh utils/test_serving_router.py
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): serve_router.sh frontend entrypoint and wait_for_frontend_model"
  ```

---

### Task 5: Dynamo branch in `serve_generic.sh`

**Files:**
- Modify: `benchmarks/single_node/agentic/serve_generic.sh`
- Test: `utils/test_serving_router.py` (add a static test), plus `bash -n`

**Interfaces:**
- Consumes: `wait_for_frontend_model` (Task 4); env `ROUTER_CMD`, `DYN_SYSTEM_PORT` (Task 6 launcher), `SERVED_MODEL_NAME`.
- Produces: with `ROUTER_CMD` set, the engine starts **without** `--host/--port`, readiness = `wait_for_frontend_model … --router-exited "$RESULT_DIR/.router_exited"`, and `AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics,http://localhost:$DYN_SYSTEM_PORT/metrics"`. Without `ROUTER_CMD`: byte-for-byte the V1 behaviour.

- [ ] **Step 1: Failing test** (append to `utils/test_serving_router.py`). It runs the script with every heavy step stubbed and captures what it would run:
  ```python
  GENERIC = REPO / "benchmarks/single_node/agentic/serve_generic.sh"
  STUB_LIB = r'''
  check_env_vars() { :; }
  install_agentic_deps() { :; }
  is_custom_dataset() { return 1; }
  resolve_trace_source() { :; }
  write_command() { local f="$1"; shift; printf '%q ' "$@" > "$f"; }
  wait_for_server_ready() { echo "WAIT health $*" >> "$RESULT_DIR/calls"; }
  wait_for_frontend_model() { echo "WAIT frontend $*" >> "$RESULT_DIR/calls"; }
  build_replay_cmd_yaml() { echo "METRICS $AIPERF_SERVER_METRICS_URLS" >> "$RESULT_DIR/calls"; }
  run_agentic_replay_and_write_outputs() { :; }
  '''


  @pytest.mark.parametrize("router", [True, False], ids=["dynamo", "engine-only"])
  def test_serve_generic_branches(tmp_path, router):
      lib = tmp_path / "benchmarks"
      (lib / "single_node/agentic").mkdir(parents=True)
      (lib / "benchmark_lib.sh").write_text(STUB_LIB)
      script = lib / "single_node/agentic/serve_generic.sh"
      script.write_text(GENERIC.read_text())
      bin_dir = tmp_path / "bin"
      bin_dir.mkdir()
      (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n")
      (bin_dir / "nvidia-smi").chmod(0o755)
      env = {
          "PATH": f"{bin_dir}:/usr/bin:/bin", "MODEL": "PhalaCloud/GLM-5.3-W4AFP8", "CONC": "8", "PORT": "8888",
          "RESULT_DIR": str(tmp_path / "results"), "FRAMEWORK": "sglang", "SERVING_SCENARIO": "s",
          "SERVING_LOADER": "l", "AIPERF_HF_CLI": "true", "SERVED_MODEL_NAME": "zai-org/GLM-5.3",
          "SERVE_CMD": "sleep 30",
      }
      if router:
          env |= {"ROUTER_CMD": "python3 -m dynamo.frontend", "DYN_SYSTEM_PORT": "9100"}
      out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)
      assert out.returncode == 0, out.stderr
      calls = (tmp_path / "results/calls").read_text()
      command = (tmp_path / "results/server_command.txt").read_text()
      if router:
          assert "WAIT frontend --port 8888 --model zai-org/GLM-5.3" in calls
          assert f"--router-exited {tmp_path}/results/.router_exited" in calls
          assert "METRICS http://localhost:8888/metrics,http://localhost:9100/metrics" in calls
          assert "--port" not in command
      else:
          assert "WAIT health --port 8888" in calls
          assert "METRICS http://localhost:8888/metrics" in calls
          assert command.strip().endswith("--host 0.0.0.0 --port 8888")
  ```

- [ ] **Step 2: Run and see the `dynamo` case fail.** `… test_serving_router.py -k serve_generic`.

- [ ] **Step 3: Implement.** In `serve_generic.sh`:
  - Header comment: add `ROUTER_CMD  set on the Dynamo router path: the frontend runs in a sidecar (serve_router.sh), the engine has no HTTP port` and `DYN_SYSTEM_PORT  Dynamo engine metrics/health port (launcher)`.
  - Replace `export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"` with:
    ```bash
    if [[ -n "${ROUTER_CMD:-}" ]]; then
        check_env_vars DYN_SYSTEM_PORT
        # Frontend (dynamo_frontend_*) + engine (sglang:* via the Dynamo system port).
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics,http://localhost:${DYN_SYSTEM_PORT}/metrics"
    else
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
    fi
    ```
    (If the Task 1 spike amendment says `sglang:*` is elsewhere, use that URL. If it says there is none, `unset AIPERF_REQUIRED_SERVER_METRIC_PREFIX` in this branch, with a comment pointing to the amendment.)
  - Replace the server start + wait block with:
    ```bash
    SERVER_LOG="$RESULT_DIR/server.log"
    eval "set -- $SERVE_CMD"
    if [[ -n "${ROUTER_CMD:-}" ]]; then
        SERVER_CMD=("$@")
    else
        SERVER_CMD=("$@" --host 0.0.0.0 --port "$PORT")
    fi
    write_command "$RESULT_DIR/server_command.txt" "${SERVER_CMD[@]}"
    "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if [[ -n "${ROUTER_CMD:-}" ]]; then
        wait_for_frontend_model --port "$PORT" --model "${SERVED_MODEL_NAME:-$MODEL}" \
            --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --router-exited "$RESULT_DIR/.router_exited"
    else
        wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
    fi
    ```
  - If the spike recorded that the offline download fails: change the download line to `HF_HUB_OFFLINE= "$AIPERF_HF_CLI" download "$MODEL"`, with the comment `# the user's env may set HF_HUB_OFFLINE=1 for the engine; the pre-download must reach the Hub`.

- [ ] **Step 4: Run all serving tests + `bash -n benchmarks/single_node/agentic/serve_generic.sh`.** Expected: pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_router.py
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): serve_generic Dynamo branch (engine without HTTP port, frontend readiness, two metrics URLs)"
  ```

---

### Task 6: `launch_hardware-hcm.sh` runs the serving path and the router sidecar

**Files:**
- Modify: `runners/launch_hardware-hcm.sh`
- Test: create `utils/test_serving_launcher_hcm.py`

**Interfaces:**
- Consumes: job env `SERVE_CMD`, `ROUTER_IMAGE`, `ROUTER_CMD`, `ROUTER_ENV` (Task 3); `serve_router.sh` (Task 4); `serve_generic.sh` reads `$RESULT_DIR/.router_exited` (Task 5).
- Produces: new knob `IX_SQUASH_DIR` (default `/mnt/containers`); env `DYN_DISCOVERY_BACKEND=file`, `DYN_FILE_KV=/workspace/results/.dynkv`, `DYN_SYSTEM_ENABLED=true`, `DYN_SYSTEM_PORT` for both containers when `ROUTER_CMD` is set; `results/router.log`; `results/.router_exited` (content `exit <rc>`) when the router step ends.

- [ ] **Step 1: Failing test** `utils/test_serving_launcher_hcm.py`. Model it on `utils/test_serving_launcher.py`. The srun stub also logs the DYN env it sees:
  ```python
  import subprocess
  from pathlib import Path

  import pytest

  LAUNCHER = Path(__file__).resolve().parents[1] / "runners/launch_hardware-hcm.sh"
  ROUTER_IMAGE = "vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1"


  def run_launcher(tmp_path: Path, extra_env: dict[str, str]) -> tuple[list[list[str]], str]:
      bin_dir = tmp_path / "bin"
      bin_dir.mkdir()
      log, envlog = tmp_path / "calls", tmp_path / "srun_env"
      stubs = {
          "hostname": "echo hgx-h200-02", "squeue": "echo 4711", "salloc": "", "scancel": "",
          "srun": f'echo "${{DYN_DISCOVERY_BACKEND:-none}} ${{DYN_FILE_KV:-none}} ${{DYN_SYSTEM_PORT:-none}}" >> {envlog}',
      }
      for name, body in stubs.items():
          (bin_dir / name).write_text(
              f"#!/bin/bash\nrec=$(printf '%s\\x1f' {name} \"$@\")\nprintf '%s\\x1e' \"$rec\" >> {log}\n{body}\n"
          )
          (bin_dir / name).chmod(0o755)
      workspace = tmp_path / "ws"
      (workspace / "results/.dynkv").mkdir(parents=True)
      (workspace / "results/.dynkv/stale").write_text("ghost worker\n")
      env = {
          "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "USER": "runner",
          "GPU_COUNT": "8", "IMAGE": "vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0",
          "GITHUB_WORKSPACE": str(workspace), "RUNNER_NAME": "hardware-hcm-8x_02", "FRAMEWORK": "sglang",
          "SPEC_DECODING": "none", "SCENARIO_SUBDIR": "agentic/", "EXP_NAME": "glm53-dyn_c8", "PRECISION": "w4afp8",
          "HF_HUB_CACHE": "/mnt/hf_hub_cache/", "IX_SQUASH_DIR": str(tmp_path / "containers"),
          **extra_env,
      }
      out = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
      assert out.returncode == 0, out.stderr[-2000:]
      calls = [record.split("\x1f")[:-1] for record in log.read_text().split("\x1e")[:-1]]
      return calls, envlog.read_text()


  def container_steps(calls):
      return [call for call in calls if call[0] == "srun" and any(a.startswith("--container-image=") for a in call)]


  @pytest.mark.parametrize("router", [True, False], ids=["dynamo", "engine-only"])
  def test_serving_path(tmp_path, router):
      extra = {"SERVE_CMD": "python3 -m dynamo.sglang --model-path=PhalaCloud/GLM-5.3-W4AFP8"}
      if router:
          extra |= {"ROUTER_IMAGE": ROUTER_IMAGE, "ROUTER_CMD": "python3 -m dynamo.frontend"}
      calls, envlog = run_launcher(tmp_path, extra)
      steps = container_steps(calls)
      # The router step runs in a background subshell, so call order is not fixed: match by script.
      bench = [s for s in steps if s[-1] == "benchmarks/single_node/agentic/serve_generic.sh"]
      assert len(bench) == 1 and bench[0][-2] == "bash"
      router_steps = [s for s in steps if s[-1] == "benchmarks/single_node/agentic/serve_router.sh"]
      imports = [c for c in calls if c[0] == "srun" and "enroot import" in c[-1]]
      if router:
          assert len(router_steps) == 1
          assert "--overlap" in router_steps[0]
          squash = str(tmp_path / "containers" / (ROUTER_IMAGE.replace("/", "_").replace(":", "_") + ".sqsh"))
          assert f"--container-image={squash}" in router_steps[0]
          assert any("/mnt/hf_hub_cache/:/mnt/hf_hub_cache/" in a for a in router_steps[0])
          assert len(imports) == 3
          assert "file /workspace/results/.dynkv" in envlog
          assert not (tmp_path / "ws/results/.dynkv").exists()
          assert (tmp_path / "ws/results/.router_exited").read_text().startswith("exit ")
      else:
          assert router_steps == []
          assert len(imports) == 2
          assert "file" not in envlog.split()
  ```
  The stub `srun` returns immediately, so the backgrounded router subshell writes `.router_exited` right away. The launcher's final `wait` (step 3) makes that deterministic.

- [ ] **Step 2: Run and see it fail.** `cd utils && uv run --no-project --with pytest --with pydantic --with pyyaml python -m pytest -q -p no:cacheprovider test_serving_launcher_hcm.py`.

- [ ] **Step 3: Implement in `runners/launch_hardware-hcm.sh`** (keep every existing comment that still applies):
  1. Squash cache knob, replacing `SQUASH_CACHE_DIR="/mnt/containers"`:
     ```bash
     # Local disk, not /shared (NFS) -- much faster for large image imports.
     # A runner's .env may set IX_SQUASH_DIR (tests point it at a temp dir).
     SQUASH_CACHE_DIR="${IX_SQUASH_DIR:-/mnt/containers}"
     ```
  2. Replace the two copy-pasted import blocks (engine and DCGM) with one function, defined after the `trap` that follows `salloc`, and its calls:
     ```bash
     squash_path() {
         echo "${SQUASH_CACHE_DIR}/$(echo "$1" | sed 's/[\/:@#]/_/g').sqsh"
     }
     # Import <image> into the shared squash cache once per box (flock: concurrent jobs).
     # Private registries (vcr.vngcloud.vn) authenticate through the runner user's
     # ~/.config/enroot/.credentials (spec 2026-10-13 §6); nothing here names an account.
     import_squash() {
         local image="$1" squash
         squash="$(squash_path "$image")"
         srun --jobid="$JOB_ID" bash -c "
             export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
             mkdir -p \$ENROOT_CACHE_PATH
             export TMPDIR=/mnt/tmp
             mkdir -p \$TMPDIR
             exec 9>\"${squash}.lock\"
             flock -w 600 9 || { echo 'Failed to acquire lock for $squash'; exit 1; }
             if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
                 echo 'Squash file already exists and is valid, skipping import'
             else
                 rm -f \"$squash\"
                 enroot import -o \"$squash\" $(enroot_uri_for_image "$image")
             fi
         " >&2
     }
     SQUASH_FILE="$(squash_path "$IMAGE")"
     import_squash "$IMAGE"
     ```
     And for DCGM: `DCGM_SQUASH="$(squash_path "$DCGM_IMAGE")"; import_squash "$DCGM_IMAGE"`. Delete the now-unused `DOCKER_IMAGE_URI`, `LOCK_FILE`, `DCGM_IMAGE_URI`, `DCGM_LOCK_FILE`.
  3. After the `BENCH_SCRIPT` formula:
     ```bash
     # Serving-YAML path: one generic recipe; the arm lives in SERVE_CMD & co.
     if [[ -n "${SERVE_CMD:-}" ]]; then
       BENCH_SCRIPT="benchmarks/single_node/agentic/serve_generic.sh"
     fi
     ```
  4. After the DCGM sidecar and its trap, before the bench `srun`:
     ```bash
     # Router sidecar (serving YAML with a router: block, spec 2026-10-13 §4): a Dynamo
     # frontend in its own image on $PORT. Frontend and engine find each other through
     # Dynamo's file discovery in the shared workspace; pyxis shares the host netns.
     ROUTER_SRUN_PID=""
     if [[ -n "${ROUTER_CMD:-}" ]]; then
         export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/workspace/results/.dynkv DYN_SYSTEM_ENABLED=true
         export DYN_SYSTEM_PORT="$(free_tcp_port)"
         mkdir -p "$GITHUB_WORKSPACE/results"
         # A previous run's registrations would show up as ghost workers.
         rm -rf "$GITHUB_WORKSPACE/results/.dynkv" "$GITHUB_WORKSPACE/results/.router_exited"
         ROUTER_SQUASH="$(squash_path "$ROUTER_IMAGE")"
         import_squash "$ROUTER_IMAGE"
         (
             rc=0
             srun --jobid="$JOB_ID" --overlap \
                 --container-image="$ROUTER_SQUASH" \
                 --container-mounts="$GITHUB_WORKSPACE:/workspace,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
                 --no-container-mount-home \
                 --container-remap-root \
                 --container-writable \
                 --container-workdir=/workspace/ \
                 --no-container-entrypoint --export=ALL,PORT="$PORT" \
                 bash benchmarks/single_node/agentic/serve_router.sh \
                 > "$GITHUB_WORKSPACE/results/router.log" 2>&1 || rc=$?
             # serve_generic.sh's readiness wait fails fast on this marker.
             echo "exit $rc" > "$GITHUB_WORKSPACE/results/.router_exited"
         ) &
         ROUTER_SRUN_PID=$!
     fi
     trap 'rc=$?; kill "$DCGM_SRUN_PID" $ROUTER_SRUN_PID 2>/dev/null || true; rm -rf "$GITHUB_WORKSPACE/results/.dynkv"; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM
     ```
     (This trap replaces the DCGM trap line. `.dynkv` is removed so it is not uploaded with `results/`.)
  5. At the end, replace `kill "$DCGM_SRUN_PID" 2>/dev/null || true` with:
     ```bash
     kill "$DCGM_SRUN_PID" $ROUTER_SRUN_PID 2>/dev/null || true
     scancel "$JOB_ID"
     wait 2>/dev/null || true
     rm -rf "$GITHUB_WORKSPACE/results/.dynkv"
     ```
     `scancel` ends the router step, so the subshell writes its marker and `wait` returns.
  6. Header comment: one line saying the serving-YAML path (`SERVE_CMD`, optional `ROUTER_*`) lives here, with a pointer to the V2 spec.
  7. **Etcd fallback, only if the Task 1 amendment says file discovery failed:** before the router block, start `srun --overlap --container-image=$(squash_path "$IX_ETCD_IMAGE") … etcd --listen-client-urls http://0.0.0.0:$ETCD_PORT --advertise-client-urls http://localhost:$ETCD_PORT --data-dir /workspace/results/.etcd &`. `IX_ETCD_IMAGE` defaults to `vcr.vngcloud.vn/108942-public69/dynamo-stack/etcd:v3.5.17` and `ETCD_PORT=$(free_tcp_port)`. Export `DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS=http://localhost:$ETCD_PORT` instead of the file vars. Add `ETCD_ENDPOINTS` to `RESERVED_ENV` in `schema.py`, and make the test expect `etcd` and 4 imports.

- [ ] **Step 4: Run all serving tests + `bash -n runners/launch_hardware-hcm.sh`.** Expected: pass. The 1x launcher tests are untouched and must still pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add runners/launch_hardware-hcm.sh utils/test_serving_launcher_hcm.py
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(runners): hardware-hcm serving path with a Dynamo router sidecar and file discovery"
  ```

---

### Task 7: Example YAML + `serving-bench` skill

**Files:**
- Create: `serving/glm53-w4afp8-dynamo.yaml`
- Modify: `.agents/skills/serving-bench/SKILL.md`
- Test: `test_every_checked_in_serving_yaml_builds` (Task 3) covers the YAML.

- [ ] **Step 1: Write `serving/glm53-w4afp8-dynamo.yaml`.** Use spec §1 verbatim, with a header comment:
  ```yaml
  # GLM-5.3 W4AFP8 on 8×H200 (hardware-hcm; runner _02, switch to hardware-hcm-8x_01 if box 02 is busy): Dynamo frontend (KV router) + dynamo.sglang engine,
  # as pasted by the user on 2026-10-13, minus --numa-node, with the HF id as model path and
  # --kv-events-config single-quoted. Dispatch: utils/serving/dispatch.sh serving/glm53-w4afp8-dynamo.yaml
  ```
  Keep `scenario: agentx-cc-062126-smoke`, `conc: [8]`.

- [ ] **Step 2: Check it.** `python3 -P -m infx.serving.to_matrix --check serving/glm53-w4afp8-dynamo.yaml` (needs pydantic+yaml: prefix with `uv run --no-project --with pydantic --with pyyaml`). Expected: one title line `serving glm53-w4afp8-dynamo agentx-cc-062126-smoke (180s) c8 hardware-hcm-8x_02`. Then run the full test command and expect all pass.

- [ ] **Step 3: Update the skill** `.agents/skills/serving-bench/SKILL.md`:
  - Schema block: change the runner comment to `# h200-greennode-slurm-1x_* (1×H200) or hardware-hcm-8x_* (8×H200, hgx-h200-01/02)`, and after `serve:` add the optional block:
    ```yaml
    router:                             # optional; only a Dynamo frontend in front of python3 -m dynamo.sglang
      image: <frontend image>
      env: {}                           # the frontend's env vars
      serve: |
        python3 -m dynamo.frontend --router-mode=kv ...
    ```
  - Translation rules, add:
    - **Router (Dynamo):** a frontend/router part (`python3 -m dynamo.frontend …`) goes under `router:` with its own `image`. The engine (`python3 -m dynamo.sglang …`) is `serve`. Engine env → `env:`, frontend env → `router.env:`. Leave out discovery settings (`DYN_*`, `ETCD_ENDPOINTS`, NATS): the benchmark wires them. Full example: `serving/glm53-w4afp8-dynamo.yaml` (a format reference only, never dispatched for a user's request).
    - **Drop `--numa-node …`** (with all its values). Benchmark convention.
    - **JSON in a flag must be single-quoted:** `--kv-events-config='{"publisher":"zmq",…}'`. A pasted `--x={"a":"b"}` loses its quotes.
    - **gpus 8** → runner `hardware-hcm-8x_01` (hgx-h200-01) or `hardware-hcm-8x_02` (hgx-h200-02). Pick the box `squeue -p hardware-hcm` shows free. Both follow the same env convention. If both are busy, pick one and tell the user it will queue.
    - Images on `vcr.vngcloud.vn` need nothing extra. The runner holds the pull credentials. Never put credentials in the YAML.
  - In "Ask the user": delete the router bullet. Replace the `gpus` bullet with: `gpus` > 1 on a `h200-greennode-slurm-1x_*` runner → if it is 8, use a `hardware-hcm-8x_*` runner (the free box), otherwise ask. Add: a router that is not `dynamo.frontend`, or an engine that is not `dynamo.sglang` next to it → say only the Dynamo pair is supported and ask.
  - Dispatch section, add: for `hardware-hcm-*` runners first run `ssh slurm-client squeue -p hardware-hcm`. If someone's job holds the node, tell the user the run will wait in the Slurm queue. Never cancel another user's job.
  - Results section, add `router.log` and `router_command.txt` to the `agentic_…` artifact list, and: "Router path: read `router.log` first when the job fails before aiperf starts."
  - Check that the frontmatter description still matches (add "or a Dynamo frontend + engine pair").

- [ ] **Step 4: Commit.**
  ```bash
  git add serving/glm53-w4afp8-dynamo.yaml .agents/skills/serving-bench/SKILL.md
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): GLM-5.3 Dynamo example and serving-bench router rules"
  ```

---

### Task 8: Smoke run on a free hgx box + spec amendments

**Files:**
- Modify: `docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md` (Amendments)

- [ ] **Step 1: Full local verification.** All serving tests; `bash -n` on `runners/launch_hardware-hcm.sh`, `benchmarks/single_node/agentic/serve_generic.sh`, `benchmarks/single_node/agentic/serve_router.sh`, `benchmarks/benchmark_lib.sh`; `to_matrix --check` on every `serving/*.yaml`. Record the output.

- [ ] **Step 2: Ask the user** to approve `git push origin feat/serving-config-dispatch` and the dispatch. Do not push without a yes.

- [ ] **Step 3: Pick the free box** (`ssh slurm-client squeue -p hardware-hcm`). If box 01 is the free one, set `runner: hardware-hcm-8x_01` in a copy of the YAML (or edit it and commit); if both are held, tell the user it will queue. Then dispatch:
  ```bash
  utils/serving/dispatch.sh serving/glm53-w4afp8-dynamo.yaml --ref feat/serving-config-dispatch
  ```
  It prints the run URL.

- [ ] **Step 4: Watch it** (skills `debug-agentx-runs` / `debug-runs` for live logs). Pass = run success; the `agentic_glm53-w4afp8-dynamo_c8_…` artifact has `results/router.log`, `router_command.txt`, `server.log`, `server_command.txt`, `serving.yaml`, `aiperf_artifacts/`; `server_metrics_export.json` contains `"sglang:` (`validate_required_agentic_server_metrics` passed). On failure: `router.log` → `server.log` → `gh run view <id> --repo vngcloud/InferenceX --log-failed | tail -100`. Use superpowers:systematic-debugging.

- [ ] **Step 5: Amend the spec.** Add "Amendments (2026-10-13, implementation)":
  - the row `tp` comes from `gpus` (V1 behaviour), so "`--tp=8` must parse" in §2 is moot;
  - `IX_SQUASH_DIR` knob;
  - `.router_exited` marker for fail-fast readiness;
  - the smoke run URL and its result.
  ```bash
  git add docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md
  git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "docs(spec): V2 implementation amendments and smoke result"
  ```

- [ ] **Step 6:** Run superpowers:verification-before-completion before telling the user it is done. Report: run URL, pass/fail, artifacts, and anything deferred.
