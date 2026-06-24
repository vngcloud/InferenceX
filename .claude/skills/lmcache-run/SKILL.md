---
name: lmcache-run
description: >
  Configures and dispatches an InferenceX benchmark with LMCache (CPU KV-offload)
  enabled on vLLM or SGLang. Use this skill whenever the user wants to run a benchmark
  with LMCache enabled, compare LMCache vs baseline performance, measure CPU KV-offload
  cache hit rates (server_lmcache_hit_rate), or test prefix-cache offloading to CPU DRAM.
  Applies to both the agentx-weka (agentic-coding) and agentic-replay (mooncake-trace)
  benchmark paths. Invoke whenever the user mentions "LMCache", "KV offload", "CPU cache",
  or wants to benchmark with KV cache persistence across requests.
---

# LMCache Run

Runs an InferenceX benchmark with LMCache (CPU KV-offload) enabled. LMCache intercepts
KV tensors produced during prefill and stores them in CPU DRAM, so repeated prompt prefixes
skip GPU recomputation on subsequent requests.

Structural differences from a plain benchmark:
1. LMCache serving flags added to the engine's launch command
2. A shared YAML config file mounted into the serving container (V1/SGLang path only)
3. A `-lmcache` suffix on the config key and model-prefix

Everything else — master config entry, perf-changelog, dispatch, watching — mirrors the
corresponding plain benchmark path.

---

## Ask the user first

Collect all answers before writing any files. Use `AskUserQuestion` — ask in two rounds
so the second round can be tailored to the chosen path.

### Round 1 — routing (ask these first, together)

1. **Benchmark path** — `agentx-weka` (weka/cc-traces corpus, `scenario-type: agentic-coding`)
   or `agentic-replay` (mooncake-trace dataset, `scenario-type: agentic-replay`)?
2. **Engine** — vLLM or SGLang? The LMCache wiring differs substantially between them.
3. **Model architecture** — does the model use hybrid attention? Look for `layer_types`
   containing `"linear_attention"`, `full_attention_interval > 0`, or `model_type` in
   `{qwen3_5, qwen3_next}` in the model's `config.json`. Hybrid models require the MP
   connector path (§ vLLM hybrid-attention below); the standard V1 path crashes at
   engine startup for these models.

### Round 2 — configuration (after routing is clear)

**Common to both paths:**

4. **Model + serving config** — HF model id, engine image, precision, TP, and any
   special serve flags (gpu-mem-util, kv-dtype, quantization, etc.). User typically
   pastes a `vllm serve …` / sglang launch line — that becomes the serve block in the
   script.
   - **Sanity-check `--model` / `--tokenizer`**: must be a plain HF slug
     (`namespace/repo`), not a local path (`/mnt/...`, `/models/...`). A raw path
     makes HF raise `OSError: Repo id must be in the form ...`. Strip leading dirs
     and confirm with the user before proceeding.
5. **Runner** — `runner:` field value verbatim. GreenNode options:
   `h100-greennode_00` (1×H100), `h100-greennode_01` (2×H100),
   `rtx5090-greennode_00` (1×RTX5090). Full list in `.github/configs/runners.yaml`.
   `search-space.tp` MUST match the runner's GPU count.
6. **Duration** — `900` (standard capacity) or `90` (smoke).
7. **Concurrency ladder** — `conc-list` in `search-space`, e.g. `[2, 4]` for smoke,
   `[8, 16, 32]` for capacity.
8. **LMCache CPU DRAM budget** — default `5.0 GB`. Increase if the runner has spare
   DRAM for a larger working set; decrease on RAM-constrained boxes.
9. **Baseline to compare against** — existing config-key for the same model/runner
   without LMCache? Note it for result comparison (optional).
10. **New branch?** — recommend `exp/<name>-lmcache`. Commit + dispatch from it
    (never `main`).
11. **DCGM?** — default no. If yes, see § DCGM below.

**agentx-weka path only:**

12. **Trace count** — full 949 (standard capacity) or smoke subset? Smoke: set
    `WEKA_NUM_DATASET_ENTRIES=64`. Full corpus takes 4–14 min to load on first run.

**agentic-replay path only:**

13. **Dataset** — which of the three:

    | Dataset | File under `benchmarks/single_node/agentic/datasets/` | Think-time |
    |---|---|---|
    | Agentic-coding | `agentic_coding_1variant_64k_150s.jsonl` | yes |
    | Claude-Code MiniMax production | `minimax_claude_code_prod_v3.jsonl` | yes |
    | Gemma blend_prod | `gemma_blend_prod.jsonl` | no — add `strip-trace-delays: true` |

---

## LMCache config file

The shared config is at **`benchmarks/lmcache_cpu.yaml`** (committed). Do not duplicate
it per-script — reference this single file via `LMCACHE_CONFIG_FILE`.

Key settings (edit only if the user explicitly requests):
- `max_local_cpu_size: 5.0` — GB of CPU DRAM for the KV cache
- `chunk_size: 256` — KV chunk granularity; changing this invalidates cached state
- `use_layerwise: True` — required by SGLang's layerwise connector; harmless for vLLM
- `internal_api_server_enabled: True` — exposes LMCache-native `/metrics` on port 7001

Inside the container the file is available at `/workspace/benchmarks/lmcache_cpu.yaml`
(GreenNode launchers mount the workspace at `/workspace`).

---

## Engine-specific additions to the launch script

### vLLM — full-attention models (in-process V1 connector)

Add to the **environment block** (before the `vllm serve` call):
```bash
export LMCACHE_CONFIG_FILE="/workspace/benchmarks/lmcache_cpu.yaml"
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
```

Add to the **`vllm serve` command**:
```bash
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
```

Why both flags: vLLM uses a two-tier cache — GPU HBM (fast, small) feeds into LMCache
CPU DRAM (larger, slower). `--enable-prefix-caching` activates the GPU tier that LMCache
intercepts when blocks are evicted. Without it the KV-transfer connector has nothing to
intercept.

LMCache `0.4.5` is pre-bundled in `vllm/vllm-openai:v0.21.0` — no extra `pip install`.

### SGLang — additions to any SGLang serving script

Add **before** the server launch (global install — not in an isolated venv):
```bash
pip install --break-system-packages "lmcache==0.4.5"
```

> **Pin `0.4.5` — mandatory.** LMCache 0.4.6+ adds a `config_file` positional argument
> to `LMCacheLayerwiseConnector.__init__()` that SGLang 0.5.12 never passes, causing an
> immediate `TypeError` crash on server startup. There is no workaround short of patching
> SGLang itself.

Add to the **environment block**:
```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE="/workspace/benchmarks/lmcache_cpu.yaml"
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
```

Add to the **`sglang.launch_server` command**:
```bash
  --enable-lmcache \
  --enable-metrics \
```

`--enable-metrics` is required — without it the `/metrics` endpoint is absent and
aiperf cannot scrape `sglang:cached_tokens_total` / `sglang:prompt_tokens_total`.

### vLLM — hybrid-attention models (MP connector, vLLM ≥ 0.23.0)

Use this path when the model has heterogeneous KV specs (linear_attention +
full_attention layers). The in-process `LMCacheConnectorV1` is NOT `SupportsHMA` and
causes vLLM to crash with `ValueError: failed to convert the KV cache specs to one
unified type`. The MP connector runs LMCache as a separate process.

**Image:** `vllm/vllm-openai:v0.23.0` (first release with `--mamba-cache-mode align`).
The bundled lmcache 0.4.6 is still not `SupportsHMA` — override it at runtime:

```bash
# Must run before lmcache server or vllm starts.
pip install --no-cache-dir "lmcache==0.5.0"
```

**Discover the unified block size N for the model** (once per model, before writing the
script). Boot vLLM without LMCache:

```bash
vllm serve <MODEL> --mamba-cache-mode align --enable-prefix-caching \
  --max-model-len 4096 2>&1 | grep -m1 "Setting attention block size"
# Expected: "Setting attention block size to N tokens"
# Qwen/Qwen3.5-4B → N = 528
```

**Start the standalone LMCache MP server** (ZMQ :5555) before `vllm serve`:

```bash
LMCACHE_CHUNK_SIZE=<N>   # set to the discovered block size
LMC_LOG="$RESULT_DIR/lmcache_server.log"
lmcache server \
  --chunk-size "$LMCACHE_CHUNK_SIZE" \
  --l1-size-gb "$TOTAL_CPU_DRAM_GB" \
  --eviction-policy LRU \
  --http-host 0.0.0.0 --http-port 8080 > "$LMC_LOG" 2>&1 &
LMC_PID=$!

# Wait for ZMQ listener
for i in $(seq 1 40); do
  grep -qiE "listening|started|serving|bound|fired|ready|MessageQueueServer|MPCacheServer" \
    "$LMC_LOG" 2>/dev/null && break
  kill -0 "$LMC_PID" 2>/dev/null || { cat "$LMC_LOG"; exit 1; }
  sleep 1
done
```

**`vllm serve` flags** (replace V1 flags entirely):

```bash
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
# DO NOT set LMCACHE_CONFIG_FILE — MP path uses server CLI flags, not a YAML.

vllm serve "$MODEL" \
  ... \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --max-num-batched-tokens "$LMCACHE_CHUNK_SIZE" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
  --trust-remote-code
```

Key constraints:
- `--chunk-size` (server) and `--max-num-batched-tokens` (vLLM) must both equal N.
- `LMCACHE_CONFIG_FILE` and `internal_api_server_enabled` are V1 path only — do not set them here.
- `--ipc=host` is needed only when server + vLLM run in separate Docker containers; inside the same benchmark container they share IPC by default.

**Reference implementation:** `benchmarks/single_node/agentic/qwen3.5-4b-weka-lmcache_bf16_h100_vllm.sh` (Qwen3.5-4B, N=528). Copy and change `LMCACHE_CHUNK_SIZE` for a different hybrid model.

---

## Path A — agentx-weka (`scenario-type: agentic-coding`)

The weka corpus is resolved and invoked entirely through aiperf (`resolve_trace_source` +
`build_replay_cmd` in `benchmark_lib.sh`). The submodule is **`utils/aiperf`** (vngcloud
fork, branch `cjq/weka-live-assistant-responses`). Do not cross submodules — agentic-replay
uses `utils/aiperf-mooncake`; weka uses `utils/aiperf`.

### A1 — Master-config entry

```yaml
<model-prefix>-weka-<hw>-<framework>-lmcache:   # -lmcache suffix on key
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-weka-lmcache           # -lmcache suffix on model-prefix
  precision: bf16
  framework: vllm
  runner: h100-greennode_00
  multinode: false
  scenarios:
    agentic-coding:
      duration: 90                               # 90 smoke / 900 standard
      search-space:
        { tp: 1, ep: 1, offloading: none, conc-list: [2, 4] }
```

`offloading: none` only — CPU/SSD KV offload is not wired for the weka launch path.
There is **no `input-file`** here (unlike agentic-replay).

### A2 — Launch script

Script path: `benchmarks/single_node/agentic/<model-prefix>-lmcache_<precision>_<hw>_<framework>.sh`

Copy the closest matching baseline:
- **vLLM** → `qwen3-4b-weka_bf16_h100_vllm.sh`
- **SGLang** → `minimaxm2.5-weka_fp8_h100_sglang.sh`

Apply ONLY the engine-specific LMCache env vars and serve flags from § Engine-specific above.
Keep everything else verbatim — especially `check_env_vars`, the **venv isolation block**
(vLLM; see Gotcha 1), the **NaN patch block** (SGLang; see Gotcha 2), `resolve_trace_source`,
`install_agentic_deps`, `build_replay_cmd`, and `write_agentic_result_json`.

There is **no `AIPERF_SOURCE_DIR` export** on this path — weka uses the submodule directly.

If the user requested a smoke trace count, add before `build_replay_cmd`:
```bash
export WEKA_NUM_DATASET_ENTRIES=64
```

### A3 — Weka-specific gotchas

**Gotcha 1 — vLLM dep isolation (critical).** `install_agentic_deps` upgrades
anyio / starlette / fastapi. vLLM v0.21.0 imports these lazily at request time; the
upgrade causes `_IncludedRouter has no attribute 'path'` and
`cannot import name 'TaskHandle' from anyio`. Fix: install aiperf into a **clean venv**
(no `--system-site-packages`) so vLLM keeps the image's untouched system python:

```bash
# (vLLM only) after the server is healthy:
AIPERF_VENV="${TMPDIR:-/tmp}/aiperf-venv"   # /tmp, NOT /workspace
python3 -m venv "$AIPERF_VENV"
source "$AIPERF_VENV/bin/activate"
resolve_trace_source
install_agentic_deps
```

**Gotcha 2 — SGLang NaN patch.** SGLang emits `sglang:fwd_occupancy=NaN`; orjson encodes
it as `null`, failing aiperf's `ServerMetricsRecordMessage` validation and dropping the
entire `/metrics` scrape (cache-hit rate lost). The SGLang reference launcher applies
`patches/aiperf-skip-nonfinite-server-metrics.patch` at runtime — keep that block when
copying the SGLang launcher.

**Gotcha 3 — runner `/mnt` disk.** HF cache + model files + docker layers share `/dev/sdc`.
A full disk kills the run with a blank conclusion. Check `ssh h100 'df -h /mnt'`; reclaim
with `sudo docker image prune -a -f` (typically frees ~100 GB). Confirm with the user before
deleting HF cache or model dirs.

### A4 — perf-changelog.yaml

```yaml
- config-keys:
    - <your-key>-lmcache
  description:
    - "LMCache CPU KV-offload (lmcache==0.4.5) + agentx-weka on <engine> <precision> TP<n> on <runner>"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-coding
```

### A5 — Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/agentic/<script>.sh
git submodule status utils/aiperf   # must be vngcloud fork, branch cjq/weka-live-assistant-responses
python3 utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --model-prefix <model-prefix>-lmcache --framework <fw>
ls -la benchmarks/lmcache_cpu.yaml
grep -E "LMCacheConnectorV1|enable-lmcache|LMCACHE" benchmarks/single_node/agentic/<script>.sh

git switch -c exp/<name>-lmcache && git add -p && git commit && git push -u origin exp/<name>-lmcache

gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=exp/<name>-lmcache \
  -f 'inputs[ref]=exp/<name>-lmcache' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --model-prefix <model-prefix>-lmcache --framework <fw>' \
  -f 'inputs[test-name]=<label>' \
  -f 'inputs[duration-override]='
```

---

## Path B — agentic-replay (`scenario-type: agentic-replay`)

Three datasets, all using `utils/aiperf-mooncake` (clean v0.9.0 fork, pinned via
`AIPERF_SOURCE_DIR`). Do not cross submodules — this path uses `utils/aiperf-mooncake`;
weka uses `utils/aiperf`.

### B1 — Master-config entry

```yaml
<model-prefix>-<precision>-<hw>-<framework>-lmcache:   # -lmcache suffix on key
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-2507-lmcache                  # -lmcache suffix on model-prefix
  precision: bf16
  framework: vllm
  runner: h100-greennode_00
  multinode: false
  scenarios:
    agentic-replay:
    - input-file: benchmarks/single_node/agentic/datasets/<dataset>.jsonl
      custom-dataset-type: mooncake_trace
      max-model-len: 131072            # must cover the trace's longest turn
      benchmark-client: [aiperf]
      no-fixed-schedule: true
      # strip-trace-delays: true       # ONLY for Gemma blend_prod (back-to-back)
      search-space:
      - { tp: 1, conc-list: [4] }
```

`duration` defaults to 1800 in the schema but is overridden at dispatch
(`duration-override`), so leave it out of the config.

### B2 — Launch script

Script path: `benchmarks/single_node/<model-prefix>-lmcache_<precision>_<hw>[_<framework>].sh`

Copy `qwen3-4b-2507_bf16_h100_vllm.sh` (closest agentic-replay baseline), then:
1. Apply the engine-specific LMCache env vars and serve flags from § Engine-specific above.
2. Keep verbatim: `check_env_vars`, the full `REPLAY_ARGS` block (`no-fixed-schedule`,
   `grace-period`, sampling, warmup, tokenizer passthrough), `STOP_ARGS` (duration),
   and `run_client_benchmark`.

**MANDATORY — `AIPERF_SOURCE_DIR` must be present.** Right after `source ../benchmark_lib.sh`:
```bash
export AIPERF_SOURCE_DIR="${INFMAX_CONTAINER_WORKSPACE:-/workspace}/utils/aiperf-mooncake"
```
Without this the run silently falls back to PyPI `aiperf==0.9.0` and any fork patches are lost.

### B3 — perf-changelog.yaml

```yaml
- config-keys:
    - <your-key>-lmcache
  description:
    - "LMCache CPU KV-offload (lmcache==0.4.5) + agentic-replay <dataset> on <engine> <precision> TP<n> on <runner>"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-replay
```

### B4 — Validate → commit → dispatch

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/<script>.sh
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml --config-keys <key>   # expect scenario-type=agentic-replay
ls -la benchmarks/lmcache_cpu.yaml
grep -E "LMCacheConnectorV1|enable-lmcache|LMCACHE" benchmarks/single_node/<script>.sh

git switch -c exp/<name>-lmcache && git add -p && git commit && git push -u origin exp/<name>-lmcache

gh api --method POST -H "Accept: application/vnd.github+json" \
  /repos/vngcloud/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=exp/<name>-lmcache \
  -f 'inputs[ref]=exp/<name>-lmcache' \
  -f 'inputs[generate-cli-command]=test-config --config-keys <key> --config-files .github/configs/nvidia-master.yaml' \
  -f 'inputs[test-name]=<label>' \
  -f 'inputs[duration-override]=<900|90>'
```

---

## Naming convention

Append `-lmcache` to both the config key and `model-prefix` so the LMCache run and any
baseline coexist cleanly in the config and on the dashboard. The `-lmcache` model-prefix
needs a corresponding dashboard declaration before results are ingested (see bench-config
skill § "Publishing results") — optional if you only need raw artifacts.

Branch name: `exp/<name>-lmcache` (top-level `ref` and inputs `ref` both point to the
branch — never `main`).

---

## DCGM (optional — only if the user said yes in Q11)

DCGM is a sidecar container on the runner, not part of the config/script. Edit the launcher
for the chosen runner — `runners/launch_<hw>-greennode.sh` — and paste this block right
before the model `docker run --rm \` line, then commit it on the same branch:

```bash
DCGM_IMAGE="${DCGM_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04}"
DCGM_NAME="dcgm-exporter-${RUNNER_NAME:-greennode}"
docker rm -f "$DCGM_NAME" 2>/dev/null || true
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name "$DCGM_NAME" "$DCGM_IMAGE"
trap 'docker rm -f "$DCGM_NAME" 2>/dev/null || true' EXIT
```

First-run check: if a host-level/k8s dcgm-exporter already holds port 9400
(`docker ps | grep dcgm`, `ss -ltn | grep 9400`), the sidecar fails to bind — surface
that before retrying.

---

## Watch + confirm LMCache is active

After the run starts, check `server.log` for the engine-specific initialization line:

**vLLM (V1 connector, full-attention):**
```
LMCacheConnectorV1 initialized
```
If absent, `LMCACHE_CONFIG_FILE` was not picked up — confirm the env var value matches
the container path.

**vLLM (MP connector, hybrid-attention):**

In `lmcache_server.log`:
```
MessageQueueServer listening on ...
MPCacheServer started ...
```
In `server.log`:
```
Prefix caching in Mamba 'align' mode is experimental
Setting attention block size to N tokens
```
If you see `Turning off hybrid kv cache manager because the KV connector does not support it`
followed by a `ValueError` crash → lmcache 0.5.0 was not installed before the server started.

**SGLang:**
```
lmcache.integration.sglang ... LMCacheLayerwiseConnector initialized
```
If you see `TypeError: __init__() got an unexpected keyword argument 'config_file'` →
wrong lmcache version (0.4.6+). Verify `pip install lmcache==0.4.5` ran before the server.

**agentic-replay — confirm the fork was used (not PyPI):**
```
[aiperf] CLI missing; installing from source: /workspace/utils/aiperf-mooncake
```
If instead you see `installing aiperf==0.9.0 from PyPI` → `AIPERF_SOURCE_DIR` export
is missing from the launch script.

---

## Reading LMCache results

> **MP connector path (hybrid models):** The `:7001` internal-API-server and `lmcache:*`
> Prometheus counters are absent — they belong to the in-process (V1) path only. Use
> `vllm:external_prefix_cache_{hits,queries}_total` exclusively. The result JSON fields
> `lmcache_hit_tokens` / `lmcache_query_tokens` / `server_lmcache_hit_rate` are populated
> from those counters and work identically on both connector stacks.

| Field | Engine | Meaning |
|---|---|---|
| `server_lmcache_hit_rate` | both | LMCache / GPU-proxy hit rate (0–1) |
| `lmcache_hit_tokens` | vLLM | tokens served from LMCache CPU DRAM |
| `lmcache_query_tokens` | vLLM | tokens that reached the LMCache tier |
| `server_gpu_cache_hit_rate` | both | GPU HBM prefix-cache hit rate |

**vLLM:** `lmcache_query_tokens > 0` confirms LMCache was actually queried. If it stays 0,
the GPU HBM tier absorbed all prefix reuse — normal for short prompts or single-pass
workloads. Increase concurrency or use a larger model to see LMCache activate.

**SGLang:** `server_lmcache_hit_rate` is derived from SGLang's native prefix-cache counters
(`sglang:cached_tokens_total / prompt_tokens_total`). The layerwise connector does not update
`lmcache:*` Prometheus counters, so `lmcache_hit_tokens` / `lmcache_query_tokens` will be
null for SGLang — use `server_lmcache_hit_rate` as the proxy.

**agentic-replay:** prefix-cache hit % also lives in `server_metrics_export.json`
(`prefix_cache_hits / prefix_cache_queries`), not only in `profile_export_aiperf.json`.

Quick sanity-check:
```bash
jq '{lmcache_hit_rate: .server_lmcache_hit_rate,
     gpu_hit_rate: .server_gpu_cache_hit_rate,
     lmc_queries: .lmcache_query_tokens}' \
  /path/to/artifacts/<config-key>/profile_<config-key>.json
```

If `lmcache_hit_rate` is null, verify `server_metrics_export.json` contains:
- vLLM: `vllm:external_prefix_cache_hits_total`, `vllm:external_prefix_cache_queries_total`
- SGLang: `sglang:cached_tokens_total`, `sglang:prompt_tokens_total`
