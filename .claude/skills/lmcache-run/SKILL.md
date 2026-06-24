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

The only structural differences from a plain benchmark are:
1. LMCache serving flags added to the engine's launch command
2. A shared YAML config file mounted into the serving container
3. A `-lmcache` suffix on the config key and model-prefix

Everything else — master config entry, perf-changelog, dispatch, watching — is identical
to the standard skill for the chosen benchmark path.

> **This skill builds on two sibling skills. Read the appropriate one first** before
> writing anything — the generic mechanics live there, not here.
>
> - **`agentx-weka-run`** — weka/cc-traces corpus, `scenario-type: agentic-coding`
> - **`agentic-replay-run`** — mooncake-trace dataset, `scenario-type: agentic-replay`
>
> This skill covers ONLY the LMCache deltas: serving flags, config file wiring, naming
> convention, and result interpretation.

---

## Ask the user first

Ask these **before** the parent skill's questions:

1. **Engine** — vLLM or SGLang? The LMCache wiring differs substantially between them.
2. **Benchmark path** — agentx-weka or agentic-replay? This determines which parent skill
   to follow and which `scenario-type` to use in `perf-changelog.yaml`.
3. **LMCache CPU DRAM budget** — default `5.0 GB`. Increase if the runner has spare DRAM
   and you want a larger working set; decrease on RAM-constrained boxes.
4. **Baseline to compare against** — is there an existing run of the same model/runner
   without LMCache? If yes, note its config-key for comparison.
5. **Model architecture** — does the model use hybrid attention? Check the model's
   `config.json` for `layer_types` containing `"linear_attention"`, or
   `full_attention_interval > 0`, or `model_type` in `{qwen3_5, qwen3_next}`.
   Hybrid models require the MP connector path (§ vLLM hybrid-attention below);
   the standard V1 path crashes at engine startup for these models.

Then proceed with the parent skill's standard questions (model, runner, duration, concurrency, etc.).

---

## LMCache config file

The shared config is at **`benchmarks/lmcache_cpu.yaml`** (committed). Do not duplicate
it per-script — reference this single file via `LMCACHE_CONFIG_FILE`.

Key settings (edit only if the user explicitly requests):
- `max_local_cpu_size: 5.0` — GB of CPU DRAM for the KV cache
- `chunk_size: 256` — KV chunk granularity; changing this invalidates cached state
- `use_layerwise: True` — required by SGLang's layerwise connector; harmless for vLLM
- `internal_api_server_enabled: True` — exposes LMCache-native `/metrics` on port 7001

Inside the container the file is available at whatever path the script mounts it.
In the GreenNode launcher scripts the workspace is mounted at `/workspace`, so the
file's container path is `/workspace/benchmarks/lmcache_cpu.yaml`.

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

Why both flags together: vLLM uses a two-tier cache — GPU HBM (fast, small) feeds into
LMCache CPU DRAM (larger, slower). `--enable-prefix-caching` activates the GPU tier that
LMCache intercepts when blocks get evicted. Without it, the KV-transfer connector has
nothing to intercept.

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

`--enable-metrics` is required — without it, the `/metrics` endpoint is absent and
aiperf cannot scrape `sglang:cached_tokens_total` / `sglang:prompt_tokens_total`.

### vLLM — hybrid-attention models (MP connector, vLLM ≥ 0.23.0)

Use this path when the model has heterogeneous KV specs (linear_attention +
full_attention layers). The in-process `LMCacheConnectorV1` is NOT `SupportsHMA`
and causes vLLM to crash with `ValueError: failed to convert the KV cache specs
to one unified type`. The MP connector runs LMCache as a separate process.

**Image:** `vllm/vllm-openai:v0.23.0` (first release with `--mamba-cache-mode align`).
The bundled lmcache 0.4.6 is still not `SupportsHMA` — override it at runtime:

```bash
# Must run before lmcache server or vllm starts.
pip install --no-cache-dir "lmcache==0.5.0"
```

**Discover the unified block size N for the model.** This is model-specific and must
be found once per model before writing the script. Boot vLLM without LMCache:

```bash
# One-time discovery — run inside the v0.23.0 container, kill after grep:
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

**`vllm serve` flags** (replace the V1 flags entirely):

```bash
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
# DO NOT set LMCACHE_CONFIG_FILE — the MP path uses server CLI flags, not a YAML.

vllm serve "$MODEL" \
  ... \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --max-num-batched-tokens "$LMCACHE_CHUNK_SIZE" \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' \
  --trust-remote-code
```

**Key constraints:**
- `--chunk-size` (server) and `--max-num-batched-tokens` (vLLM) must both equal N.
- `LMCACHE_CONFIG_FILE` and `internal_api_server_enabled` are in-process (V1) path only — do not set them here.
- `--ipc=host` is needed only when server + vLLM run in **separate Docker containers**; inside the same benchmark container they share IPC by default.

**Reference implementation:** `benchmarks/single_node/agentic/qwen3.5-4b-weka-lmcache_bf16_h100_vllm.sh` (Qwen3.5-4B, N=528). Copy and change `LMCACHE_CHUNK_SIZE` for a different hybrid model.

---

## Naming convention — config key and model-prefix

Append `-lmcache` to both the config key and `model-prefix` so the LMCache run and
any baseline coexist cleanly in the config and on the dashboard:

```yaml
# In .github/configs/nvidia-master.yaml
qwen3-4b-weka-h100-vllm-lmcache:        # <-- -lmcache suffix on key
  image: vllm/vllm-openai:v0.21.0
  model: Qwen/Qwen3-4B-Instruct-2507
  model-prefix: qwen3-4b-weka-lmcache    # <-- -lmcache suffix on model-prefix
  precision: bf16
  framework: vllm
  runner: h100-greennode_00
  multinode: false
  scenarios:
    agentic-coding:                      # or agentic-replay — match the parent skill
      duration: 90
      search-space:
        { tp: 1, ep: 1, offloading: none, conc-list: [2, 4] }
```

The `model-prefix` with `-lmcache` means you need a corresponding dashboard declaration
before results are ingested (see bench-config skill § "Publishing results"). This is
optional if you only need the raw artifacts and don't need dashboard visibility yet.

## Launch script name

Follow the parent skill's naming rule, using the `-lmcache` model-prefix:

```
# agentx-weka path:
benchmarks/single_node/agentic/qwen3-4b-weka-lmcache_bf16_h100_vllm.sh

# agentic-replay path:
benchmarks/single_node/qwen3-4b-2507-lmcache_bf16_h100_vllm.sh
```

**Create the script by copying the closest baseline reference script** (see parent skill)
and applying only the engine-specific env vars and serve flags above. Keep everything else
verbatim — especially `check_env_vars`, the venv isolation block (vLLM weka path),
the NaN patch block (SGLang weka path), `build_replay_cmd` / `run_client_benchmark`,
`AIPERF_SOURCE_DIR` export (agentic-replay path), and `write_agentic_result_json`.

---

## perf-changelog.yaml entry

Scenario-type must match the benchmark path:

```yaml
- config-keys:
    - qwen3-4b-weka-h100-vllm-lmcache
  description:
    - "LMCache CPU KV-offload (lmcache==0.4.5) + agentx-weka on vLLM v0.21.0 TP1 on h100-greennode_00"
  pr-link: https://github.com/vngcloud/InferenceX/pull/TBD
  scenario-type:
    - agentic-coding      # for agentx-weka; use agentic-replay for mooncake-trace path
```

---

## Validate → commit → dispatch

Follow the parent skill's validate/commit/dispatch section. Before committing, also run:

```bash
# Confirm the shared config is present
ls -la benchmarks/lmcache_cpu.yaml

# Confirm the LMCache serve flags appear in the script
grep -E "LMCacheConnectorV1|enable-lmcache|LMCACHE" benchmarks/single_node/agentic/<your-script>.sh
```

Branch name: use `exp/<name>-lmcache` to distinguish from any baseline run on the same model.

Dispatch: `ref=exp/<name>-lmcache` (top-level **and** inputs `ref`) — never `main`.

---

## Watch + confirm LMCache is active

After the run starts, check `server.log` for the engine-specific LMCache initialization line:

**vLLM (V1 connector, full-attention):**
```
LMCacheConnectorV1 initialized
```
If absent, `LMCACHE_CONFIG_FILE` was not picked up — confirm the env var value matches
the actual container path.

**vLLM (MP connector, hybrid-attention):**

In `lmcache_server.log` — one of:
```
MessageQueueServer listening on ...
MPCacheServer started ...
```
In `server.log` — confirms vLLM kept the hybrid KV manager on (SupportsHMA check passed):
```
Prefix caching in Mamba 'align' mode is experimental
Setting attention block size to N tokens
```
If instead you see `Turning off hybrid kv cache manager because the KV connector does
not support it` followed by a `ValueError` crash → lmcache 0.5.0 was not installed
(pip install ran after the server started, or was skipped).

**SGLang:**
```
lmcache.integration.sglang ... LMCacheLayerwiseConnector initialized
```
If you see `TypeError: __init__() got an unexpected keyword argument 'config_file'` →
the lmcache version is wrong (0.4.6+). Verify `pip install lmcache==0.4.5` ran before
the server started and that no later step re-installs without pinning.

---

## Reading LMCache results

> **MP connector path (hybrid models):** The `:7001` internal-API-server and `lmcache:*`
> Prometheus counters are absent — they belong to the in-process (V1) path only. Use
> `vllm:external_prefix_cache_{hits,queries}_total` exclusively. The result JSON fields
> `lmcache_hit_tokens` / `lmcache_query_tokens` / `server_lmcache_hit_rate` are populated
> from those counters and work identically on both connector stacks.

The result JSON carries these LMCache-specific fields (added by C3/C4 in the codebase):

| Field | Engine | Meaning |
|---|---|---|
| `server_lmcache_hit_rate` | both | LMCache / GPU-proxy hit rate (0–1) |
| `lmcache_hit_tokens` | vLLM | tokens served from LMCache CPU DRAM |
| `lmcache_query_tokens` | vLLM | tokens that reached the LMCache tier |
| `server_gpu_cache_hit_rate` | both | GPU HBM prefix-cache hit rate |

**vLLM interpretation:** `lmcache_query_tokens > 0` confirms LMCache was actually queried.
If it stays 0, the GPU HBM tier absorbed all prefix reuse — this is normal for short prompts
or a single-pass workload. Increase concurrency, run more turns, or use a larger model
(more KV blocks → GPU tier fills faster) to see LMCache activate.

**SGLang interpretation:** `server_lmcache_hit_rate` is derived from SGLang's native
prefix-cache counters (`sglang:cached_tokens_total / prompt_tokens_total`), not from
LMCache-native metrics. The SGLang layerwise connector does not update `lmcache:*`
Prometheus counters, so `lmcache_hit_tokens` / `lmcache_query_tokens` will be null for
SGLang — use `server_lmcache_hit_rate` as the proxy for cache effectiveness.

**The 0/0 trap:** if `lmcache_hit_tokens = 0` and `lmcache_query_tokens = 0` on vLLM,
LMCache was never queried (all reuse came from the GPU tier). Not a failure — it means
the workload fits in GPU HBM. A good LMCache workload has either (a) long shared prefixes
that saturate the GPU cache, or (b) repeated requests across a cold-start gap.

Quick sanity-check after the run:
```bash
jq '{lmcache_hit_rate: .server_lmcache_hit_rate,
     gpu_hit_rate: .server_gpu_cache_hit_rate,
     lmc_queries: .lmcache_query_tokens}' \
  /path/to/artifacts/<config-key>/profile_<config-key>.json
```

Expected shape: `lmcache_hit_rate` non-null, `gpu_hit_rate` non-null, `lmc_queries`
non-null and positive (vLLM) or null (SGLang is expected).

If `lmcache_hit_rate` is null: check that `server_metrics_export.json` exists in the
artifact directory and contains the right metric names:
- vLLM: `vllm:external_prefix_cache_hits_total`, `vllm:external_prefix_cache_queries_total`
- SGLang: `sglang:cached_tokens_total`, `sglang:prompt_tokens_total`
