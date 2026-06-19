# LMCache Integration Plan

Branch: `lmcache-integration`

---

## What is LMCache and why integrate it?

LMCache is a KV-cache management layer that sits alongside vLLM or SGLang. It pools and reuses KV blocks across requests and sessions — including across nodes when configured with a shared backend (Redis, Infiniband RDMA, etc.). The motivation is to measure whether LMCache materially improves throughput and cache hit rates on prefix-heavy workloads (RAG, multi-turn, agentic replay).

---

## How LMCache plugs into the serving pipeline

```
Config (nvidia-master.yaml)
    framework: vllm-lmcache          ← new framework variant
    image: <lmcache-enabled image>   ← must have LMCache installed
          │
          ▼
generate_sweep_configs.py  →  matrix JSON
          │
          ▼
benchmark-tmpl.yml  →  env vars including FRAMEWORK=vllm-lmcache
          │
          ▼
runners/launch_h100-greennode.sh
    BENCH_SCRIPT = benchmarks/single_node/{prefix}_{prec}_h100_vllm-lmcache.sh
          │
          ▼
benchmarks/single_node/{prefix}_{prec}_{hw}_vllm-lmcache.sh
    python3 -m vllm.entrypoints.openai.api_server \
        --kv-transfer-config '{"kv_connector":"LMCacheConnector",...}' ...
    aiperf ... --server-metrics-url http://localhost:$PORT/metrics
          │
          ▼
server_metrics_export.json  (written by AIPerf)
    vllm:prefix_cache_hits, vllm:prefix_cache_queries
    lmcache_local_hit_tokens, lmcache_local_query_tokens   ← LMCache-specific
          │
          ▼
aiperf_adapter.py  →  build_result() now includes cache fields
          │
          ▼
process_result.py  →  explicit passthrough of cache fields
          │
          ▼
agg_{filename}.json  →  dashboard ingestion
```

LMCache does NOT replace vLLM or SGLang. It is a plugin. vLLM exposes it via `--kv-transfer-config` (v0.7+). SGLang support is more experimental; treat vLLM as the primary target for this integration.

---

## Where do cache metrics come from?

There are three independent sources, each measuring a different layer:

### Source 1 — vLLM's Prometheus endpoint (`/metrics`)

vLLM always exposes these counters regardless of whether LMCache is enabled:

| Prometheus key | Meaning |
|---|---|
| `vllm:prefix_cache_hits` | KV blocks served from vLLM's built-in prefix cache |
| `vllm:prefix_cache_queries` | Total KV block lookups |
| `vllm:cpu_prefix_cache_hits` | CPU-tier cache hits (when KV offloading is on) |
| `vllm:cpu_prefix_cache_queries` | CPU-tier lookups |

`server_gpu_cache_hit_rate = vllm:prefix_cache_hits / vllm:prefix_cache_queries`

This works even without LMCache. It reflects vLLM's own RadixAttention/prefix-cache layer.

### Source 2 — LMCache's Prometheus metrics (only when LMCache is active)

When LMCache is running, it exposes additional counters at the same `/metrics` endpoint:

| Prometheus key | Meaning |
|---|---|
| `lmcache_local_hit_tokens` | Tokens served from local (GPU/CPU) LMCache |
| `lmcache_local_query_tokens` | Total tokens looked up in local LMCache |
| `lmcache_remote_hit_tokens` | Tokens served from remote LMCache tier (Redis/RDMA) |
| `lmcache_remote_query_tokens` | Total tokens looked up in remote LMCache |

`lmcache_local_hit_rate = lmcache_local_hit_tokens / lmcache_local_query_tokens`

These are absent from `/metrics` when LMCache is not running, so the adapter must treat them as optional.

### Source 3 — OpenAI API response field `cached_tokens` (per-request)

vLLM and SGLang both return `usage.prompt_tokens_details.cached_tokens` in each response. This is already collected in the agentic path as `usage_prompt_cache_read_tokens`. For fixed-seq-len runs this can be aggregated across all responses by `benchmark_serving.py` (inferencex_native client) or by AIPerf (via its per-request profile export). This gives a client-side cross-check of the server-side Prometheus counters.

### What happens when LMCache is NOT configured?

- Source 1 (vLLM counters) still works — vLLM's internal prefix cache is always active.
- Source 2 (LMCache counters) simply won't appear in `/metrics` → adapter sets those fields to `null`.
- Source 3 (cached_tokens) still works — it reflects whatever the engine cached.

No special-casing is needed in the result processing code; optional fields that are absent remain `null` in the output JSON.

---

## Files to change

### 1. `.github/configs/nvidia-master.yaml`

**What:** Add `vllm-lmcache` config entries.

**How:** Mirror an existing vLLM config entry but change `framework` and `image`:

```yaml
gemma4-fp8-h100-2x-vllm-lmcache:
  image: lmcache/vllm-openai:nightly-2026-06-17   # see "Image decision" below
  model: google/gemma-4-31B-it
  model-prefix: gemma4
  runner: h100-2x
  precision: fp8
  framework: vllm-lmcache               # drives script name lookup
  multinode: false
  scenarios:
    fixed-seq-len:
    - { isl: 1024, osl: 1024, search-space: [ { tp: 2, conc-start: 4, conc-end: 16, spec-decoding: none } ] }
```

No schema changes required — `framework` is already a free-form string.

#### Image decision

LMCache publishes `lmcache/vllm-openai` on Docker Hub. The relevant tags as of 2026-06-18:

| Tag | Notes |
|---|---|
| `lmcache/vllm-openai:v0.4.7` | Latest stable. Changelog explicitly fixes "vLLM 0.20+ pydantic error". 8.8 GB. |
| `lmcache/vllm-openai:v0.4.7-cu129` | Same + CUDA 12.9 build. 11.95 GB. H100/H200 greennode runners use cu121 → skip. |
| `lmcache/vllm-openai:nightly-2026-06-17` | Nightly tracking latest vLLM main. Closer to vLLM v0.21.0 which InferenceX uses. |

**Recommendation: `lmcache/vllm-openai:nightly-2026-06-17`**

Reasoning: InferenceX currently uses `vllm/vllm-openai:v0.21.0`. The stable `v0.4.7` image bundles an older vLLM (the March 2026 nightly shipped vLLM 0.18.3.dev). The June 17 nightly tracks the most recent vLLM main, which should be at or past v0.21.0. Using a nightly image with a recent vLLM minimises architecture/performance deltas vs the baseline, making the LMCache comparison more meaningful.

**Alternative (maximum reproducibility):** Build a thin custom image so the vLLM version is guaranteed identical to baseline:
```dockerfile
FROM vllm/vllm-openai:v0.21.0
RUN pip install lmcache==0.4.7
```
Push to a registry and use that tag in the config. This is the most defensible choice if strict numeric comparability matters.

---

### 2. `benchmarks/single_node/{prefix}_{prec}_{hw}_vllm-lmcache.sh` (new files)

**What:** New benchmark scripts that launch vLLM with LMCache enabled (local-only CPU tier) and pass `--server-metrics-url` to AIPerf so it scrapes `/metrics` and writes `server_metrics_export.json`.

**How:** Derived from the corresponding `_vllm.sh` script with two changes: the server launch flags and the AIPerf call.

```bash
#!/usr/bin/env bash
# Example: gemma4_fp8_h100_vllm-lmcache.sh
# Same as gemma4_fp8_h100.sh (vLLM path) but with LMCache local-only CPU tier.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME

nvidia-smi
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MAX_SEQ_LEN=$((ISL + OSL + 20))

start_gpu_monitor

# --- LMCache local-only config (env vars, no external config file needed) ---
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5       # GB; tune per runner's DRAM headroom

set -x
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  ... \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# AIPerf with server-metrics scraping so cache stats end up in the result JSON
run_benchmark_aiperf \
    --model "$MODEL" \
    --url "http://localhost:${PORT}/v1" \
    --concurrency "$CONC" \
    --isl "$ISL" \
    --osl "$OSL" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --server-metrics-url "http://localhost:${PORT}/metrics"   # ← key addition

stop_gpu_monitor
set +x
```

**Key flags explained:**

| Flag / env var | Value | Purpose |
|---|---|---|
| `LMCACHE_LOCAL_CPU=True` | env var | Activates local CPU-RAM tier only; no Redis/remote |
| `LMCACHE_MAX_LOCAL_CPU_SIZE=5` | env var | Cap in GB; prevents OOM on runner (tune per box) |
| `LMCACHE_CHUNK_SIZE=256` | env var | KV block chunk granularity (tokens); 256 is LMCache default |
| `--kv-transfer-config` | JSON | Wires vLLM's KV-transfer layer to LMCacheConnectorV1; `kv_role=kv_both` = single-node (no prefill/decode split) |
| `--server-metrics-url` | AIPerf flag | Tells AIPerf to scrape this endpoint and write `server_metrics_export.json` to the artifact dir |

**No config file needed.** The three `LMCACHE_*` env vars fully describe the local-only policy; the `lmcache_config.yaml` file approach is only needed for remote/Redis tier configuration.

**Script naming convention** (existing rule, unchanged):
`{model-prefix}_{precision}_{hw}_{framework}.sh`
→ `gemma4_fp8_h100_vllm-lmcache.sh`

---

### 3. `runners/launch_*.sh`

**What:** Pass `USE_LMCACHE` (derived from `FRAMEWORK`) through to the container if needed. Also ensure `--server-metrics-url` reaches the bench script via an env var.

**How:** The launcher already derives the bench script name from `FRAMEWORK`, so `vllm-lmcache` scripts will be picked up automatically — no launcher changes required for script dispatch.

However, the env var `SERVER_METRICS_URL` should be added to `RUN_ENV` so that benchmark scripts can reference it without hardcoding the port:

```bash
# In launch_h100-greennode.sh and other launchers, add to RUN_ENV:
RUN_ENV=(
  ...
  SERVER_METRICS_URL   # new: http://localhost:$PORT/metrics for AIPerf scraping
)
```

Then in `benchmark-tmpl.yml`, set:
```yaml
SERVER_METRICS_URL: "http://localhost:${{ env.PORT }}/metrics"
```

---

### 4. `.github/workflows/benchmark-tmpl.yml`

**What:** Add `server-metrics-url` input and `SERVER_METRICS_URL` env var so benchmark scripts don't need to hardcode the port.

```yaml
inputs:
  server-metrics-url:
    description: "Prometheus /metrics endpoint for AIPerf server-metrics scraping. Auto-set to http://localhost:{port}/metrics for LMCache runs."
    required: false
    type: string
    default: ''

env:
  SERVER_METRICS_URL: ${{ inputs.server-metrics-url }}
```

Alternatively, the benchmark scripts can derive `http://localhost:${PORT}/metrics` themselves since `PORT` is already in the environment — this avoids a workflow change entirely. **Recommendation:** derive in the script; skip the workflow change.

---

### 5. `utils/bench_serving/server_metrics.py` (new shared utility)

**What:** Extract the `_index_server_metrics` and `_final_value` logic that already exists in `process_agentic_result.py` into a standalone module so `aiperf_adapter.py` can reuse it without circular imports or code duplication.

```python
# utils/bench_serving/server_metrics.py

def load_server_metrics(path) -> dict:
    """Load server_metrics_export.json; return {} if missing/malformed."""
    ...

def index_server_metrics(server_metrics: dict) -> dict[str, dict]:
    """Return {metric_name: entry_dict} from AIPerf's server_metrics_export.json."""
    ...

def final_value(metrics_by_name: dict, metric_name: str) -> float | None:
    """Sum the total/max/avg stat across all series for a given metric name."""
    ...

def extract_cache_stats(server_metrics: dict) -> dict:
    """
    Return a dict of cache metrics from a server_metrics_export.json blob.
    All keys are present; values are None when the metric was absent.

    Keys:
      server_gpu_cache_hit_rate     # vllm:prefix_cache_hits / queries
      server_cpu_cache_hit_rate     # vllm:cpu_prefix_cache_hits / queries
      lmcache_local_hit_rate        # lmcache_local_hit_tokens / query_tokens
      lmcache_remote_hit_rate       # lmcache_remote_hit_tokens / query_tokens
    """
    ...
```

**Update `process_agentic_result.py`** to import from this module instead of carrying its own copy. This is a refactor-only change — no behavior change.

---

### 6. `utils/bench_serving/aiperf_adapter.py`

**What:** After `run_aiperf()` or `run_search()` completes, read `server_metrics_export.json` from the artifact dir and append cache stats to the result dict.

**Changes to `build_result()`** (lines 89–108): add optional `server_metrics` parameter:

```python
def build_result(artifact: dict, max_concurrency: int, server_metrics: dict | None = None) -> dict:
    result = {
        "model_id": ...,
        ...existing fields...
    }
    if server_metrics:
        result.update(extract_cache_stats(server_metrics))
    return result
```

**Changes to `run_fixed()`** (lines 379–383):

```python
def run_fixed(args: argparse.Namespace) -> dict:
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    artifact = run_aiperf(args, args.concurrency, artifact_dir)
    server_metrics = load_server_metrics(artifact_dir / "server_metrics_export.json")
    return build_result(artifact, extract_max_concurrency(artifact), server_metrics)
```

**Changes to `run_search()`** (lines 386–414): same — load `server_metrics_export.json` from the winner iteration's directory and pass to `build_result()`.

**No new CLI flags needed** — `--server-metrics-url` is already accepted (line 301) and forwarded to AIPerf (line 124–125). When passed, AIPerf writes `server_metrics_export.json`; when not passed, the file won't exist and `load_server_metrics` returns `{}`, producing all-`None` cache fields.

---

### 7. `utils/process_result.py`

**What:** The existing passthrough loop (lines 141–146) only propagates `*_ms` and `tpot` fields. Cache rate fields like `server_gpu_cache_hit_rate` are floats whose names don't end in `_ms`, so they are silently dropped today.

**Change:** Add an explicit passthrough for cache stat keys:

```python
CACHE_STAT_KEYS = (
    "server_gpu_cache_hit_rate",
    "server_cpu_cache_hit_rate",
    "lmcache_local_hit_rate",
    "lmcache_remote_hit_rate",
)

for key in CACHE_STAT_KEYS:
    if key in bmk_result and bmk_result[key] is not None:
        data[key] = bmk_result[key]
```

Place this block after line 146 (after the existing `*_ms` / `tpot` loop).

---

### 8. `utils/matrix_logic/validation.py`

No changes required for the initial integration. The `framework` field is already a free-form string — `vllm-lmcache` passes validation without a schema change. Adding a formal `use_lmcache: bool` field to `SingleNodeMatrixEntry` is a follow-up if per-row fine-grained control (e.g., enabling LMCache for some concurrency points but not others) is needed.

---

### 9. `utils/bench_serving/benchmark_serving.py` (inferencex_native client, optional)

The `inferencex_native` client path does not support `--server-metrics-url` today. For the initial integration AIPerf is the target client (it has native Prometheus scraping), so this is **deferred**.

If `inferencex_native` + LMCache metrics are needed later, the change is:
- Add `num_cached_tokens: int = 0` to `RequestFuncOutput`
- In the per-backend request functions, extract `response["usage"]["prompt_tokens_details"]["cached_tokens"]` if present
- Aggregate in `calculate_metrics()`: `cache_hit_rate = sum(cached) / sum(input_tokens)`
- Emit as `response_cache_hit_rate` in the result JSON

---

## Implementation order

| Step | File(s) | Notes |
|---|---|---|
| 1 | `utils/bench_serving/server_metrics.py` (new) | Extract shared utility first; no risk |
| 2 | `utils/process_agentic_result.py` | Import from new module; pure refactor, tests should still pass |
| 3 | `utils/bench_serving/aiperf_adapter.py` | Load server_metrics_export.json, call extract_cache_stats |
| 4 | `utils/process_result.py` | Add CACHE_STAT_KEYS passthrough |
| 5 | `benchmarks/single_node/lmcache/lmcache_config.yaml` (new) | LMCache local-only config |
| 6 | `benchmarks/single_node/gemma4_fp8_h100_vllm-lmcache.sh` (new, one example) | Prove the pipeline end-to-end |
| 7 | `.github/configs/nvidia-master.yaml` | Add one config entry using the new script |
| 8 | Validate + dispatch a one-off benchmark to verify metrics appear in result JSON |
| 9 | Add more `_vllm-lmcache.sh` scripts for other model-prefix/hw combinations | Fan out once step 6 is confirmed |

---

## Testing the changes locally

```bash
# Unit tests (validation + matrix generation)
python -m pytest utils/matrix_logic/ -v

# Validate that the new config entry generates a valid matrix
python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys gemma4-fp8-h100-2x-vllm-lmcache

# Smoke test the aiperf adapter's server_metrics parsing
# (create a mock server_metrics_export.json and run the adapter logic)
python -c "
from utils.bench_serving.server_metrics import load_server_metrics, extract_cache_stats
import json, pathlib
mock = {'metrics': {
    'vllm:prefix_cache_hits': {'series': [{'stats': {'total': 800}}]},
    'vllm:prefix_cache_queries': {'series': [{'stats': {'total': 1000}}]},
    'lmcache_local_hit_tokens': {'series': [{'stats': {'total': 600}}]},
    'lmcache_local_query_tokens': {'series': [{'stats': {'total': 1000}}]},
}}
print(extract_cache_stats(mock))
# Expected: server_gpu_cache_hit_rate=0.8, lmcache_local_hit_rate=0.6
"

# Run existing agentic tests to confirm refactor didn't break anything
python -m pytest utils/test_process_agentic_result.py -v
```

---

## Decisions (resolved)

| # | Decision | Resolution |
|---|---|---|
| 1 | **LMCache Docker image** | Use `lmcache/vllm-openai:nightly-2026-06-17` (latest nightly, tracks vLLM ~v0.21). Alternative: custom `FROM vllm/vllm-openai:v0.21.0` + `pip install lmcache==0.4.7` for guaranteed vLLM version parity. |
| 2 | **LMCache config** | Local-only CPU tier via `LMCACHE_LOCAL_CPU=True`. No config file, no Redis. Controlled by three env vars in the bench script. |
| 3 | **Engine scope** | vLLM only for this PR. SGLang deferred. |
| 4 | **Dashboard / UI** | Out of scope for this PR. Only the result JSON is updated; dashboard columns are a follow-up `vngcloud/InferenceX-app` change. |
| 5 | **`--server-metrics-url`** | Hardcoded as `http://localhost:${PORT}/metrics` inside the bench script. No workflow input change needed. |
