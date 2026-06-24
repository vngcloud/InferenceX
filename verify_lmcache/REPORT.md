# LMCache Compatibility Report — vLLM v0.21.0 & SGLang v0.5.12

**Date:** 2026-06-24  **GPU:** NVIDIA RTX 4090 (24 GB), device 3  **Model:** `Qwen/Qwen3-8B` (cached locally)
**Host:** CUDA 13.2 / driver 595.58.03, Docker 29.4.3 with nvidia runtime

---

## 1. Verdict

| Image | LMCache runs? | Out-of-the-box? | Cache-hit metrics scrapable? | Speedup observed |
|---|---|---|---|---|
| `vllm/vllm-openai:v0.21.0` | ✅ **Yes** | ✅ **Yes** — lmcache **0.4.5 is bundled** | ✅ **Yes, fully** (LMCache-native **and** vLLM-side) | **5.5×** (1.46s → 0.26s) |
| `lmsysorg/sglang:v0.5.12-cu130` | ✅ **Yes** | ⚠️ **No** — needs `pip install lmcache==0.4.5` (NOT latest) | ⚠️ **Partially** — via **SGLang's native** metric only; LMCache's own counters stay 0 | **3.7×** (0.87s → 0.23s) |

**Bottom line:** Both images run LMCache and both deliver a clear KV-reuse speedup. The differences are in packaging and in *which* metrics expose the cache hits.

---

## 2. What's inside each image

| | vLLM image | SGLang image |
|---|---|---|
| Engine version | vllm **0.21.0** | sglang **0.5.12** |
| torch / CUDA | 2.11.0 / **cu130** | 2.11.0 / **cu130** |
| Python | 3.12.13 | 3.12.3 |
| LMCache | **0.4.5 pre-installed** | **not installed** (only the connector *code* ships) |
| Integration hook | `LMCacheConnectorV1` (KV-connector API) | `--enable-lmcache` (`LMCRadixCache` / layerwise connector) |

---

## 3. vLLM v0.21.0 + LMCache — ✅ works out of the box

LMCache 0.4.5 is already in the image; no install needed. Launch (`run_vllm.sh`):

```
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
--no-enable-prefix-caching          # so any cache hit MUST come from LMCache (clean isolation)
LMCACHE_CONFIG_FILE=/config/lmcache_cpu.yaml   # CPU-offload KV cache
internal_api_server_enabled: True   # exposes LMCache's own /metrics on :7001
```

**Result** (same long prompt twice, GPU prefix-cache disabled):
- Latency **1.46s → 0.26s (5.5× faster)**.
- LMCache-native metrics on **`http://localhost:7001/metrics`**:
  - `lmcache:num_hit_tokens_total` = **2048**
  - `lmcache:num_requested_tokens_total` = 2048
  - `lmcache:retrieve_hit_rate` = **1.0**
  - `lmcache:num_stored_tokens_total` = 2048
  - plus full timing/throughput histograms (`time_to_retrieve`, `retrieve_speed`, …)
- vLLM-side view on **`http://localhost:8100/metrics`**:
  - `vllm:external_prefix_cache_hits_total` = **2048**, `..._queries_total` = 4214

➡️ **Cache hits are scrapable two independent ways.** `external_prefix_cache_*` is vLLM's view of *any* external KV connector (= LMCache here). For LMCache-specific detail, scrape the LMCache internal API server on port 7001.

---

## 4. SGLang v0.5.12 + LMCache — ✅ works, but needs the right LMCache version

### 4.1 The compatibility trap (important)
SGLang 0.5.12 ships the LMCache connector *code* but not the `lmcache` package. The docs say `pip install lmcache`, **but that pulls lmcache 0.5.0, which crashes on startup**:

```
TypeError: LMCacheLayerwiseConnector.__init__() missing 1 required positional argument: 'config_file'
```

Cause: lmcache **≥ 0.4.6** added a required `config_file` argument to the connector constructor that SGLang 0.5.12's call site does not pass. Bisected across versions:

| lmcache | connector signature | SGLang 0.5.12 |
|---|---|---|
| 0.5.0, 0.4.7, 0.4.6 | requires `config_file` | ❌ crashes |
| **0.4.5**, 0.4.3, 0.4.1, 0.3.15, 0.3.13 | no `config_file` | ✅ works |

**Fix: pin `lmcache==0.4.5`** (newest compatible — and the same version vLLM bundles). Its CUDA `c_ops` backend loads fine against torch 2.11/cu130. This is baked into `Dockerfile.sglang` → image `sglang-lmcache:v0.5.12`.

### 4.2 Result (with lmcache 0.4.5)
Launch (`run_sglang.sh`): `--enable-lmcache --enable-metrics`, `LMCACHE_USE_EXPERIMENTAL=True`, same `lmcache_cpu.yaml`.

- Starts cleanly; LMCacheEngine + LocalCPUBackend initialized (layerwise CPU offload).
- Cold → flush GPU radix → warm: **0.87s → 0.23s (3.7× faster)**.
- LMCache logs prove the path:
  - COLD: `Stored 4096 out of total 4132 tokens ... 40.9 GB/s`
  - WARM: `LMCache retrieve started: lookup=4096 ... retrieve 4096 new tokens`
  - SGLang prefill log on warm run: `#cached-token: 4096`

### 4.3 Metrics caveat (the key finding for SGLang)
- The **LMCache-native counters are registered but never incremented** through SGLang's integration: all 126 `lmcache:*` lines on `/metrics` stay **0** (`num_hit_tokens_total`, `retrieve_hit_rate`, … = 0). The standalone LMCache internal API server (`:7001`) is **not** active in this in-process path.
- The cache hit **is** captured by **SGLang's own native metrics** on the *same* port (`http://localhost:30000/metrics`):
  - `sglang:cache_hit_rate` = **0.996**
  - `sglang:cached_tokens_total{cache_source="device"}` grows with each warm hit
  - `sglang:prompt_tokens_total`

➡️ **Scrape `sglang:cache_hit_rate` / `sglang:cached_tokens_total`, not the `lmcache:*` counters, when running under SGLang.** Note this metric counts *total* cache effectiveness (GPU radix + LMCache-backed loads), so it does not isolate LMCache's contribution by itself — pair it with the cold/flush/warm method (or LMCache logs) to attribute the hit to LMCache.

### 4.4 Did we disable SGLang's prefix cache? No — and here's how we know the hit is LMCache's

Unlike the vLLM test (where we passed `--no-enable-prefix-caching` to isolate LMCache), SGLang's GPU **radix prefix cache stayed enabled**. We isolated LMCache differently: `POST /flush_cache` between the cold and warm runs, which empties the GPU radix tree so the warm prefix can only be served from LMCache.

This raises a fair question: **`sglang:cache_hit_rate` is labelled `cache_source="device"` and counts *all* prefix hits (GPU radix + LMCache-backed loads) — so the metric alone does not prove the hit came from LMCache rather than SGLang's own radix.** Attribution rests on two things instead: (a) the radix is flushed immediately before the warm run, and (b) LMCache logs an explicit `retrieve 4096 new tokens` on the warm run (SGLang only consults LMCache on a radix *miss*).

To turn that inference into proof, we ran a **control: identical SGLang, identical cold→flush→warm, but with `--enable-lmcache` OFF**:

| Run (cold → flush → warm) | warm vs cold | `sglang:cache_hit_rate` | LMCache `retrieve` log? |
|---|---|---|---|
| **With LMCache** (lmcache 0.4.5) | 0.87s → **0.23s** (3.7× faster) | **0.996** | ✅ `retrieve 4096 new tokens` |
| **Control — no LMCache** | 0.75s → **0.75s** (no speedup) | **0.0** | — (`enable_lmcache=False`) |

The control is decisive: with no LMCache, the flushed prefix is **fully recomputed** (warm = cold, hit rate 0), which proves `/flush_cache` genuinely empties the GPU radix. Same engine, same flush, same prompt — the *only* variable is LMCache. Therefore the 3.7× speedup and the `cache_hit_rate → 0.996` in the LMCache run are attributable to LMCache, not the native radix.

**Why vLLM was cleaner:** vLLM exposes `vllm:external_prefix_cache_hits_total`, a counter scoped to the *external* KV connector (= LMCache). That attributes hits to LMCache by source, so disabling the local prefix cache was enough. SGLang has no equivalent per-source counter, so the flush + control method is required to attribute a hit specifically to LMCache.

---

## 5. How to reproduce

```bash
cd /home/phucnlt2/LMCache/verify_lmcache

# vLLM (lmcache bundled)
./run_vllm.sh                       # serves :8100, lmcache metrics :7001
ENGINE=vllm   ./test_and_scrape.sh

# SGLang (needs the prebuilt image with lmcache 0.4.5)
docker build -f Dockerfile.sglang -t sglang-lmcache:v0.5.12 .   # one-time
./run_sglang.sh                     # serves :30000 (metrics merged here)
ENGINE=sglang ./test_and_scrape.sh

# Control proving the SGLang hit is LMCache's, not the native radix:
# launch the SAME image with --enable-lmcache OMITTED and no lmcache env, then
# run the same cold -> POST /flush_cache -> warm cycle. Expected: warm == cold
# (no speedup), sglang:cache_hit_rate = 0. See section 4.4.
docker run -d --name sglang-ctrl --gpus '"device=3"' --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -p 30001:30000 \
  sglang-lmcache:v0.5.12 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.80 --context-length 16384 --enable-metrics
```

**Files in this folder**
- `Dockerfile.sglang` — SGLang image + pinned `lmcache==0.4.5`
- `lmcache_cpu.yaml` — shared LMCache config (CPU-offload, layerwise, internal metrics server on)
- `run_vllm.sh` / `run_sglang.sh` — launchers (GPU 3, Qwen3-8B)
- `test_and_scrape.sh` — cold/warm test + per-engine metrics scrape
- `REPORT.md` — this file

---

## 6. Metric units & how the hit rate is calculated

The metrics fall into **two kinds**, and conflating them is the most common mistake.

### 6.1 Counters — unit = **tokens** (monotonic, cumulative since process start)

All names and label sets below were **verified live** against the running images (vLLM v0.21.0 + lmcache 0.4.5, prefix caching enabled). Scrape ports: vLLM engine `:8100`, LMCache internal API server `:7001`, SGLang `:30000`.

| Metric (exact name) | Tier / source | Endpoint | Meaning |
|---|---|---|---|
| `vllm:prefix_cache_queries_total` | vLLM **GPU/HBM** native | `:8100/metrics` | tokens queried against the on-GPU prefix cache |
| `vllm:prefix_cache_hits_total` | vLLM **GPU/HBM** native | `:8100/metrics` | tokens served from GPU cache |
| `vllm:external_prefix_cache_queries_total` | **LMCache** (external connector) | `:8100/metrics` | tokens queried against the external (LMCache) tier |
| `vllm:external_prefix_cache_hits_total` | **LMCache** (external connector) | `:8100/metrics` | tokens served from the LMCache tier |
| `lmcache:num_requested_tokens_total` | LMCache native | `:7001/metrics` | tokens LMCache was *asked to look up* |
| `lmcache:num_hit_tokens_total` | LMCache native | `:7001/metrics` | of those, how many it *found* in cache |
| `lmcache:num_stored_tokens_total` | LMCache native | `:7001/metrics` | tokens written into the cache |
| `sglang:cached_tokens_total` | SGLang aggregate (radix **+** LMCache) | `:30000/metrics` | prefix tokens served from cache |
| `sglang:prompt_tokens_total` | SGLang | `:30000/metrics` | total prompt tokens processed |

Exact label sets (so an agent can build selectors that won't silently miss series):
- `vllm:*` → `{engine="0", model_name="<model>"}`
- `lmcache:*` → `{model_name="<model>", served_model_name="<model>", role="worker", worker_id="0"}`  *(one series per worker; sum across `worker_id` for tensor-parallel > 1)*
- `sglang:*` → `cached_tokens_total{cache_source="device"}` (see §4.3)

These are Prometheus **counters**: they only increase, and are meant to be read as a *rate over time* — `rate(metric[5m])` — not as an instantaneous value. The raw number is "tokens accumulated since the server started." **Ignore the `*_created` sibling series** that the Prometheus client emits next to each counter — they are unix timestamps of when the counter was created, not data.

### 6.2 Ratios — dimensionless gauges in `[0,1]`

```
cache_hit_rate = (prompt tokens served from cache) / (total prompt tokens)
```

- `lmcache:retrieve_hit_rate` = `num_hit_tokens / num_requested_tokens`
- `sglang:cache_hit_rate`     = `cached_tokens / prompt_tokens`

**It is a fraction of TOKENS, not of requests.** The observed `sglang:cache_hit_rate = 0.996` means *99.6% of the warm prompt's tokens were a prefix match* (≈4096 of ~4132), **not** "99.6% of requests hit."

**Window caveat (SGLang):** if `cache_hit_rate` were a lifetime cumulative ratio, the cold+warm sequence would average to ~0.5 (cold = 0 hits / ~4132, warm = ~4096 / ~4132 → 4096/8264). Observing ~0.996 means the gauge reflects the **most recent logging window** (dominated by the warm request just sent), not the all-time average. So treat `sglang:cache_hit_rate` as a *recent-window* reading; use `sglang:cached_tokens_total` / `prompt_tokens_total` for a stable lifetime rate.

**Why the vLLM counts look "off" (2048 hits / 4214 queries):** that pair sums *both* runs — cold (~2048 queries, 0 hits) + warm (~2048 queries, 2048 hits). Dividing gives the *blended* cold+warm rate (~0.49), not the warm hit rate. The LMCache-native `retrieve_hit_rate=1.0` is the cleaner per-lookup signal because it is recomputed per retrieve rather than blended across the whole process lifetime.

> ⚠️ **`retrieve_hit_rate` is a 0/0 trap — do not alert on it alone.** Verified live: when prefix caching is enabled and the GPU tier serves everything, LMCache is never queried, so `num_requested_tokens_total = 0`, `num_hit_tokens_total = 0`, **yet `retrieve_hit_rate` reports `1.0`** (the `0/0` case defaults to 1.0). A perfect-looking hit rate can therefore mean "LMCache did nothing." **Always gate the ratio on `num_requested_tokens_total > 0`**, and treat the absolute `num_hit_tokens_total` (or `vllm:external_prefix_cache_hits_total`) as the real "is LMCache earning its keep" signal.

### 6.3 Which metric to scrape for good LMCache insight

| You are running… | Scrape this for LMCache insight | Why |
|---|---|---|
| **vLLM** | **`lmcache:*` on the internal API server (`:7001`)** — primary | Directly attributable to LMCache, per-retrieve. Key ones: `lmcache:retrieve_hit_rate` (hit quality), `num_hit_tokens_total` / `num_requested_tokens_total` (build your own `rate()`), `num_stored_tokens_total` (write volume), plus the `time_to_retrieve` / `retrieve_speed` histograms for throughput. |
| **vLLM** | `vllm:external_prefix_cache_hits_total` / `_queries_total` (`:8100`) — secondary | Engine-side confirmation. Scoped to the *external* connector (= LMCache), so it attributes by source. Use `rate(hits)/rate(queries)` over a window for a true blended hit rate. |
| **SGLang** | **`sglang:cached_tokens_total` / `sglang:prompt_tokens_total`** (`:30000`) | The `lmcache:*` counters stay **0** under SGLang (see §4.3) — do **not** rely on them here. These SGLang counters are the only ones that move. But they are **aggregate** (GPU radix + LMCache combined); to attribute specifically to LMCache, pair them with the flush + no-LMCache control (§4.4) or watch LMCache's `store`/`retrieve` log lines. |

**Single best signal per engine (for "is LMCache actually serving hits?"):**
- **vLLM → `rate(vllm:external_prefix_cache_hits_total[5m])`** (equivalently `rate(lmcache:num_hit_tokens_total[5m])`). Use the *absolute hit counter*, not `retrieve_hit_rate`, because the rate reads `1.0` even when nothing was requested (§6.2 trap). For hit *quality*, compute `rate(external_prefix_cache_hits_total) / rate(external_prefix_cache_queries_total)` and ignore it whenever the query rate is ~0.
- **SGLang → `rate(sglang:cached_tokens_total) / rate(sglang:prompt_tokens_total)`** over a window — but remember it is not LMCache-exclusive on its own (radix + LMCache combined; attribute via the §4.4 A/B).

---

## 7. Recommendations
1. **vLLM v0.21.0**: use as-is. For dashboards, scrape `vllm:external_prefix_cache_hits_total` / `_queries_total` from the engine port, and optionally the richer `lmcache:*` metrics from the internal API server (`internal_api_server_enabled: True`, port 7001).
2. **SGLang v0.5.12**: **do not `pip install lmcache` unpinned** — pin `lmcache==0.4.5`. For monitoring, rely on `sglang:cache_hit_rate` / `sglang:cached_tokens_total`; the `lmcache:*` counters are not populated here, so don't build alerts on them. Remember this metric is *aggregate* (GPU radix + LMCache combined, `cache_source="device"`) and does **not** attribute hits to LMCache by itself — to verify LMCache specifically, use the flush + no-LMCache control (section 4.4) or watch LMCache's `store`/`retrieve` log lines.
3. Keep `chunk_size` consistent (256 used here) and set `PYTHONHASHSEED=0` for consistent cross-process hashing in production.

---

## 8. Running with prefix caching enabled — measuring LMCache's incremental value

The §3 vLLM test passed `--no-enable-prefix-caching` to *force* all traffic through LMCache and prove the path. In production you will leave prefix caching **on**, so there are **two cache tiers**. This section explains how they interact and exactly what to scrape to see LMCache's contribution. **The conclusions here were verified live** (vLLM v0.21.0, prefix caching enabled, GPU 3).

### 8.1 The tiers form a hierarchy, not a race

The engine **always checks its GPU tier first**; LMCache is consulted only for the prefix portion the GPU tier *missed*:

```
request prefix
  ├─ 1. GPU/HBM cache (vllm prefix cache / sglang radix)  ← fastest, SMALL capacity
  │       hit → near-free; LMCache NOT consulted for that span
  ├─ 2. LMCache tier (CPU DRAM → disk → remote)           ← slower, LARGE capacity
  │       hit → PCIe load into GPU (avoids recompute)
  └─ 3. miss everywhere → full prefill (recompute)
```

**Consequence:** LMCache only adds value when the reused working set **overflows GPU capacity**, so prefixes get evicted from HBM but survive in LMCache's DRAM/disk tier. The governing condition is:

> **(working set of reused prefixes) > (GPU prefix-cache capacity)**  *and*  prefixes are revisited *after* they would have been evicted from HBM.

If everything fits in HBM, the GPU tier serves every hit and LMCache stays idle — enabling it then changes nothing (and adds a small store cost). This is why a **small benchmark shows no LMCache gain**, while **agentic coding** (long shared system prompts + large repo/file context + multi-turn history, across many concurrent/sequential sessions) is the showcase: its reuse working set vastly exceeds HBM, so the GPU tier thrashes and LMCache catches the evicted reuse.

### 8.2 Live proof of the above (vLLM, prefix caching ENABLED)

Same cold→warm shared-prefix test as §3, but **with** the GPU prefix cache on (working set easily fits in 24 GB):

| Metric (after cold+warm) | Value | Reading |
|---|---|---|
| `vllm:prefix_cache_hits_total` | **4944** | GPU tier served the entire warm hit |
| `vllm:external_prefix_cache_hits_total` | **0** | **LMCache tier never hit** |
| `lmcache:num_stored_tokens_total` | 4864 | LMCache *did* store on the cold run… |
| `lmcache:num_hit_tokens_total` | **0** | …but was never read from |
| `lmcache:retrieve_hit_rate` | `1.0` ⚠️ | misleading 0/0 (see §6.2) — nothing was requested |
| warm latency | 0.23 s | fast — but from **GPU cache, not LMCache** |

This is the entire thesis in one run: the warm request was fast, but `external_prefix_cache_hits_total = 0` proves **the speedup came from the GPU tier, not LMCache**. On a small/fits-in-HBM workload LMCache contributes nothing — exactly as predicted. To see LMCache earn hits you must overflow the GPU tier (§8.4).

### 8.3 Attribution differs by engine — this is the key operational point

| | **vLLM** | **SGLang** |
|---|---|---|
| Can metrics separate GPU-tier vs LMCache-tier hits? | ✅ **Yes** — `vllm:prefix_cache_*` (GPU) and `vllm:external_prefix_cache_*` (LMCache) are distinct counters | ❌ **No** — only the aggregate `sglang:cached_tokens_total{cache_source="device"}` exists; `lmcache:*` stay 0 (§4.3) |
| LMCache's contribution = | `rate(vllm:external_prefix_cache_hits_total)` directly | **must run an A/B** (below) |
| Need to disable native cache to measure? | No — read the external counter while both run | No, but attribution requires the A/B |

**vLLM:** just scrape `vllm:external_prefix_cache_hits_total` (or `lmcache:num_hit_tokens_total` on :7001). Non-zero and growing = LMCache is serving evicted reuse. Zero = the GPU tier is absorbing everything (LMCache not needed yet at this scale).

**SGLang A/B (required, since no per-tier counter):** run the *same large workload* twice —
- Run A: `--enable-lmcache` ON
- Run B: identical, `--enable-lmcache` OFF (GPU radix only)
- LMCache's contribution = the delta in latency / throughput / TTFT (and in aggregate `sglang:cache_hit_rate`) between A and B. On a fits-in-radix workload A≈B (LMCache idle); on an overflowing workload A pulls ahead, and that gap *is* LMCache.

### 8.4 Benchmark design checklist (to make LMCache's value visible)

1. **Overflow the GPU tier.** Use a reuse working set larger than HBM: many distinct long contexts, high concurrency, long multi-turn sessions. Agentic-coding traces are ideal. For a quick controlled demo you can also *shrink the GPU tier* (lower `--gpu-memory-utilization` on vLLM / `--mem-fraction-static` on SGLang) so eviction — and thus LMCache hits — appear sooner.
2. **Ensure temporal reuse across evictions** — prefixes revisited *after* others have pushed them out of HBM. Back-to-back identical prompts only exercise the GPU tier and will show `external_prefix_cache_hits_total = 0`.
3. **Measure prefill-bound signals:** TTFT and prefill throughput (LMCache accelerates prefill, not decode). Keep `max_tokens` small so decode doesn't dominate the wall-clock.
4. **Read the right signal:**
   - vLLM → `rate(vllm:external_prefix_cache_hits_total[5m])` > 0, and `time_to_retrieve` / `retrieve_speed` histograms on :7001.
   - SGLang → A/B latency/throughput delta + aggregate `sglang:cache_hit_rate`.
5. **Sanity gate:** never trust `lmcache:retrieve_hit_rate` unless `lmcache:num_requested_tokens_total > 0` (§6.2).
