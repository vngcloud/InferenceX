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

> ⚠️ **Hybrid-attention models (Qwen3.5 / Qwen3-Next class) are NOT usable with LMCache on this stack.** This was verified live (see §9). vLLM **hard-crashes at startup**; SGLang **serves the model but LMCache stays silently inert** (no offload, no benefit). The verdict above is for **full-attention** models like `Qwen/Qwen3-8B`. LMCache *does* support hybrids on a **newer stack** — **verified live**: vLLM 0.23.0 + lmcache 0.5.0 + MP connector restored a hybrid model's KV at a 99.4% external hit rate (§9.6). But **not on these pinned images** — a newer lmcache alone won't fix it (and 0.4.6 is *not* enough; you need 0.5.0). See §9.5–§9.6.

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
- `test_and_scrape.sh` — cold/warm test + per-engine metrics scrape (full-attention models)
- `verify_hybrid.sh` — reproduces the hybrid-model **incompatibility** on the *pinned* stack, both engines (§9.1–§9.4)
- `Dockerfile.vllm_mp` — `FROM vllm/vllm-openai:v0.23.0` + `pip install lmcache==0.5.0` → image `vllm-lmcache-mp:v0.23.0` (the stack that **does** support hybrids, §9.6)
- `mp_bootstrap.sh` — runs `lmcache server` + vLLM (MP connector) in one container for hybrid serving (§9.6)
- `verify_hybrid_mp.sh` — reproduces the **working** hybrid offload (cold → fresh engine → warm; ~99% external hit rate, §9.6)
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

---

## 9. Hybrid-attention models (Qwen3.5 / Qwen3-Next) — ❌ NOT supported by LMCache on this stack

**Verified live on 2026-06-24** (GPU 3, model `Qwen/Qwen3.5-4B`, same pinned images). Reproduce with `verify_hybrid.sh` (see end of section).

### 9.1 Why hybrid models break LMCache

"Hybrid attention" models interleave **linear-attention layers** (Gated DeltaNet / Mamba-style) with periodic **full-attention** layers. Qwen3.5-4B's `config.json` shows `layer_types` = a repeating pattern of **3× `linear_attention` + 1× `full_attention`** (`full_attention_interval: 4`). The two layer kinds keep **different-shaped per-layer caches**: full-attention layers store a normal KV tensor, while linear-attention layers keep a small **recurrent state** of a different shape. There is no single unified KV tensor shape.

The `lmcache 0.4.5` / `LMCacheConnectorV1` connector baked into both images **assumes one unified KV shape** (it materializes a single-shape `MemoryObj.tensor`). Hybrid-model support lives only in LMCache's much newer *multiprocess (MP) connector* — not in the 0.4.5 stack here. (Upstream context: LMCache issue [#3106](https://github.com/LMCache/LMCache/issues/3106) — multi-group / heterogeneous KV layouts.)

> **Not an engine limitation.** Both pinned images *can* load the architecture: `vllm/vllm-openai:v0.21.0` registers `Qwen3_5ForConditionalGeneration` (and `Qwen3NextForCausalLM`), and `sglang:v0.5.12` ships the `qwen3_5` / `qwen3_next` model modules. So every failure below is attributable to **LMCache**, not the serving engine.

### 9.2 vLLM v0.21.0 — hard crash, server never starts

Launching `MODEL=Qwen/Qwen3.5-4B ./run_vllm.sh` (LMCache via `--kv-transfer-config`) fails during engine-core init:

```
WARNING [vllm.py:1345] Turning off hybrid kv cache manager because `--kv-transfer-config` is set.
        ... please consider supporting hybrid kv cache manager for your connector by making sure
        your connector is a subclass of `SupportsHMA` ...
...
ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.
RuntimeError: Engine core initialization failed. See root cause above.
```

**Mechanism (chain of causation):**
1. A KV connector is present (`LMCacheConnectorV1`), but it is **not** a subclass of `SupportsHMA` (the hybrid-memory-allocator interface).
2. → vLLM therefore **force-disables its hybrid KV cache manager**.
3. → With the hybrid manager off, vLLM tries to coerce all layers' KV specs into **one unified type**.
4. → The linear-attention state spec and the full-attention KV spec **cannot be unified** → `ValueError` → engine core dies (crashes in `_initialize_kv_caches → … → _init_minimal_kv_cache_for_profiling`).

**Result:** the container exits (code 1); the HTTP server never reaches "Application startup complete". **You cannot serve a hybrid model with LMCache on vLLM v0.21.0 at all.** (Removing `--kv-transfer-config`, i.e. disabling LMCache, lets vLLM serve the model normally with its own hybrid KV manager.)

### 9.3 SGLang v0.5.12 — serves the model, but LMCache is silently inert

Unlike vLLM, `MODEL=Qwen/Qwen3.5-4B ./run_sglang.sh` (`--enable-lmcache`) **starts cleanly** ("The server is fired up") — SGLang has native multi-group / linear-attention KV handling (note `mamba usage` and `linear_attn_backend` in its logs), so its **own radix cache works**. But LMCache contributes nothing:

| Test (cold → **flush GPU radix** → warm) | Hybrid `Qwen3.5-4B` | Full-attn `Qwen3-8B` (§4.4) |
|---|---|---|
| warm prefill `#cached-token` (after flush) | **0** | 4096 |
| LMCache `store`/`retrieve` log lines | **none** | `retrieve 4096 new tokens` |
| `lmcache:*` metrics populated | **no** | no (always 0 under SGLang, §4.3) |
| `sglang:cache_hit_rate` after flushed-warm | **0.0** | 0.996 |

The flush is the discriminator: it empties SGLang's GPU radix, so a warm hit can **only** come from LMCache. For the full-attention model LMCache restored 4096 tokens; **for the hybrid model it restored 0** — LMCache never engaged. (A repeat request *without* flushing does hit — `#cached-token: 6144`, `cache_hit_rate ≈ 0.98` — but that is SGLang's **native radix**, not LMCache.)

**Result:** SGLang serves hybrid models, but enabling LMCache provides **zero incremental KV reuse** — no CPU offload, no cross-eviction restore, no metrics. It is dead weight (silently a no-op), not a crash.

### 9.4 Summary & guidance

| | **vLLM v0.21.0 + LMCache** | **SGLang v0.5.12 + LMCache** |
|---|---|---|
| Hybrid model loads? | ❌ no — engine core crashes | ✅ yes — server starts |
| Can you serve at all? | **No** (must drop LMCache) | Yes |
| Does LMCache add value? | n/a (crashes) | **No** — silently inert |
| Failure mode | loud (`ValueError`, exit 1) | **silent** (no error, no benefit) |
| Native (non-LMCache) prefix cache on hybrid? | works if LMCache removed | works (radix, `cache_hit_rate ≈ 0.98`) |

**Recommendations for an integrating agent:**
1. **Do not enable LMCache for hybrid-attention models** on this stack. On vLLM it breaks startup; on SGLang it wastes resources for no gain.
2. **Detect hybrid models up front:** read the model's `config.json` — treat it as hybrid if `layer_types` contains `linear_attention` (or `full_attention_interval` is present, or `model_type` ∈ {`qwen3_5`, `qwen3_next`, …}). Gate LMCache off when detected.
3. **The SGLang failure is silent** — there is no error to alert on. Verify LMCache is actually working via the flush test (post-flush warm `#cached-token` must be > 0) or by confirming LMCache `store`/`retrieve` log lines appear. If they don't, LMCache is inert.
4. **To use LMCache with hybrid models you must upgrade both LMCache *and* the engine** — vLLM 0.23.0 + lmcache 0.5.0 + MP connector, **verified working live in §9.6**. A newer LMCache alone is **not** enough, and the lmcache 0.4.6 *bundled* in v0.23.0 is also not enough (its MP connector isn't `SupportsHMA`) — pin **0.5.0**.

**Reproduce:**
```bash
cd /home/phucnlt2/LMCache/verify_lmcache
ENGINE=vllm   ./verify_hybrid.sh     # expect: ValueError / engine-core crash, no server
ENGINE=sglang ./verify_hybrid.sh     # expect: server up, post-flush #cached-token: 0, LMCache inert
# Default hybrid model is Qwen/Qwen3.5-4B; override with MODEL=Qwen/Qwen3-Next-... etc.
```

### 9.5 Does a newer LMCache fix this on the *old* images? — ❌ No (re-verified live 2026-06-24)

LMCache **does** now support hybrid models, via its **multiprocess (MP) connector** (`LMCacheMPConnector`), documented at [docs.lmcache.ai/mp/hybrid_models](https://docs.lmcache.ai/mp/hybrid_models.html). The HMA (hybrid memory allocator, per-group block sizes) landed in the **0.4.6 → 0.5.0** line; latest is **lmcache 0.5.0** (2026-06-23) vs. the **0.4.5** baked into both images here.

**But upgrading LMCache alone does not make the pinned images work** — because the enabling piece for vLLM is on the **vLLM side**, and it is absent from v0.21.0:

| Requirement (per the hybrid_models doc) | Provided by | In `vllm v0.21.0`? |
|---|---|---|
| `kv_connector: "LMCacheMPConnector"` registered | vLLM connector registry | ✅ **yes** (verified: it *is* in the registry) |
| `--mamba-cache-mode align` flag | **vLLM engine** (Mamba/GDN prefix-cache support, tracking [vllm#26201](https://github.com/vllm-project/vllm/issues/26201)) | ❌ **no — flag does not exist** (`vllm serve --help` has no `mamba`/`cache-mode` option) |
| `--enable-prefix-caching` on the hybrid | vLLM | ✅ yes (but useless without the above) |
| HMA / per-group block sizes + `SupportsHMA` connector | lmcache **0.5.0** (0.4.6's `LMCacheMPConnector` is **not** yet a `SupportsHMA` subclass — verified, see §9.6) | ❌ image has 0.4.5 |

The `--mamba-cache-mode align` mode is what makes vLLM checkpoint the GDN/Mamba recurrent state in fixed, block-aligned positions so LMCache can chunk and offload it. Without it the state stays one per-request tensor and the **exact `failed to convert the KV cache specs to one unified type` crash from §9.2 remains** — no LMCache version can change that, because it's a vLLM allocator behavior. That flag was added **after v0.21.0** (it is not in 0.21.0; current vLLM is 0.23.0).

**Verdict — neither pinned image works for hybrid models, even with lmcache 0.5.0:**

| Image | Has MP connector | Has `--mamba-cache-mode` | Has HMA lmcache | Works for hybrid? |
|---|---|---|---|---|
| `vllm/vllm-openai:v0.21.0` | ✅ | ❌ (vLLM too old) | ❌ (0.4.5) | ❌ **No** — must upgrade the **vLLM image** |
| `lmsysorg/sglang:v0.5.12` | n/a (SGLang path) | n/a (vLLM-only flag) | ❌ (0.4.5) | ❌ **No** — and hybrid-via-LMCache on SGLang is **not documented/validated** even upstream |

**To actually run Qwen3.5-class hybrids with LMCache offload:**
1. **vLLM:** upgrade the image to one that ships `--mamba-cache-mode` (≥ v0.22; recommend **v0.23.0**, latest) **and** install **lmcache ≥ 0.4.6** (recommend **0.5.0**). Launch with the MP connector + a standalone `lmcache server`, `--mamba-cache-mode align`, `--enable-prefix-caching`, and `--chunk-size = model block size`. This is a **different deployment shape** (standalone ZMQ cache service) than the in-process `LMCacheConnectorV1` used elsewhere in this report.
2. **SGLang:** no supported recipe today. The hybrid_models doc is vLLM-only (`--mamba-cache-mode` is a vLLM flag); SGLang serves hybrids on its **own** radix without LMCache. Don't expect LMCache offload here regardless of version.
3. **Caveat:** vLLM's `align`-mode Mamba prefix caching is **experimental upstream** with open correctness bugs (e.g. [vllm#45238](https://github.com/vllm-project/vllm/issues/45238) silent drop to 0%, [vllm#40696](https://github.com/vllm-project/vllm/issues/40696) ineffective when prompt < block size). Validate hit rate before relying on it.

> **Bottom line for the integrating agent:** "LMCache now supports hybrid models" is true — but only on a **new vLLM image (≥ 0.22) + lmcache 0.5.0 + the MP connector** (see §9.6 for the live-verified recipe). The images pinned in this report (`vllm v0.21.0`, `sglang v0.5.12`) **cannot** run hybrid models with LMCache no matter which lmcache you install, because v0.21.0 lacks the `--mamba-cache-mode` engine flag.

### 9.6 The working stack — ✅ verified live (2026-06-24): vLLM 0.23.0 + lmcache 0.5.0 + MP connector

**Built `vllm-lmcache-mp:v0.23.0` (`Dockerfile.vllm_mp` = `FROM vllm/vllm-openai:v0.23.0` + `pip install lmcache==0.5.0`) and ran `Qwen/Qwen3.5-4B` end-to-end on GPU 3.** This stack **stores and restores** the hybrid model's KV cache. Concrete evidence below.

**Why 0.5.0, not the bundled 0.4.6.** The v0.23.0 image *bundles lmcache 0.4.6*, which is **not enough**. Verified by introspection:

| lmcache version | `LMCacheMPConnector` is `SupportsHMA`? | Result on Qwen3.5-4B |
|---|---|---|
| 0.4.6 (bundled in v0.23.0) | ❌ `False` | vLLM logs *"Turning off hybrid kv cache manager because the KV connector does not support it"* → crashes later in cudagraph profiling with the **same** `ValueError: ... failed to convert the KV cache specs to one unified type` as §9.2 |
| **0.5.0** (`pip install lmcache==0.5.0`) | ✅ `True` (MRO: `… KVConnectorBase_V1, SupportsHMA …`) | vLLM **keeps** the hybrid KV manager on; connector registers **both** group types `{'mamba-page-view': 24, 'subpaged-attention-view': 8}`; serves normally |

So the gate is the connector being a `SupportsHMA` subclass — present in 0.5.0, absent in 0.4.6. (`--mamba-cache-mode align` is still required on top, for the engine-side GDN state checkpointing.)

**Recipe (per [docs.lmcache.ai/mp/hybrid_models](https://docs.lmcache.ai/mp/hybrid_models.html)):**
1. **Find the unified block size N.** Boot vLLM once with `--mamba-cache-mode align --enable-prefix-caching` and grep the log: `Setting attention block size to N tokens`. For **Qwen3.5-4B, N = 528** (the log also notes *"Padding mamba page size by 0.76% so mamba and attention page sizes are exactly equal"* — this equalization is what lets align-mode unify the specs).
2. **Start a standalone LMCache MP server** (ZMQ, default port 5555): `lmcache server --chunk-size 528 --l1-size-gb 20 --eviction-policy LRU` (chunk-size must be a multiple of N).
3. **Launch vLLM** wired to it: `--mamba-cache-mode align --enable-prefix-caching --max-num-batched-tokens 528 --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}'`. (`mp_bootstrap.sh` runs server+vLLM in one container.)

**Live evidence — cross-instance KV restore (the gold-standard test):**

| Step | `vllm:external_prefix_cache_*` | LMCache server log |
|---|---|---|
| Cold (first ever request, ~4250-tok prompt) | queries **0 → 4250**, hits **0** | `Stored 528 tokens` × 8 chunks |
| Warm — sent to a **brand-new vLLM engine** (empty GPU cache; server kept its DRAM copy) | queries **0 → 4250**, **hits → 4224 (99.4%)** | `Retrieved 4224 tokens in 0.012 s` |

vLLM's own logger on the warm request: **`Prefix cache hit rate: 0.0%, External prefix cache hit rate: 99.4%`** — the GPU tier was empty, so 99.4% of the prefix came purely from LMCache. A fresh engine that never saw the prompt could only have gotten this from the offload tier → **LMCache offload of a hybrid-attention model is confirmed working.**

**Operational gotchas discovered (matter for the integrating agent):**
- **Shared IPC namespace is mandatory** when the server and vLLM are separate containers: the MP connector hands GPU KV tensors to the server over **CUDA IPC**. Run **both** with `--ipc=host` (and `--gpus` on the server too). Without it, startup hangs forever at `Wrapping N KV cache tensors for IPC`.
- **The server pins vLLM's GPU memory.** Because it IPC-maps the engine's KV tensors, killing vLLM does **not** immediately free that GPU memory; a restarted vLLM sees reduced free VRAM. Plan capacity accordingly (or co-locate server+engine in one container and accept that restarting the container clears the cache).
- **Small KV budgets trip a Mamba-blocks check:** `max_num_seqs (256) exceeds available Mamba cache blocks (K)` → lower `--max-num-seqs` or raise `--gpu-memory-utilization`. Each decode sequence needs one Mamba cache block.
- **Still flagged experimental upstream:** vLLM logs *"Prefix caching in Mamba 'align' mode … is experimental"* (see §9.5 caveat / vllm#45238, #40696). Validate hit rate per model before relying on it.

**Reproduce:** `Dockerfile.vllm_mp`, `mp_bootstrap.sh` (single-container serve), `verify_hybrid_mp.sh` (two-container cold→fresh-engine→warm proof).

---

## 10. Migration guide — from the old integration (vLLM 0.21.0 + lmcache 0.4.5, in-process) to the new stack (vLLM 0.23.0 + lmcache 0.5.0, MP connector)

This section is for the case where you **already have the §3 integration working** — `vllm/vllm-openai:v0.21.0` with the bundled lmcache 0.4.5 and the in-process `LMCacheConnectorV1` — and want to know precisely what to change to move to the new stack. Everything below is grounded in what was verified live in §3 and §9.6.

### 10.1 First: do you even need to migrate?

The new stack is **required only for hybrid-attention models** (Qwen3.5 / Qwen3-Next class). For ordinary **full-attention** models (Qwen3-8B, Llama, etc.) the old in-process integration in §3 **already works and is simpler** — keep it.

| Your situation | Recommendation |
|---|---|
| Serving **full-attention models only** | **Stay on the §3 stack.** The MP connector buys you nothing here and adds a standalone server + CUDA-IPC complexity. Migrating full-attention traffic to MP was **not** verified in this report — do not assume it works without testing. |
| Need to serve **hybrid models** with LMCache | **Migrate** — this is the only path that works (§9.6). The old stack hard-crashes (§9.2). |
| Mixed fleet (both kinds) | Run **two deployment profiles**: keep §3 for full-attention, use §10 for hybrids. Don't try to force one connector to do both. |

> The single biggest conceptual change: the old path runs LMCache **in-process inside vLLM** (a connector library). The new path runs LMCache as a **separate `lmcache server` process** that vLLM talks to over ZMQ + CUDA IPC. This changes how you launch, how you monitor, and your GPU-memory accounting.

### 10.2 What changes — side-by-side

| Dimension | Old (§3, in-process) | New (§9.6, MP) |
|---|---|---|
| vLLM image | `vllm/vllm-openai:v0.21.0` | **`vllm/vllm-openai:v0.23.0`** (needs `--mamba-cache-mode`, absent in 0.21.0) |
| lmcache version | **0.4.5 bundled** (no install) | **0.5.0** — `pip install lmcache==0.5.0`. The 0.4.6 **bundled** in v0.23.0 is **not** enough (its MP connector isn't `SupportsHMA`, §9.6). |
| Connector | `LMCacheConnectorV1` (in-process) | `LMCacheMPConnector` (talks to standalone server) |
| `kv_connector` value | `"LMCacheConnectorV1"` | `"LMCacheMPConnector"` + `kv_connector_extra_config` pointing at the server (`lmcache.mp.host` / `lmcache.mp.port`) |
| Where LMCache runs | inside the vLLM process | **separate `lmcache server` process** (ZMQ :5555) |
| LMCache config | `LMCACHE_CONFIG_FILE=lmcache_cpu.yaml` (CPU offload, `internal_api_server_enabled`) | **server CLI flags**: `--chunk-size N --l1-size-gb … --eviction-policy LRU` |
| Extra vLLM flags | `--no-enable-prefix-caching` (for clean isolation) | `--mamba-cache-mode align`, `--enable-prefix-caching`, `--max-num-batched-tokens N` |
| `chunk_size` | 256 (arbitrary) | **must be a multiple of the unified block size N** (Qwen3.5-4B → N=528) |
| Containers | 1 | 2 (server + engine) **or** 1 via `mp_bootstrap.sh`; both need `--ipc=host` |
| GPU-mem accounting | vLLM owns all its VRAM | **server IPC-pins the engine's KV VRAM** — killing vLLM doesn't free it immediately (§9.6) |
| Primary hit metric | `lmcache:*` on `:7001` + `vllm:external_prefix_cache_*` on `:8100` | **`vllm:external_prefix_cache_*`** (verified) + server `Retrieved … tokens` log line. The `:7001` internal-API-server metrics path is the in-process config — **do not assume `lmcache:*` on `:7001` is available under MP; it was not verified.** Use the vLLM external counters as the reliable signal. |

### 10.3 Step-by-step migration (hybrid models)

1. **Build the image.** `Dockerfile.vllm_mp` = `FROM vllm/vllm-openai:v0.23.0` + `RUN pip install --no-cache-dir lmcache==0.5.0` → `docker build -f Dockerfile.vllm_mp -t vllm-lmcache-mp:v0.23.0 .`
2. **Discover the unified block size N for your model.** Boot vLLM once with `--mamba-cache-mode align --enable-prefix-caching` and grep the log for `Setting attention block size to N tokens`. (Qwen3.5-4B → **N=528**.) This N drives both `--chunk-size` and `--max-num-batched-tokens`.
3. **Start the standalone server:** `lmcache server --chunk-size N --l1-size-gb 20 --eviction-policy LRU` — on `--network host --ipc=host --gpus device=<id>` (the server needs the GPU for CUDA IPC).
4. **Launch vLLM** on the **same** `--ipc=host` and GPU, with:
   ```
   --mamba-cache-mode align --enable-prefix-caching --max-num-batched-tokens N
   --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}'
   ```
   (or run server+engine in one container via `mp_bootstrap.sh`).
5. **Verify** with `verify_hybrid_mp.sh` — cold request → recreate vLLM as a fresh engine → warm request should show `vllm:external_prefix_cache_hits_total` jump to ~99% of queries on the fresh engine, and the server log `Retrieved … tokens`.

### 10.4 Translating your existing config

- **`lmcache_cpu.yaml` → server flags.** The in-process YAML knobs map to `lmcache server` CLI: CPU-offload size → `--l1-size-gb`, chunk size → `--chunk-size`, eviction → `--eviction-policy`. There is **no `LMCACHE_CONFIG_FILE` / `internal_api_server_enabled`** in the MP launch as used here.
- **Drop `--no-enable-prefix-caching`.** The old §3 test disabled the GPU tier purely to *isolate* LMCache for measurement. The new stack **requires** `--enable-prefix-caching` (align-mode Mamba prefix caching is the mechanism). In production you want both tiers on anyway (§8).
- **Re-point your dashboards.** Keep scraping `vllm:external_prefix_cache_hits_total` / `_queries_total` — these are the verified, source-attributed LMCache signal and they work identically on both stacks. Treat the `:7001` `lmcache:*` series as **old-stack-only** until you confirm an equivalent endpoint on the MP server (`--http-port 8080` exists, but its metric surface was not validated here).

### 10.5 Risks & rollback

- **Two new operational dependencies:** a separate server process and a shared IPC namespace. If startup hangs at `Wrapping N KV cache tensors for IPC`, you forgot `--ipc=host` on one of the containers (§9.6).
- **GPU memory:** budget for the server pinning the engine's KV VRAM; a restarted engine sees less free VRAM until the server is also restarted.
- **Upstream-experimental:** vLLM still flags align-mode Mamba prefix caching as experimental (vllm#45238, #40696). **Validate the external hit rate per model** before relying on it — don't assume parity with full-attention behavior.
- **Rollback is clean:** the new stack is a separate image (`vllm-lmcache-mp:v0.23.0`) and separate launch scripts. The old §3 deployment is untouched, so reverting is just pointing traffic back at the v0.21.0 containers.
