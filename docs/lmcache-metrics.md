# LMCache Metrics Reference

Two connector paths exist, each with different metric namespaces:

- **V1** — in-process connector (`LMCacheConnectorV1`), lmcache ≤ 0.4.5, metrics on **port 7001**. Used for full-attention models on vLLM ≤ 0.21.0.
- **MP** — standalone server (`LMCacheMPConnector`), lmcache ≥ 0.5.0, metrics on **port 8080**. Required for hybrid-attention models (Qwen3.5/Qwen3-Next) on vLLM ≥ 0.23.0.

Two cache tiers:

- **L1** — CPU DRAM (always present when LMCache is enabled).
- **L2** — disk / remote backend (optional; null metrics when not configured).

---

## Result JSON fields

| Field | Prometheus source | Connector | Tier | Explanation | Objective measured | Impact on real metrics |
|---|---|---|---|---|---|---|
| `server_lmcache_hit_rate` | `vllm:external_prefix_cache_hits / queries` | Both | L1+L2 | Fraction of prompt tokens served from LMCache (vLLM-side view) | Cache warm-up, KV reuse | TTFT↓ |
| `lmcache_hit_tokens` | `vllm:external_prefix_cache_hits_total` | Both | L1+L2 | Absolute token count served from LMCache | Reuse volume | TTFT↓ |
| `lmcache_query_tokens` | `vllm:external_prefix_cache_queries_total` | Both | L1+L2 | Tokens that reached the LMCache tier | Whether LMCache was consulted | — |
| `lmcache_stored_tokens` | `lmcache:num_stored_tokens_total` | V1 | L1 | KV tokens written into CPU DRAM. Compare to `lmcache_hit_tokens` for write ROI — if stored ≫ hits, the working set has low temporal locality | Write cost vs reuse benefit | TTFT↑ on cold requests |
| `lmcache_retrieve_latency_ms_p50` | `lmcache:time_to_retrieve` histogram p50 | V1 | L1 | Median CPU DRAM→GPU PCIe transfer latency per retrieve call | PCIe transfer baseline | TTFT↓ |
| `lmcache_retrieve_latency_ms_p95` | same, p95 | V1 | L1 | 95th-pct retrieve latency; high tail inflates TTFT under concurrency | Tail latency under load | TTFT p95↑ |
| `lmcache_retrieve_speed_GBps_p50` | `lmcache:retrieve_speed` histogram p50 | V1 | L1 | Median PCIe read throughput during a retrieve. Low value = PCIe saturated | PCIe saturation check | TTFT↓ |
| `lmcache_retrieve_speed_GBps_p95` | same, p95 | V1 | L1 | 95th-pct throughput; low p95 = PCIe contention under concurrent requests | Bandwidth ceiling | TTFT p95↑ |
| `lmcache_mp_hit_rate` | `lmcache_mp_lookup_hit_tokens / requested` | MP | L1+L2 | MP combined (L1+L2) hit rate | Overall cache effectiveness | TTFT↓ |
| `lmcache_mp_hit_tokens` | `lmcache_mp_lookup_hit_tokens_total` | MP | L1+L2 | Absolute MP hit token count | Reuse volume | TTFT↓ |
| `lmcache_mp_query_tokens` | `lmcache_mp_lookup_requested_tokens_total` | MP | L1+L2 | Tokens queried from the MP server | Whether MP cache was reached | — |
| `lmcache_mp_l1_write_chunks` | `lmcache_mp_l1_write` | MP | L1 | KV chunks written GPU→L1 (store path). MP equivalent of `lmcache_stored_tokens`. Compare to `lmcache_mp_l1_read_chunks` for write ROI | Write volume vs reuse | TTFT↑ on cold requests |
| `lmcache_mp_l1_read_chunks` | `lmcache_mp_l1_read` | MP | L1 | KV chunks read L1→GPU (cache hits served). In steady state should exceed write chunks | Read volume / hit throughput | TTFT↓ |
| `lmcache_mp_l1_evicted_chunks` | `lmcache_mp_l1_evicted` | MP | L1 | Chunks evicted from L1 when capacity is exceeded. Non-zero means the working set overflows the DRAM budget — future requests for those chunks will miss | L1 capacity adequacy | TTFT↑ (future misses) |
| `lmcache_mp_l1_eviction_loop_ticks` | `lmcache_mp_l1_eviction_loop_ticks` | MP | L1 | Total eviction loop iterations. Denominator for eviction pressure ratio | Eviction loop activity | — |
| `lmcache_mp_l1_eviction_loop_triggered` | `lmcache_mp_l1_eviction_loop_triggered` | MP | L1 | Iterations where the eviction policy fired. `triggered / ticks` = eviction pressure ratio; ratio → 1.0 means sustained eviction pressure | Eviction pressure rate | TTFT↑ when ratio → 1 |
| `lmcache_mp_l1_read_throughput_GBps_p50` | `lmcache_mp_l1_read_throughput_GB_per_second` histogram p50 ⚠️ | MP | L1 | Median L1→GPU read bandwidth (PCIe). Low value = PCIe saturated on the hit path | PCIe read saturation | TTFT↓ |
| `lmcache_mp_l1_read_throughput_GBps_p95` | same, p95 ⚠️ | MP | L1 | 95th-pct read bandwidth; reveals contention under concurrent load | Tail bandwidth ceiling | TTFT p95↑ |
| `lmcache_mp_l1_write_throughput_GBps_p50` | `lmcache_mp_l1_write_throughput_GB_per_second` histogram p50 ⚠️ | MP | L1 | Median GPU→L1 write bandwidth (store path speed) | Store path throughput | TTFT↑ on cold requests |
| `lmcache_mp_l1_write_throughput_GBps_p95` | same, p95 ⚠️ | MP | L1 | 95th-pct write bandwidth; high tail = slow store under concurrency | Store tail latency | TTFT cold p95↑ |
| `lmcache_mp_l2_hit_rate` | `lmcache_mp_l2_prefetch_hit / lookup` | MP | L2 | L2 (disk/remote) prefetch success fraction | L2 tier health | TTFT↓ (L2 only) |
| `lmcache_mp_l2_prefetch_failures` | `lmcache_mp_l2_prefetch_failure_total` | MP | L2 | Prefetch jobs lost to eviction race or OOM. High count = L1 too small for working set, causing miss storms | L1 capacity vs working set | TTFT↑ (miss storms) |
| `lmcache_mp_l1_usage_ratio` | `lmcache_mp_l1_usage_ratio` | MP | L1 | CPU DRAM fill level (0–1). Approaching 1.0 triggers evictions | L1 saturation | TTFT↑ when → 1 |
| `lmcache_mp_l1_memory_bytes` | `lmcache_mp_l1_memory_usage_bytes` | MP | L1 | L1 DRAM bytes currently in use | Capacity planning | — |
| `lmcache_mp_active_prefetch_jobs` | `lmcache_mp_active_prefetch_jobs` | MP | L2 | In-flight L2→L1 async loads at scrape time. High sustained value = prefetch pipeline saturated | Prefetch pipeline saturation | TTFT next batch |
| `lmcache_mp_l2_load_throughput_GBps_p50` | `lmcache_mp_l2_load_throughput` histogram p50 | MP | L2 | Median L2→L1 load throughput. Low value = L2 backend (disk/network) is slow | L2 backend speed | TTFT↓ (L2 only) |
| `lmcache_mp_l2_load_throughput_GBps_p95` | same, p95 | MP | L2 | 95th-pct L2→L1 throughput; high tail = L2 I/O spikes | L2 tail latency | TTFT p95↑ (L2 only) |

⚠️ Prometheus metric name unconfirmed — field will be `null` until validated against a live `/metrics` dump from the MP server.

---

## Connector × Tier availability

| Field group | V1, L1-only | MP, L1-only | MP, L1+L2 |
|---|---|---|---|
| `server_lmcache_hit_rate` / `hit_tokens` / `query_tokens` | ✓ | ✓ | ✓ |
| `lmcache_stored_tokens` / `retrieve_latency_*` / `retrieve_speed_*` | ✓ | — | — |
| `lmcache_mp_l1_*_chunks` / eviction counters | — | ✓ | ✓ |
| `lmcache_mp_l1_*_throughput_*` (⚠️) | — | ✓ if metric exists | ✓ if metric exists |
| `lmcache_mp_l2_*` / `lmcache_mp_l2_load_throughput_*` | — | — | ✓ |

---

## Key diagnostic patterns

**Is LMCache actually serving hits?**
Check `lmcache_hit_tokens > 0` (or `lmcache_mp_hit_tokens > 0`). If zero and `lmcache_mp_l1_usage_ratio < 0.15`, the GPU KV tier absorbed everything — LMCache is correctly wired but never needed at this scale.

**Is the DRAM budget large enough?**
`lmcache_mp_l1_evicted_chunks > 0` means the working set exceeds L1. Increase `max_local_cpu_size` in `lmcache_cpu.yaml` (V1) or `--l1-size-gb` (MP server).

**Is PCIe the bottleneck (not miss rate)?**
V1: `lmcache_retrieve_latency_ms_p95 > 200` or `lmcache_retrieve_speed_GBps_p95 < 5` on PCIe 4.0.
MP: `lmcache_mp_l1_read_throughput_GBps_p95 < 5` (once confirmed).

**Is write cost justified?**
V1: `lmcache_stored_tokens / lmcache_hit_tokens` — ratio should trend toward < 1 as the cache warms.
MP: `lmcache_mp_l1_write_chunks / lmcache_mp_l1_read_chunks` — same principle.

---

## How to scrape

```bash
# MP connector — all current production benchmark scripts use this:
scrape_lmcache_server_metrics "$RESULT_DIR" 8080   # port can be omitted (default)

# V1 connector — pass explicit port:
scrape_lmcache_server_metrics "$RESULT_DIR" 7001
```

Both produce `lmcache_server_metrics.json` in `$RESULT_DIR`, which is uploaded as a CI artifact automatically (see `.github/workflows/benchmark-tmpl.yml` lines 326, 346).
