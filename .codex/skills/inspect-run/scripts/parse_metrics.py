#!/usr/bin/env python3
"""
parse_metrics.py — InferenceX benchmark run artifact parser.

Usage:
    python parse_metrics.py <scratch_dir>

<scratch_dir> must contain flat copies of the artifact files (not the original
long-named download paths, which exceed Windows MAX_PATH):
    agg.json               ← from agg_*.json
    server_metrics.json    ← from server_metrics_export.json
    profile.json           ← from profile_export_aiperf.json
    server.log             ← from server.log
    aiperf.log             ← from aiperf.log (optional)

Outputs a single JSON object to stdout. The calling agent formats this into a report.
"""
import json, sys, os, re

scratch = sys.argv[1]
result = {}


# ── agg.json ──────────────────────────────────────────────────────────────────
agg_path = os.path.join(scratch, "agg.json")
if os.path.exists(agg_path):
    with open(agg_path) as f:
        agg = json.load(f)
    if isinstance(agg, list):
        agg = agg[0] if agg else {}
    result["perf"] = {k: agg.get(k) for k in [
        "model", "framework", "precision", "conc", "tp", "hw", "isl", "osl",
        "mean_ttft", "p50_ttft", "p90_ttft", "p99_ttft",
        "mean_tpot", "p99_tpot", "mean_intvty", "p50_intvty",
        "tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu",
        "mean_e2el", "p50_e2el", "p99_e2el",
        "mean_power_w", "tok_per_watt",
    ]}


# ── profile.json ──────────────────────────────────────────────────────────────
profile_path = os.path.join(scratch, "profile.json")
if os.path.exists(profile_path):
    with open(profile_path) as f:
        prof = json.load(f)
    result["profile"] = {
        "request_count":      prof.get("request_count", {}).get("avg"),
        "benchmark_duration": prof.get("benchmark_duration", {}).get("avg"),
        "total_isl":          prof.get("total_isl", {}).get("avg"),
        "total_osl":          prof.get("total_osl", {}).get("avg"),
    }


# ── server_metrics.json + lmcache_metrics.json ────────────────────────────────
sm_path = os.path.join(scratch, "server_metrics.json")
lmc_path = os.path.join(scratch, "lmcache_metrics.json")
if os.path.exists(sm_path):
    with open(sm_path) as f:
        data = json.load(f)
    m = data.get("metrics", {})

    # Merge LMCache MP scrape if available (separate file, same schema).
    if os.path.exists(lmc_path):
        with open(lmc_path) as f:
            lmc_data = json.load(f)
        m = {**m, **lmc_data.get("metrics", {})}

    def series_stats(key):
        entry = m.get(key, {})
        series = entry.get("series", [{}])
        # Sum "total" across all series (handles multi-label counters correctly).
        # For gauges use max of first series — adequate for a single-snapshot scrape.
        totals = [s.get("stats", {}).get("total") for s in series if s.get("stats", {}).get("total") is not None]
        if totals:
            return {"total": sum(totals)}
        return series[0].get("stats", {}) if series else {}

    ext_hits    = series_stats("vllm:external_prefix_cache_hits").get("total", 0)
    ext_queries = series_stats("vllm:external_prefix_cache_queries").get("total", 0)
    gpu_hits    = series_stats("vllm:prefix_cache_hits").get("total", 0)
    gpu_queries = series_stats("vllm:prefix_cache_queries").get("total", 0)
    cached_tok  = series_stats("vllm:prompt_tokens_cached").get("total", 0)
    kv_st       = m.get("vllm:kv_cache_usage_perc", {}).get("series", [{}])[0].get("stats", {})

    # LMCache MP internal metrics
    mp_hit_tok  = series_stats("lmcache_mp_lookup_hit_tokens_total").get("total", 0)
    mp_req_tok  = series_stats("lmcache_mp_lookup_requested_tokens_total").get("total", 0)
    l2_hits     = series_stats("lmcache_mp_l2_prefetch_hit_total").get("total", 0)
    l2_lookups  = series_stats("lmcache_mp_l2_prefetch_lookup_total").get("total", 0)
    l2_fail     = series_stats("lmcache_mp_l2_prefetch_failure_total").get("total", 0)
    l1_ratio_st = m.get("lmcache_mp_l1_usage_ratio", {}).get("series", [{}])[0].get("stats", {})
    l1_bytes_st = m.get("lmcache_mp_l1_memory_usage_bytes", {}).get("series", [{}])[0].get("stats", {})
    pf_jobs_st  = m.get("lmcache_mp_active_prefetch_jobs", {}).get("series", [{}])[0].get("stats", {})

    result["cache"] = {
        "ext_hits":             ext_hits,
        "ext_queries":          ext_queries,
        "ext_hit_rate_pct":     round(ext_hits / ext_queries * 100, 2) if ext_queries else 0.0,
        "gpu_hits":             gpu_hits,
        "gpu_queries":          gpu_queries,
        "gpu_hit_rate_pct":     round(gpu_hits / gpu_queries * 100, 2) if gpu_queries else 0.0,
        "prompt_tokens_cached": cached_tok,
        "kv_usage_avg_pct":     round(kv_st.get("avg", 0) * 100, 2),
        "kv_usage_max_pct":     round(kv_st.get("max", 0) * 100, 2),
        "kv_usage_min_pct":     round(kv_st.get("min", 0) * 100, 2),
        # LMCache MP internal view (None fields absent when scrape not available)
        "mp_hit_tokens":        mp_hit_tok or None,
        "mp_query_tokens":      mp_req_tok or None,
        "mp_hit_rate_pct":      round(mp_hit_tok / mp_req_tok * 100, 2) if mp_req_tok else None,
        "mp_l2_hit_rate_pct":   round(l2_hits / l2_lookups * 100, 2) if l2_lookups else None,
        "mp_l2_prefetch_fails": l2_fail or None,
        "mp_l1_usage_ratio":    l1_ratio_st.get("max"),
        "mp_l1_memory_gb":      round(l1_bytes_st["max"] / 1e9, 3) if l1_bytes_st.get("max") else None,
        "mp_active_prefetch_jobs": pf_jobs_st.get("max"),
    }


# ── server.log ────────────────────────────────────────────────────────────────
slog_path = os.path.join(scratch, "server.log")
if os.path.exists(slog_path):
    with open(slog_path, encoding="utf-8", errors="replace") as f:
        log_text = f.read()

    # Initialization checks
    bsm = re.search(r"Setting attention block size to (\d+) tokens", log_text)
    vm  = re.search(r"LMCache v(\S+)", log_text)
    init = {
        "block_size_align":       int(bsm.group(1)) if bsm else None,
        "lmcache_mp_connector":   "LMCacheMPConnector" in log_text and
                                  "Using external LMCacheMPConnector" in log_text,
        "lmcache_v1_connector":   "LMCacheConnectorV1 initialized" in log_text,
        "lmcache_version":        vm.group(1) if vm else None,
        "heartbeat_running":      "lmcache-heartbeat entering main loop" in log_text,
        "hybrid_kv_turned_off":   "Turning off hybrid kv cache manager" in log_text,
        "connector_crash":        "failed to convert the KV cache specs" in log_text,
    }

    # Runtime 10-second log stats
    running_counts = [int(x) for x in re.findall(r"Running: (\d+) reqs", log_text)]
    waiting_counts = [int(x) for x in re.findall(r"Waiting: (\d+) reqs", log_text)]
    ext_rates      = [float(x) for x in re.findall(r"External prefix cache hit rate: ([\d.]+)%", log_text)]
    gpu_rates      = [float(x) for x in re.findall(r"(?<!External )Prefix cache hit rate: ([\d.]+)%", log_text)]

    def _stats(lst):
        if not lst:
            return {}
        return {"avg": round(sum(lst)/len(lst), 1), "max": max(lst), "min": min(lst)}

    runtime = {
        "running": _stats(running_counts),
        "waiting": _stats(waiting_counts),
        "ext_hit_rate_log_pct":  {"first": ext_rates[0] if ext_rates else None,
                                   "last":  ext_rates[-1] if ext_rates else None},
        "gpu_hit_rate_log_pct":  {"first": gpu_rates[0] if gpu_rates else None,
                                   "last":  gpu_rates[-1] if gpu_rates else None},
        "sample_count": len(running_counts),
    }

    result["server_log"] = {"init": init, "runtime": runtime}


# ── aiperf.log ────────────────────────────────────────────────────────────────
aiperf_path = os.path.join(scratch, "aiperf.log")
if os.path.exists(aiperf_path):
    with open(aiperf_path, encoding="utf-8", errors="replace") as f:
        aiperf_text = f.read()

    wm = re.search(r"Phase warmup complete.*?elapsed=([\d.]+)s", aiperf_text)
    pm = re.search(
        r"Phase profiling sending complete \| sent=(\d+), completed=(\d+), in_flight=(\d+)",
        aiperf_text)
    em = re.search(r"errors=(\d+)", aiperf_text)

    aiperf_info = {
        "warmup_elapsed_s":           float(wm.group(1)) if wm else None,
        "profiling_sent":             int(pm.group(1))   if pm else None,
        "profiling_completed":        int(pm.group(2))   if pm else None,
        "profiling_in_flight_at_end": int(pm.group(3))   if pm else None,
        "errors":                     int(em.group(1))   if em else 0,
        "timeout_triggered":          "timeout_triggered=True" in aiperf_text,
    }
    result["aiperf"] = aiperf_info


print(json.dumps(result, indent=2))
