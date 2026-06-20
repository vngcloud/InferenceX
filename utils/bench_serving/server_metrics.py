"""Prometheus server-metrics helpers shared between the AIPerf adapter and
the agentic result processor.

Parses AIPerf's ``server_metrics_export.json`` (schema:
``{"metrics": {<name>: {"type": ..., "series": [{"stats": {...}}, ...]}}}``).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_server_metrics(path: Path) -> dict:
    """Load AIPerf's server_metrics_export.json; return {} if missing or malformed."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def index_server_metrics(server_metrics: dict) -> dict[str, dict]:
    """Return the metrics dict keyed by metric name.

    aiperf v0.8 schema: top-level ``{"metrics": {<name>: {"type": ...,
    "series": [{"stats": {...}}, ...]}}}``. The ``metrics`` value is a
    ``dict`` keyed by metric name, NOT a list.
    """
    if not isinstance(server_metrics, dict):
        return {}
    metrics = server_metrics.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def final_value(metrics_by_name: dict, metric_name: str) -> float | None:
    """Sum total/max/avg stat across all series for a given Prometheus metric name.

    Tries ``total`` first (counters), then ``max``, then ``avg`` (gauges).
    Sums across series to aggregate multi-label sets.
    Returns None if the metric is absent or has no usable stat.
    """
    entry = metrics_by_name.get(metric_name)
    if not isinstance(entry, dict):
        return None
    series = entry.get("series") or []
    if not isinstance(series, list):
        return None
    for stats_key in ("total", "max", "avg"):
        agg = 0.0
        found = False
        for s in series:
            if not isinstance(s, dict):
                continue
            stats = s.get("stats")
            if not isinstance(stats, dict):
                continue
            v = stats.get(stats_key)
            if v is None:
                continue
            try:
                agg += float(v)
                found = True
            except (TypeError, ValueError):
                continue
        if found:
            return agg
    return None


def extract_cache_stats(server_metrics: dict) -> dict:
    """Extract cache hit rates from a server_metrics_export.json blob.

    Keys are present only when the underlying metric was found in the scrape.

    Keys:
        server_gpu_cache_hit_rate  vLLM built-in prefix cache: hits/queries
        server_cpu_cache_hit_rate  vLLM CPU-tier prefix cache: hits/queries
        lmcache_hit_rate           LMCache L1+L2 lookup hit rate (MP mode,
                                   scraped from LMCache server port 8080):
                                   lmcache_mp_lookup_hit_tokens_total /
                                   lmcache_mp_lookup_requested_tokens_total

    LMCache MP metrics are exposed by the LMCache server process at port 8080,
    not by vLLM at port 8888. The benchmark script must set
    --server-metrics-url to http://0.0.0.0:8080/metrics for lmcache_hit_rate
    to be populated. LMCacheConnectorV1 (single-process) exposes metrics
    within vLLM's process; those metric names may differ and are logged via
    the pre-run metrics snapshot in the benchmark script for discovery.
    """
    m = index_server_metrics(server_metrics)

    def _rate(hit_key: str, query_key: str) -> float | None:
        hits = final_value(m, hit_key)
        queries = final_value(m, query_key)
        if hits is not None and queries and queries > 0:
            return hits / queries
        return None

    return {
        "server_gpu_cache_hit_rate": _rate(
            "vllm:prefix_cache_hits", "vllm:prefix_cache_queries"
        ),
        "server_cpu_cache_hit_rate": _rate(
            "vllm:cpu_prefix_cache_hits", "vllm:cpu_prefix_cache_queries"
        ),
        # LMCache MP mode (LMCacheMPConnector): metrics live in the separate
        # lmcache server process. OTel counter names (Prometheus export):
        #   lmcache_mp_lookup_hit_tokens_total       numerator
        #   lmcache_mp_lookup_requested_tokens_total denominator
        "lmcache_hit_rate": _rate(
            "lmcache_mp_lookup_hit_tokens_total",
            "lmcache_mp_lookup_requested_tokens_total",
        ),
    }
