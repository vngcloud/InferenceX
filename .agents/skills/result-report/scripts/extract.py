"""Turn benchmark files into report points: throughput, latency, cache tiers, search summary."""

from __future__ import annotations

import re

PCTS = ("avg", "p50", "p90", "p95", "p99")
GRID_RE = re.compile(r"^(sweep|fixed)-isl(\d+k?)-osl(\d+k?)-cache(\d+)$")
TO_MS = {"ms": 1.0, "sec": 1000.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}


def tokens(label: str) -> int:
    return int(label[:-1]) * 1024 if label.endswith("k") else int(label)


def parse_grid(scenario: str | None) -> dict | None:
    # source: utils/serving/collect_sweep_table.py:11 (widened to fixed-*; no raise on other names)
    match = GRID_RE.match(scenario or "")
    if not match:
        return None
    kind, isl, osl, cache = match.groups()
    return {"kind": "search" if kind == "sweep" else "fixed", "isl": isl, "osl": osl, "cache": int(cache)}


def ms_stats(entry: object) -> dict:
    if not isinstance(entry, dict):
        return dict.fromkeys(PCTS)
    scale = TO_MS.get(entry.get("unit", "ms"), 1.0)
    return {k: entry[k] * scale if entry.get(k) is not None else None for k in PCTS}


def _avg(export: dict, tag: str) -> float | None:
    entry = export.get(tag)
    return entry.get("avg") if isinstance(entry, dict) else None


def _empty_metrics(source: str) -> dict:
    return {
        "throughput": dict.fromkeys(("rps", "input_tps", "output_tps", "total_tps", "output_tps_per_gpu")),
        "ttft": dict.fromkeys(PCTS), "itl": dict.fromkeys(PCTS), "e2e": dict.fromkeys(PCTS),
        "tps_per_user": None, "error_rate_pct": None, "errors": None, "client_cache_hit": None,
        "source": source,
    }


def from_aiperf(export: dict) -> dict:
    """profile_export_aiperf.json: latency in ms with avg..p99, throughputs carry avg only."""
    m = _empty_metrics("aiperf")
    m["throughput"].update(
        rps=_avg(export, "request_throughput"),
        input_tps=_avg(export, "input_token_throughput"),
        output_tps=_avg(export, "output_token_throughput"),
        total_tps=_avg(export, "total_token_throughput"),
    )
    m["ttft"] = ms_stats(export.get("time_to_first_token"))
    m["itl"] = ms_stats(export.get("inter_token_latency"))
    m["e2e"] = ms_stats(export.get("request_latency"))
    m["tps_per_user"] = _avg(export, "output_token_throughput_per_user")
    rate, errors = _avg(export, "request_error_rate"), _avg(export, "error_request_count")
    if rate is None and errors is None and _avg(export, "request_count") is not None:
        rate, errors = 0.0, 0  # aiperf omits the error metrics when no request failed
    m["error_rate_pct"] = rate
    m["errors"] = int(errors) if errors is not None else None
    hit = _avg(export, "overall_usage_prompt_cache_read_pct")
    m["client_cache_hit"] = hit / 100 if hit is not None else None
    return m


def _ms(seconds: float | None) -> float | None:
    return seconds * 1000.0 if seconds is not None else None


def from_aggregate(agg: dict) -> dict:
    """Aggregate JSON fallback: latency in seconds with mean..p95 (no p99)."""
    m = _empty_metrics("aggregate")
    rm = agg.get("request_metrics") or {}
    latency, thr = rm.get("latency") or {}, rm.get("throughput") or {}
    for key, src in (("ttft", "ttft"), ("itl", "itl"), ("e2e", "e2el")):
        d = latency.get(src) or {}
        m[key] = {"avg": _ms(d.get("mean")), "p50": _ms(d.get("p50")), "p90": _ms(d.get("p90")),
                  "p95": _ms(d.get("p95")), "p99": None}
    duration, ok = thr.get("duration_seconds"), agg.get("num_requests_successful")
    m["throughput"].update(
        rps=ok / duration if ok and duration else None,
        input_tps=(thr.get("input") or {}).get("tokens_per_second"),
        output_tps=(thr.get("output") or {}).get("tokens_per_second"),
        total_tps=(thr.get("total") or {}).get("tokens_per_second"),
    )
    return m


def from_search_aggregate(agg: dict) -> dict:
    """Search-mode aggregate (infx/results/sweep.py): winning probe's rps/tps and ttft/itl avg+p90 in ms."""
    m = _empty_metrics("search_aggregate")
    m["throughput"].update(rps=agg.get("rps"), output_tps=agg.get("output_tps"))
    m["ttft"].update(avg=agg.get("ttft_avg_ms"), p90=agg.get("ttft_p90_ms"))
    m["itl"].update(avg=agg.get("itl_avg_ms"), p90=agg.get("itl_p90_ms"))
    return m


def _cache(source: str, engine: str | None) -> dict:
    return {"overall": None, "gpu": None, "cpu": None, "external": None, "router": None,
            "tokens_by_tier": {}, "engine": engine, "source": source}


def aggregate_tiers(agg: dict) -> dict | None:
    sm = agg.get("server_metrics") or {}
    if not sm.get("present"):
        return None
    cache = sm.get("cache") or {}
    out = _cache("aggregate", sm.get("adapter"))
    tiers = dict(cache.get("cached_tokens_by_source") or {})
    overall = cache.get("overall_cache_hit_rate")
    if "total" in tiers:  # the repo adapter sums a cache_source=total series into overall
        tiers.pop("total")
        prompt_total = (sm.get("tokens") or {}).get("prompt_total")
        overall = sum(tiers.values()) / prompt_total if prompt_total else None
    router = cache.get("router_kv_hit_rate")
    out.update(
        overall=overall,
        gpu=cache.get("gpu_cache_hit_rate"),
        # vLLM copies its external (offload) rate into cpu: the tiers are not separable
        cpu=None if out["engine"] == "vllm" else cache.get("cpu_cache_hit_rate"),
        external=cache.get("external_cache_hit_rate"),
        router=router if router is not None else cache.get("frontend_cache_hit_rate"),
        tokens_by_tier=tiers,
    )
    return out


def _series(metrics: dict, name: str) -> list[dict]:
    entry = metrics.get(name)
    return [s for s in entry.get("series", []) if isinstance(s, dict)] if isinstance(entry, dict) else []


def _value(series: dict) -> float | None:
    # source: infx/results/agentic/common.py:114 (series_stat, preferred total/sum/max/avg)
    stats = series.get("stats") or {}
    for key in ("total", "sum", "max", "avg"):
        if stats.get(key) is not None:
            return float(stats[key])
    return None


def _sum(metrics: dict, name: str) -> float | None:
    values = [v for s in _series(metrics, name) if (v := _value(s)) is not None]
    return sum(values) if values else None


def _by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    # source: infx/results/agentic/common.py:186 (sum_by_label)
    out: dict[str, float] = {}
    for s in _series(metrics, name):
        key, value = (s.get("labels") or {}).get(label), _value(s)
        if key is not None and value is not None:
            out[str(key)] = out.get(str(key), 0.0) + value
    return out


def _rate(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den else None


def tiers_from_server_metrics(export: dict) -> dict:
    """Cache tiers from a (small, per-probe) server_metrics_export.json."""
    metrics = export.get("metrics") if isinstance(export.get("metrics"), dict) else {}
    names = set(metrics)
    if any(n.startswith("sglang:") for n in names):
        # source: infx/results/agentic/backends/sglang.py:53-81, minus the cache_source=total double count
        out = _cache("server_metrics", "sglang")
        prompt = _sum(metrics, "sglang:prompt_tokens")
        by = _by_label(metrics, "sglang:cached_tokens", "cache_source")
        by.pop("total", None)
        storage = [v for k, v in by.items() if k.startswith("storage_")]
        out.update(
            gpu=_rate(by.get("device"), prompt),
            cpu=_rate(by.get("host"), prompt),
            external=_rate(sum(storage), prompt) if storage else None,
            overall=_rate(sum(by.values()), prompt) if by else None,
            tokens_by_tier=by,
        )
    elif any(n.startswith("vllm:") for n in names):
        # source: infx/results/agentic/backends/vllm.py:44-95; cpu stays None (not separable from L3)
        out = _cache("server_metrics", "vllm")
        by = _by_label(metrics, "vllm:prompt_tokens_by_source", "source")
        total = sum(by.values())
        if total > 0:
            local, external = by.get("local_cache_hit"), by.get("external_kv_transfer")
            out.update(
                gpu=_rate(local, total),
                external=_rate(external, total),
                overall=((local or 0.0) + (external or 0.0)) / total,
                tokens_by_tier={k: by[k] for k in ("local_cache_hit", "external_kv_transfer") if k in by},
            )
        else:
            gpu = _rate(_sum(metrics, "vllm:prefix_cache_hits"), _sum(metrics, "vllm:prefix_cache_queries"))
            out.update(gpu=gpu, overall=gpu, external=_rate(
                _sum(metrics, "vllm:external_prefix_cache_hits"), _sum(metrics, "vllm:external_prefix_cache_queries")))
    else:
        out = _cache("server_metrics", None)
    hits = _by_label(metrics, "smg_cache_aware_hit", "result")
    if hits:
        out["router"] = _rate(hits.get("hit"), sum(hits.values()))
    return out
