#!/usr/bin/env python3
"""
scrape_lmcache_metrics.py — Convert Prometheus text exposition to the
server_metrics_export.json schema used by aiperf.

Usage:
    curl -sf http://localhost:8080/metrics | python3 scrape_lmcache_metrics.py

Outputs JSON to stdout:
  {"metrics": {"<name>": {"type": "<type>", "series": [{"stats": {...}}]}}}

For counter metrics: stats = {"total": <value>}
For gauge metrics:   stats = {"max": <value>, "avg": <value>, "min": <value>}
For histogram metrics: emits two synthetic gauge entries per histogram:
  "<name>_p50" and "<name>_p95" computed via linear interpolation over buckets.
  Raw _bucket / _count / _sum / _created entries are excluded from output.

A single end-of-run snapshot means max == avg == min for gauges, which is
correct — _final_value() in process_agentic_result.py prefers "total" then
"max" then "avg", so both counter and gauge lookups resolve cleanly.

Labeled metrics (e.g. lmcache_mp_l2_usage_bytes{l2_name="fs"}) emit one
series entry per label set; _final_value() sums across series, giving the
correct aggregate (e.g. total L2 bytes across all backends).
"""
import json
import re
import sys


def _interpolate_percentile(
    buckets: list[tuple[float, float]], quantile: float
) -> float:
    """Linear interpolation of a quantile from Prometheus histogram buckets.

    buckets: sorted list of (le_boundary, cumulative_count) including +Inf.
    Returns 0.0 if total count is zero.
    """
    if not buckets:
        return 0.0
    total = buckets[-1][1]  # +Inf bucket holds the total count
    if total == 0.0:
        return 0.0
    target = quantile * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if count == prev_count:
                return prev_le
            # Linear interpolation within this bucket
            frac = (target - prev_count) / (count - prev_count)
            lower = prev_le
            upper = le if le != float("inf") else prev_le * 2 or 1.0
            return lower + frac * (upper - lower)
        prev_le, prev_count = le, count
    return buckets[-1][0] if buckets[-1][0] != float("inf") else prev_le


def parse_prometheus_text(text: str) -> dict:
    metric_types: dict[str, str] = {}
    # metric_name -> list of (value, labels_dict)
    samples: dict[str, list[tuple[float, dict[str, str]]]] = {}
    # histogram base name -> list of (le_value, cumulative_count)
    # one entry per label-set combination; we aggregate across label sets
    hist_buckets: dict[str, list[tuple[float, float]]] = {}

    # Suffixes that belong to histogram internals and should be excluded
    _HIST_SUFFIXES = ("_bucket", "_count", "_sum", "_created")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"^#\s+TYPE\s+(\S+)\s+(\S+)", line)
            if m:
                metric_types[m.group(1)] = m.group(2)
            continue

        # Sample: name{labels} value [timestamp]  OR  name value [timestamp]
        m = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
            r"(?:\{([^}]*)\})?"
            r"\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)"
            r"(?:\s+\d+)?$",
            line,
        )
        if not m:
            continue

        name, labels_str, value_str = m.group(1), m.group(2) or "", m.group(3)

        labels: dict[str, str] = {}
        for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
            labels[lm.group(1)] = lm.group(2)

        # Collect histogram bucket entries separately
        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            if value_str not in ("NaN", "-Inf"):
                le_str = labels.get("le", "+Inf")
                le = float("inf") if le_str == "+Inf" else float(le_str)
                try:
                    count = float(value_str)
                except ValueError:
                    count = 0.0
                hist_buckets.setdefault(base, []).append((le, count))
            continue

        # Skip other histogram internals
        if any(name.endswith(s) for s in ("_count", "_sum", "_created")):
            base_candidate = None
            for s in ("_count", "_sum", "_created"):
                if name.endswith(s):
                    base_candidate = name[: -len(s)]
                    break
            if base_candidate and metric_types.get(base_candidate) == "histogram":
                continue

        if value_str in ("NaN", "Inf", "+Inf", "-Inf"):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue

        samples.setdefault(name, []).append((value, labels))

    metrics: dict[str, dict] = {}

    # Emit counters and gauges
    for name, sample_list in samples.items():
        mtype = metric_types.get(name, "untyped")
        # Skip raw histogram internals that slipped through
        if mtype == "histogram":
            continue
        series = []
        for value, labels in sample_list:
            stats = (
                {"total": value}
                if mtype == "counter"
                else {"max": value, "avg": value, "min": value}
            )
            entry: dict = {"stats": stats}
            if labels:
                entry["labels"] = labels
            series.append(entry)
        metrics[name] = {"type": mtype, "series": series}

    # Emit synthetic p50 / p95 gauges for each histogram
    for base, raw_buckets in hist_buckets.items():
        # Sort by le; aggregate counts across label sets by summing at each boundary
        boundary_totals: dict[float, float] = {}
        for le, count in raw_buckets:
            boundary_totals[le] = boundary_totals.get(le, 0.0) + count
        sorted_buckets = sorted(boundary_totals.items())
        # Ensure +Inf is present (use last finite bucket count if missing)
        if not any(le == float("inf") for le, _ in sorted_buckets):
            if sorted_buckets:
                sorted_buckets.append((float("inf"), sorted_buckets[-1][1]))

        for label, quantile in (("p50", 0.50), ("p95", 0.95)):
            value = _interpolate_percentile(sorted_buckets, quantile)
            key = f"{base}_{label}"
            metrics[key] = {
                "type": "gauge",
                "series": [{"stats": {"max": value, "avg": value, "min": value}}],
            }

    return {"metrics": metrics}


def main() -> None:
    text = sys.stdin.read()
    result = parse_prometheus_text(text)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
