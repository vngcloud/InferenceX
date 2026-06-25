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


def parse_prometheus_text(text: str) -> dict:
    metric_types: dict[str, str] = {}
    # metric_name -> list of (value, labels_dict)
    samples: dict[str, list[tuple[float, dict[str, str]]]] = {}

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

        if value_str in ("NaN", "Inf", "+Inf", "-Inf"):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue

        labels: dict[str, str] = {}
        for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
            labels[lm.group(1)] = lm.group(2)

        samples.setdefault(name, []).append((value, labels))

    metrics: dict[str, dict] = {}
    for name, sample_list in samples.items():
        mtype = metric_types.get(name, "untyped")
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

    return {"metrics": metrics}


def main() -> None:
    text = sys.stdin.read()
    result = parse_prometheus_text(text)
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
