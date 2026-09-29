"""Turn an aiperf SPEED-Bench run into a fixed-seq-len benchmark result.

SPEED-Bench recipes drive load with aiperf (``run_speedbench_aiperf`` in
``benchmarks/benchmark_lib.sh``) but land in the same fixed-seq-len pipeline
as every ``benchmark_serving.py`` arm, so the raw result must keep that
client's shape: ``infx.results.fixed_sequence`` reads ``model_id``,
``max_concurrency``, ``total_token_throughput``, ``output_throughput`` and every
``*_ms`` key. This module maps aiperf's ``profile_export_aiperf.json`` onto
those keys and attaches the speculative-decoding acceptance measured from the
server's Prometheus counters around the run.

Metric mapping (aiperf -> benchmark_serving):
  time_to_first_token -> ttft  (identical definition)
  inter_token_latency -> tpot  ((e2e - ttft) / (osl - 1), per request)
  inter_chunk_latency -> itl   (gap between streamed chunks; under speculative
                                decoding one chunk may carry several tokens,
                                exactly as in benchmark_serving's itl)
  request_latency     -> e2el
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from infx.bench_serving.benchmark_outcome import benchmark_outcome

PERCENTILES = ("p90", "p99")
LATENCY_METRICS = {
    "ttft": "time_to_first_token",
    "tpot": "inter_token_latency",
    "itl": "inter_chunk_latency",
    "e2el": "request_latency",
}
# vLLM exports monotonic counters, so acceptance over the run is a delta.
VLLM_COUNTERS = {
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "draft_events": "vllm:spec_decode_num_drafts_total",
    "drafted_tokens": "vllm:spec_decode_num_draft_tokens_total",
}
# SGLang exports only a running-average gauge; it is recorded as read after the run.
SGLANG_ACCEPT_LENGTH = "sglang:spec_accept_length"
_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample of each metric across label sets (e.g. per-engine series)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        if math.isfinite(value):
            totals[match["name"]] = totals.get(match["name"], 0.0) + value
    return totals


def snapshot(metrics_url: str, timeout: float = 10.0) -> dict[str, float]:
    """Read the speculative-decoding series this module consumes; missing ones are omitted."""
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:  # noqa: S310
        totals = parse_prometheus(response.read().decode())
    wanted = {*VLLM_COUNTERS.values(), SGLANG_ACCEPT_LENGTH}
    return {name: value for name, value in totals.items() if name in wanted}


def acceptance(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """Acceptance length AL = 1 + accepted / draft events, independent of draft depth."""
    delta = {
        key: round(after.get(name, 0.0) - before.get(name, 0.0))
        for key, name in VLLM_COUNTERS.items()
    }
    result: dict[str, Any] = {f"spec_{key}": value for key, value in delta.items()}
    result["acceptance_length"] = (
        1 + delta["accepted_tokens"] / delta["draft_events"] if delta["draft_events"] > 0 else None
    )
    result["acceptance_rate"] = (
        delta["accepted_tokens"] / delta["drafted_tokens"] if delta["drafted_tokens"] > 0 else None
    )
    if result["acceptance_length"] is None and SGLANG_ACCEPT_LENGTH in after:
        result["acceptance_length"] = after[SGLANG_ACCEPT_LENGTH]
    return result


def _stat(export: dict[str, Any], metric: str, stat: str = "avg") -> float:
    value = export.get(metric, {}).get(stat)
    if value is None:
        raise KeyError(f"aiperf export lacks {metric}.{stat}")
    return float(value)


def convert(
    export: dict[str, Any],
    *,
    model: str,
    concurrency: int,
    spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a benchmark_serving-shaped result from one aiperf profile export."""
    if export.get("was_cancelled"):
        raise ValueError("aiperf run was cancelled; refusing to publish a partial result")
    completed = int(_stat(export, "request_count"))
    failed = sum(int(entry.get("count", 0)) for entry in export.get("error_summary") or [])
    requested = completed + failed
    result: dict[str, Any] = {
        "backend": "aiperf",
        "model_id": model,
        "max_concurrency": concurrency,
        "num_prompts": requested,
        "completed": completed,
        "duration": _stat(export, "benchmark_duration"),
        "total_input_tokens": int(_stat(export, "total_isl")),
        "total_output_tokens": int(_stat(export, "total_osl")),
        "request_throughput": _stat(export, "request_throughput"),
        "output_throughput": _stat(export, "output_token_throughput"),
        "total_token_throughput": _stat(export, "total_token_throughput"),
        "benchmark_outcome": benchmark_outcome(requested, completed),
        "aiperf_version": export.get("aiperf_version"),
        "aiperf_schema_version": export.get("schema_version"),
    }
    for short, metric in LATENCY_METRICS.items():
        result[f"mean_{short}_ms"] = _stat(export, metric)
        result[f"median_{short}_ms"] = _stat(export, metric, "p50")
        result[f"std_{short}_ms"] = _stat(export, metric, "std")
        for percentile in PERCENTILES:
            result[f"{percentile}_{short}_ms"] = _stat(export, metric, percentile)
    for key in ("input_sequence_length", "output_sequence_length"):
        if key in export:
            result[f"mean_{key}"] = _stat(export, key)
    result.update(spec or {})
    result.update(metadata or {})
    return result


def _metadata_pairs(pairs: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"--meta expects key=value, got {pair!r}")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)

    snap = commands.add_parser("snapshot", help="Save the server's spec-decode series as JSON")
    snap.add_argument("--metrics-url", required=True)
    snap.add_argument("--out", type=Path, required=True)

    conv = commands.add_parser("convert", help="Write the fixed-seq-len raw result JSON")
    conv.add_argument("--aiperf-json", type=Path, required=True)
    conv.add_argument("--out", type=Path, required=True)
    conv.add_argument("--model", required=True)
    conv.add_argument("--concurrency", type=int, required=True)
    conv.add_argument("--before", type=Path, help="snapshot taken before the run")
    conv.add_argument("--after", type=Path, help="snapshot taken after the run")
    conv.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")

    args = parser.parse_args()
    if args.operation == "snapshot":
        try:
            values = snapshot(args.metrics_url)
        except OSError as exc:
            # A server without /metrics must not fail the benchmark; AL is then unknown.
            print(f"[speedbench_aiperf] metrics snapshot failed: {exc}", file=sys.stderr)
            values = {}
        args.out.write_text(json.dumps(values, indent=2))
        return 0

    spec = None
    if args.before and args.after and args.before.is_file() and args.after.is_file():
        spec = acceptance(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    result = convert(
        json.loads(args.aiperf_json.read_text()),
        model=args.model,
        concurrency=args.concurrency,
        spec=spec,
        metadata=_metadata_pairs(args.meta),
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "completed",
                    "num_prompts",
                    "output_throughput",
                    "total_token_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                    "acceptance_length",
                    "acceptance_rate",
                )
            },
            indent=2,
        )
    )
    # The request gate travels in benchmark_outcome; process_result fails the job on it,
    # after the raw result and aiperf artifacts have been kept for diagnosis.
    return 0


if __name__ == "__main__":
    sys.exit(main())
