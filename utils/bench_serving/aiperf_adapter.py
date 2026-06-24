#!/usr/bin/env python3
"""Run AIPerf and adapt its artifact to InferenceX benchmark JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROFILE_EXPORT = "profile_export_aiperf.json"
SEARCH_HISTORY = "search_history.json"

# Percentiles surfaced from AIPerf for every latency metric. AIPerf computes the
# full distribution (avg/min/max/p1..p99); InferenceX summary tables use the mean
# plus this set. p50 (median) and p99 (tail) round out the p75/p90/p95 the
# summary already renders.
_PCTL_KEYS = ("p50", "p75", "p90", "p95", "p99")


def _latency_stats(metric: dict, name: str) -> dict:
    """Map one AIPerf metric block to InferenceX <stat>_<name>_ms keys.

    process_result.py strips the `_ms` suffix and converts to seconds, and for
    `tpot` keys also derives the matching `intvty` (1000/value), so adding a
    percentile here automatically flows through to the aggregate JSON.
    """
    stats = {f"mean_{name}_ms": metric["avg"]}
    for pctl in _PCTL_KEYS:
        stats[f"{pctl}_{name}_ms"] = metric[pctl]
    return stats


def detect_mode(artifact_dir: Path) -> str:
    """Return the AIPerf artifact mode for a completed run."""
    return "search" if (artifact_dir / SEARCH_HISTORY).exists() else "fixed"

def _metric_avg(artifact: dict, metric_name: str) -> float | None:
    metric = artifact.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("avg")
    return float(value) if value is not None else None

def _whole_count(value: float | None, metric_name: str) -> int | None:
    if value is None:
        return None
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"AIPerf metric {metric_name} is not an integer count: {value}")
    return int(rounded)

def validate_request_counts(artifact: dict, expected_request_count: int) -> None:
    """Fail closed when AIPerf produced a partial or error-tainted run."""
    successful = _whole_count(_metric_avg(artifact, "request_count"), "request_count")
    errors = _whole_count(_metric_avg(artifact, "error_request_count"), "error_request_count") or 0

    if successful is None:
        raise ValueError(
            "AIPerf artifact is missing request_count; refusing to aggregate an "
            "unverifiable benchmark result."
        )

    if errors > 0:
        raise ValueError(
            f"AIPerf reported {errors} failed requests "
            f"({successful} successful, expected {expected_request_count}); "
            "refusing to aggregate partial results."
        )

    if successful != expected_request_count:
        raise ValueError(
            f"AIPerf completed {successful}/{expected_request_count} successful "
            "requests; refusing to aggregate partial results."
        )


def extract_max_concurrency(artifact: dict, search_history: dict | None, mode: str) -> int:
    """Extract the concurrency value InferenceX should record."""
    if mode == "fixed":
        for phase in artifact["input_config"]["phases"]:
            if phase.get("name") == "profiling":
                return int(phase["concurrency"])
        raise ValueError("AIPerf artifact is missing the profiling phase")

    if mode == "search":
        if search_history is None:
            raise ValueError("search mode requires search_history.json")
        return int(
            search_history["best_trials"][0]["variation_values"][
                "phases.profiling.concurrency"
            ]
        )

    raise ValueError(f"Unknown AIPerf artifact mode: {mode}")


def build_result(artifact: dict, max_concurrency: int) -> dict:
    """Build the intermediate schema consumed by utils/process_result.py."""
    # AIPerf reports a single inter-token-latency block; InferenceX records it as
    # both tpot and itl (process_result derives interactivity from the tpot keys).
    itl = artifact["inter_token_latency"]
    result = {
        "model_id": artifact["input_config"]["models"]["items"][0]["name"],
        "max_concurrency": max_concurrency,
        "total_token_throughput": artifact["total_token_throughput"]["avg"],
        "output_throughput": artifact["output_token_throughput"]["avg"],
        **_latency_stats(artifact["time_to_first_token"], "ttft"),
        **_latency_stats(itl, "tpot"),
        **_latency_stats(itl, "itl"),
        **_latency_stats(artifact["request_latency"], "e2el"),
    }

    # Benchmark duration (seconds) lets process_result.py window the power log
    # to the load-generation interval. Best-effort: omitted if AIPerf didn't
    # emit it (e.g. older artifacts).
    duration = artifact.get("benchmark_duration", {}).get("avg")
    if duration is not None:
        result["duration"] = duration

    return result


def _extract_lmcache_metrics(artifact_dir: Path) -> dict:
    """Read server_metrics_export.json and extract LMCache hit-rate fields.

    Returns a dict with keys server_lmcache_hit_rate / lmcache_hit_tokens /
    lmcache_query_tokens (all None when the file is absent or LMCache is off).
    Covers both engines:
    - vLLM: vllm:external_prefix_cache_hits_total / _queries_total
    - SGLang: sglang:cached_tokens_total / sglang:prompt_tokens_total (fallback)
    """
    result: dict = {
        "server_lmcache_hit_rate": None,
        "lmcache_hit_tokens": None,
        "lmcache_query_tokens": None,
    }
    metrics_path = artifact_dir / "server_metrics_export.json"
    if not metrics_path.exists():
        return result

    try:
        raw = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return result

    metrics: dict = raw.get("metrics") if isinstance(raw, dict) else {}
    if not isinstance(metrics, dict):
        return result

    def _final_value(metric_name: str) -> float | None:
        entry = metrics.get(metric_name)
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

    # vLLM with LMCache connector: external KV-connector hit counters
    hits = _final_value("vllm:external_prefix_cache_hits_total")
    queries = _final_value("vllm:external_prefix_cache_queries_total")
    if hits is not None and queries is not None:
        result["lmcache_hit_tokens"] = int(hits)
        result["lmcache_query_tokens"] = int(queries)
        if queries > 0:
            result["server_lmcache_hit_rate"] = hits / queries
        return result

    # SGLang with LMCache: lmcache native counters stay 0 (connector limitation);
    # fall back to SGLang's own prefix-cache counters as a proxy.
    sg_cached = _final_value("sglang:cached_tokens_total")
    sg_prompt = _final_value("sglang:prompt_tokens_total")
    if sg_cached is not None and sg_prompt is not None and sg_prompt > 0:
        result["lmcache_hit_tokens"] = int(sg_cached)
        result["lmcache_query_tokens"] = int(sg_prompt)
        result["server_lmcache_hit_rate"] = sg_cached / sg_prompt

    return result


def run_aiperf(args: argparse.Namespace) -> Path:
    """Run `aiperf profile` and return the artifact directory."""
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    cmd = [
        "aiperf",
        "profile",
        "--model",
        args.model,
        "--url",
        args.url,
        "--endpoint-type",
        args.endpoint_type,
        "--streaming",
        "--concurrency",
        str(args.concurrency),
        "--artifact-dir",
        str(artifact_dir),
    ]

    # Stop condition: a fixed request count (single-replay / Mode-1 resample) or a
    # wall-clock duration cap (duration-based smoke). At least one is always set
    # (enforced in parse_args).
    if args.request_count is not None:
        cmd.extend(["--request-count", str(args.request_count)])
    if args.benchmark_duration is not None:
        cmd.extend(["--benchmark-duration", str(args.benchmark_duration)])

    if args.warmup_request_count is not None:
        cmd.extend(["--warmup-request-count", str(args.warmup_request_count)])
    if args.num_warmup_sessions is not None:
        cmd.extend(["--num-warmup-sessions", str(args.num_warmup_sessions)])
    # Mode 1 (capacity sweep): suppress AIPerf's automatic switch to
    # fixed-schedule mode for trace datasets carrying timestamps, so the run
    # is driven purely by --concurrency back-pressure. The trace's recorded
    # inter-turn delays are stripped upstream in the launcher (aiperf 0.9.0 has
    # no CLI flag to ignore mooncake_trace delays); this flag only governs the
    # timing mode, not the per-turn think-time.
    if args.no_fixed_schedule:
        cmd.append("--no-fixed-schedule")
    if args.server_metrics_url:
        cmd.extend(["--server-metrics", args.server_metrics_url])
    if args.gpu_telemetry_url:
        cmd.extend(["--gpu-telemetry", args.gpu_telemetry_url])
    if args.public_dataset:
        cmd.extend(["--public-dataset", args.public_dataset])
    if args.input_file:
        cmd.extend(["--input-file", args.input_file])
    if args.custom_dataset_type:
        cmd.extend(["--custom-dataset-type", args.custom_dataset_type])
    # Explicit tokenizer; unset → aiperf defaults to --model (the standard flow).
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if args.isl is not None:
        cmd.extend(["--isl", str(args.isl)])
    if args.osl is not None:
        cmd.extend(["--osl", str(args.osl)])
    if args.random_seed is not None:
        cmd.extend(["--random-seed", str(args.random_seed)])
    for extra_input in args.extra_inputs:
        # Repeat the flag per value — `aiperf profile` expects one key:value per
        # --extra-inputs, not several values sharing a single flag.
        cmd.extend(["--extra-inputs", extra_input])

    # Placeholder SLA / canonical-command flags. Wired through to `aiperf profile`
    # but inert in current configs (left unset). The team computes SLA (tok/s/user,
    # goodput) offline from the retained raw artifact; these exist so a future
    # config can activate them without another plumbing change.
    if args.goodput is not None:
        cmd.extend(["--goodput", args.goodput])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.inter_turn_delay_cap_seconds is not None:
        cmd.extend(["--inter-turn-delay-cap-seconds", str(args.inter_turn_delay_cap_seconds)])
    if args.dataset_sampling_strategy is not None:
        cmd.extend(["--dataset-sampling-strategy", args.dataset_sampling_strategy])
    if args.benchmark_grace_period is not None:
        cmd.extend(["--benchmark-grace-period", str(args.benchmark_grace_period)])
    if args.workers_max is not None:
        cmd.extend(["--workers-max", str(args.workers_max)])

    subprocess.run(cmd, check=True)
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--benchmark-duration", type=float)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--endpoint-type", default="chat")
    parser.add_argument("--warmup-request-count", type=int)
    parser.add_argument("--num-warmup-sessions", type=int)
    parser.add_argument("--no-fixed-schedule", action="store_true")
    parser.add_argument("--server-metrics-url")
    parser.add_argument("--gpu-telemetry-url")
    parser.add_argument("--public-dataset")
    parser.add_argument("--input-file")
    parser.add_argument("--custom-dataset-type")
    parser.add_argument("--tokenizer")
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--extra-inputs",
        action="append",
        default=[],
        help="Additional key:value inputs to pass through to aiperf profile.",
    )
    # Placeholder SLA / canonical-command flags — wired but inert (see run_aiperf).
    parser.add_argument("--goodput")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--inter-turn-delay-cap-seconds", type=float)
    parser.add_argument("--dataset-sampling-strategy")
    parser.add_argument("--benchmark-grace-period", type=float)
    parser.add_argument("--workers-max", type=int)
    args = parser.parse_args()

    if args.request_count is None and args.benchmark_duration is None:
        parser.error("one of --request-count or --benchmark-duration is required")
    if args.request_count is not None and args.request_count < args.concurrency:
        parser.error("--request-count must be greater than or equal to --concurrency")

    return args


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_aiperf(args)

    artifact = json.loads((artifact_dir / PROFILE_EXPORT).read_text())
    # ponytail: duration mode tolerates overflow/errored turns and an unknown
    # completed-count — exact request-count validation only applies to fixed replay.
    if args.request_count is not None:
        validate_request_counts(artifact, args.request_count)
    mode = detect_mode(artifact_dir)
    search_history = None
    if mode == "search":
        search_history = json.loads((artifact_dir / SEARCH_HISTORY).read_text())

    result = build_result(
        artifact,
        extract_max_concurrency(artifact, search_history, mode),
    )
    result.update(_extract_lmcache_metrics(artifact_dir))
    output_path = args.result_dir / f"{args.result_filename}.json"
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
