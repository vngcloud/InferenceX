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


def _latency_stats(metric: dict | None, name: str) -> dict:
    """Map one AIPerf metric block to InferenceX <stat>_<name>_ms keys.

    process_result.py strips the `_ms` suffix and converts to seconds, and for
    `tpot` keys also derives the matching `intvty` (1000/value), so adding a
    percentile here automatically flows through to the aggregate JSON.

    AIPerf marks every metric block as optional (`JsonMetricResult | None`) and
    omits it from the exported JSON when it couldn't be computed -- e.g. too
    few completed requests in the window to derive inter-token latency. Treat
    that as "no data for this stat" rather than a crash.
    """
    if metric is None:
        return {}
    stats = {f"mean_{name}_ms": metric["avg"]}
    for pctl in _PCTL_KEYS:
        stats[f"{pctl}_{name}_ms"] = metric[pctl]
    return stats


def detect_mode(artifact_dir: Path) -> str:
    """Return the AIPerf artifact mode for a completed run."""
    return "search" if (artifact_dir / SEARCH_HISTORY).exists() else "fixed"

def extract_max_concurrency(artifact: dict, search_history: dict | None, mode: str) -> int:
    """Extract the concurrency value InferenceX should record."""
    if mode == "fixed":
        input_config = artifact["input_config"]
        if "phases" in input_config:
            for phase in input_config["phases"]:
                if phase.get("name") == "profiling":
                    return int(phase["concurrency"])
            raise ValueError("AIPerf artifact is missing the profiling phase")

        concurrency = input_config.get("loadgen", {}).get("concurrency")
        if concurrency is not None:
            return int(concurrency)
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
    # Every block below is optional in AIPerf's export schema and omitted from
    # the JSON (not null) when it couldn't be computed -- degrade gracefully
    # rather than KeyError on a legitimately data-sparse run (e.g. very short
    # duration, very few completed requests).
    itl = artifact.get("inter_token_latency")
    input_config = artifact["input_config"]
    model_id = input_config.get("models", {}).get("items", [{}])[0].get("name")
    if model_id is None:
        model_id = input_config["endpoint"]["model_names"][0]
    total_throughput = artifact.get("total_token_throughput")
    output_throughput = artifact.get("output_token_throughput")
    result = {
        "model_id": model_id,
        "max_concurrency": max_concurrency,
        **({"total_token_throughput": total_throughput["avg"]} if total_throughput else {}),
        **({"output_throughput": output_throughput["avg"]} if output_throughput else {}),
        **_latency_stats(artifact.get("time_to_first_token"), "ttft"),
        **_latency_stats(itl, "tpot"),
        **_latency_stats(itl, "itl"),
        **_latency_stats(artifact.get("request_latency"), "e2el"),
    }

    # Benchmark duration (seconds) lets process_result.py window the power log
    # to the load-generation interval. Best-effort: omitted if AIPerf didn't
    # emit it (e.g. older artifacts).
    duration = artifact.get("benchmark_duration", {}).get("avg")
    if duration is not None:
        result["duration"] = duration

    return result


def run_aiperf(args: argparse.Namespace) -> Path:
    """Run `aiperf profile` and return the artifact directory."""
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    cmd = [
        "aiperf",
        "profile",
    ]
    if args.scenario:
        cmd.extend(["--scenario", args.scenario])
    cmd.extend([
        "--model",
        args.model,
        "--url",
        args.url,
    ])
    if args.endpoint:
        cmd.extend(["--endpoint", args.endpoint])
    cmd.extend([
        "--endpoint-type",
        args.endpoint_type,
        "--streaming",
        "--concurrency",
        str(args.concurrency),
        "--output-artifact-dir" if args.scenario else "--artifact-dir",
        str(artifact_dir),
    ])
    if (max_workers := getattr(args, "max_workers", None)) is not None:
        cmd.extend(["--workers-max", str(max_workers)])
    agentx_weka = args.scenario == "inferencex-agentx-mvp" and (
        args.custom_dataset_type == "weka_trace"
        or (args.public_dataset or "").startswith("semianalysis_cc_traces_weka")
    )

    cmd.extend(["--benchmark-duration", str(args.benchmark_duration)])
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
    if args.failed_request_threshold is not None:
        cmd.extend(["--failed-request-threshold", str(args.failed_request_threshold)])
    if args.trajectory_start_min_ratio is not None:
        cmd.extend(["--trajectory-start-min-ratio", str(args.trajectory_start_min_ratio)])
    if args.trajectory_start_max_ratio is not None:
        cmd.extend(["--trajectory-start-max-ratio", str(args.trajectory_start_max_ratio)])
    if args.use_server_token_count:
        cmd.append("--use-server-token-count")
    if args.tokenizer_trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    if args.num_dataset_entries is not None:
        cmd.extend(["--num-dataset-entries", str(args.num_dataset_entries)])
    if args.slice_duration is not None:
        cmd.extend(["--slice-duration", str(args.slice_duration)])
    if args.unsafe_override:
        cmd.append("--unsafe-override")

    subprocess.run(cmd, check=True)
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--benchmark-duration", type=float, required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--endpoint-type", default="chat")
    parser.add_argument("--scenario")
    parser.add_argument("--endpoint")
    parser.add_argument("--server-metrics-url")
    parser.add_argument("--gpu-telemetry-url")
    parser.add_argument("--public-dataset")
    parser.add_argument("--input-file")
    parser.add_argument("--custom-dataset-type")
    parser.add_argument("--tokenizer")
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--failed-request-threshold", type=float)
    parser.add_argument("--trajectory-start-min-ratio", type=float)
    parser.add_argument("--trajectory-start-max-ratio", type=float)
    parser.add_argument("--use-server-token-count", action="store_true")
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    parser.add_argument("--num-dataset-entries", type=int)
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "Passed through as aiperf's --workers-max. AIPerf's default "
            "auto-scales worker count with CPU count, and each worker "
            "appears to hold its own copy of the reconstructed dataset -- on "
            "a memory-constrained client host this can OOM-kill the whole "
            "process (observed: a 16-core client host, 31GB RAM, killed by "
            "the OOM killer after growing to ~31.6GB RSS running a 100-entry "
            "semianalysis_cc_traces_weka sweep). Cap this on such hosts."
        ),
    )
    parser.add_argument("--slice-duration", type=float)
    parser.add_argument("--unsafe-override", action="store_true")
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_aiperf(args)

    artifact = json.loads((artifact_dir / PROFILE_EXPORT).read_text())
    mode = detect_mode(artifact_dir)
    search_history = None
    if mode == "search":
        search_history = json.loads((artifact_dir / SEARCH_HISTORY).read_text())

    result = build_result(
        artifact,
        extract_max_concurrency(artifact, search_history, mode),
    )
    output_path = args.result_dir / f"{args.result_filename}.json"
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
