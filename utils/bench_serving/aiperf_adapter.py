#!/usr/bin/env python3
"""Run AIPerf and adapt its artifact to InferenceX benchmark JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROFILE_EXPORT = "profile_export_aiperf.json"
SEARCH_HISTORY = "search_history.json"


def detect_mode(artifact_dir: Path) -> str:
    """Return the AIPerf artifact mode for a completed run."""
    return "search" if (artifact_dir / SEARCH_HISTORY).exists() else "fixed"


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
    result = {
        "model_id": artifact["input_config"]["models"]["items"][0]["name"],
        "max_concurrency": max_concurrency,
        "total_token_throughput": artifact["total_token_throughput"]["avg"],
        "output_throughput": artifact["output_token_throughput"]["avg"],
        "mean_ttft_ms": artifact["time_to_first_token"]["avg"],
        "p99_ttft_ms": artifact["time_to_first_token"]["p99"],
        "mean_tpot_ms": artifact["inter_token_latency"]["avg"],
        "p99_tpot_ms": artifact["inter_token_latency"]["p99"],
        "mean_itl_ms": artifact["inter_token_latency"]["avg"],
        "p99_itl_ms": artifact["inter_token_latency"]["p99"],
        "mean_e2el_ms": artifact["request_latency"]["avg"],
        "p99_e2el_ms": artifact["request_latency"]["p99"],
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
        "--model",
        args.model,
        "--url",
        args.url,
        "--endpoint-type",
        args.endpoint_type,
        "--streaming",
        "--concurrency",
        str(args.concurrency),
        "--request-count",
        str(args.request_count),
        "--artifact-dir",
        str(artifact_dir),
    ]

    if args.warmup_request_count is not None:
        cmd.extend(["--warmup-request-count", str(args.warmup_request_count)])
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
    if args.isl is not None:
        cmd.extend(["--isl", str(args.isl)])
    if args.osl is not None:
        cmd.extend(["--osl", str(args.osl)])
    if args.random_seed is not None:
        cmd.extend(["--random-seed", str(args.random_seed)])

    subprocess.run(cmd, check=True)
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--request-count", required=True, type=int)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--endpoint-type", default="chat")
    parser.add_argument("--warmup-request-count", type=int)
    parser.add_argument("--server-metrics-url")
    parser.add_argument("--gpu-telemetry-url")
    parser.add_argument("--public-dataset")
    parser.add_argument("--input-file")
    parser.add_argument("--custom-dataset-type")
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--random-seed", type=int)
    args = parser.parse_args()

    if args.request_count < args.concurrency:
        parser.error("--request-count must be greater than or equal to --concurrency")

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
