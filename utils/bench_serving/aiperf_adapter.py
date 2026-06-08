#!/usr/bin/env python3
"""Run AIPerf and adapt its artifact to InferenceX benchmark JSON.

Two modes:

- fixed   : run a single concurrency, record that point (default).
- search  : run a concurrency ladder and select the winning point according to
            a ``--search-recipe`` (e.g. ``max-throughput-itl-sla``). The adapter
            drives the ladder itself (one ``aiperf profile`` per concurrency),
            so it only depends on AIPerf's single-run contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROFILE_EXPORT = "profile_export_aiperf.json"


@dataclass(frozen=True)
class SearchRecipe:
    """A rule for picking the winning point from a concurrency ladder.

    The recipe is expressed over the keys of the intermediate result produced by
    ``build_result`` so selection reuses the exact metrics InferenceX records.

    Attributes:
        objective_key: Result key to optimize (e.g. "total_token_throughput").
        objective_direction: "max" or "min".
        constraint_key: Result key constrained by the SLA, or None for no SLA.
        constraint_cmp: "le" (value <= sla) or "ge" (value >= sla).
        sla_required: Whether --sla-ms must be supplied for this recipe.
    """

    objective_key: str
    objective_direction: str
    constraint_key: str | None
    constraint_cmp: str | None
    sla_required: bool


# Registry of named recipes. Add new selection rules here.
SEARCH_RECIPES: dict[str, SearchRecipe] = {
    # Highest total token throughput among points whose p99 inter-token latency
    # stays under the SLA (ms).
    "max-throughput-itl-sla": SearchRecipe(
        objective_key="total_token_throughput",
        objective_direction="max",
        constraint_key="p99_itl_ms",
        constraint_cmp="le",
        sla_required=True,
    ),
}


def build_result(artifact: dict, max_concurrency: int) -> dict:
    """Build the intermediate schema consumed by utils/process_result.py."""
    return {
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


def extract_max_concurrency(artifact: dict) -> int:
    """Read the concurrency AIPerf actually applied in the profiling phase."""
    for phase in artifact["input_config"]["phases"]:
        if phase.get("name") == "profiling":
            return int(phase["concurrency"])
    raise ValueError("AIPerf artifact is missing the profiling phase")


def select_winner(
    results: list[tuple[int, dict]], recipe: SearchRecipe, sla_ms: float | None
) -> tuple[int, dict, bool]:
    """Pick the winning (concurrency, result) per the recipe.

    Args:
        results: List of (concurrency, build_result dict) for each ladder point.
        recipe: Selection rule.
        sla_ms: SLA threshold for the recipe's constraint (ms), or None.

    Returns:
        (winning_concurrency, winning_result, sla_met) where sla_met is False
        when no point satisfied the constraint (the best-objective point is then
        returned as a best-effort fallback).
    """
    if not results:
        raise ValueError("select_winner requires at least one ladder point")

    def meets_sla(result: dict) -> bool:
        if recipe.constraint_key is None or sla_ms is None:
            return True
        value = result[recipe.constraint_key]
        if recipe.constraint_cmp == "le":
            return value <= sla_ms
        if recipe.constraint_cmp == "ge":
            return value >= sla_ms
        raise ValueError(f"Unknown constraint comparator: {recipe.constraint_cmp}")

    feasible = [(c, r) for c, r in results if meets_sla(r)]
    pool = feasible if feasible else results
    pick = max if recipe.objective_direction == "max" else min
    winner_conc, winner_result = pick(pool, key=lambda cr: cr[1][recipe.objective_key])
    return winner_conc, winner_result, bool(feasible)


def run_aiperf(args: argparse.Namespace, concurrency: int, artifact_dir: Path) -> dict:
    """Run one `aiperf profile` at a fixed concurrency and return its artifact."""
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
        str(concurrency),
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
    return json.loads((artifact_dir / PROFILE_EXPORT).read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int)
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
    parser.add_argument(
        "--search-recipe",
        choices=sorted(SEARCH_RECIPES),
        help="Run a concurrency ladder and select the winning point by this recipe.",
    )
    parser.add_argument(
        "--search-concurrencies",
        help="Comma-separated concurrency ladder for --search-recipe (e.g. 16,24,32).",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        help="SLA threshold (ms) for the recipe's constraint metric.",
    )
    args = parser.parse_args()

    if args.search_recipe:
        recipe = SEARCH_RECIPES[args.search_recipe]
        if not args.search_concurrencies:
            parser.error("--search-recipe requires --search-concurrencies")
        # Accept both "8,16,32" and a JSON array string "[8, 16, 32]" (the form
        # produced by toJson() in the GitHub Actions workflow).
        raw = args.search_concurrencies.strip().strip("[]")
        try:
            args.search_concurrencies = [
                int(c) for c in raw.split(",") if c.strip()
            ]
        except ValueError:
            parser.error("--search-concurrencies must be a comma-separated list of ints")
        if not args.search_concurrencies:
            parser.error("--search-concurrencies must contain at least one value")
        if recipe.sla_required and args.sla_ms is None:
            parser.error(f"--search-recipe {args.search_recipe} requires --sla-ms")
        if args.request_count < max(args.search_concurrencies):
            parser.error(
                "--request-count must be >= the largest --search-concurrencies value"
            )
    else:
        if args.concurrency is None:
            parser.error("--concurrency is required unless --search-recipe is set")
        if args.request_count < args.concurrency:
            parser.error("--request-count must be greater than or equal to --concurrency")

    return args


def run_fixed(args: argparse.Namespace) -> dict:
    """Run a single concurrency and return the intermediate result."""
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    artifact = run_aiperf(args, args.concurrency, artifact_dir)
    return build_result(artifact, extract_max_concurrency(artifact))


def run_search(args: argparse.Namespace) -> dict:
    """Run the concurrency ladder and select the winning point per the recipe."""
    recipe = SEARCH_RECIPES[args.search_recipe]
    results: list[tuple[int, dict]] = []
    for concurrency in args.search_concurrencies:
        artifact_dir = args.result_dir / f"{args.result_filename}_aiperf_c{concurrency}"
        artifact = run_aiperf(args, concurrency, artifact_dir)
        results.append((concurrency, build_result(artifact, concurrency)))

    winner_conc, winner_result, sla_met = select_winner(results, recipe, args.sla_ms)
    if not sla_met:
        print(
            f"[aiperf-search] WARNING: no point met the SLA "
            f"({recipe.constraint_key} {recipe.constraint_cmp} {args.sla_ms}); "
            f"returning best-effort {recipe.objective_key} point at "
            f"concurrency={winner_conc}.",
            file=sys.stderr,
        )
    else:
        print(
            f"[aiperf-search] recipe={args.search_recipe} winner concurrency="
            f"{winner_conc} {recipe.objective_key}={winner_result[recipe.objective_key]}",
            file=sys.stderr,
        )

    winner_result["search_recipe"] = args.search_recipe
    winner_result["sla_met"] = sla_met
    return winner_result


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    result = run_search(args) if args.search_recipe else run_fixed(args)

    output_path = args.result_dir / f"{args.result_filename}.json"
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
