#!/usr/bin/env python3
"""Run AIPerf and adapt its artifact to InferenceX benchmark JSON.

Two modes:

- fixed   : run a single concurrency, record that point (default).
- search  : delegate to AIPerf's native Bayesian-Optimization search recipes
            (``--search-recipe``, e.g. ``max-throughput-itl-sla``). AIPerf
            itself chooses which concurrency points to probe within
            ``[--concurrency-min, --concurrency-max]``, enforces the p95 SLA via
            its built-in ``SLAFilter`` (stat defaults to p95), and writes the
            optimisation trajectory to ``search_history.json``. The adapter then
            reads the winning trial and maps that point's per-variation
            ``profile_export_aiperf.json`` into the InferenceX schema.

Native search requires AIPerf >= 0.9.0 (the version InferenceX installs from
PyPI), which ships the ``search_recipes`` / adaptive-sweep machinery.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROFILE_EXPORT = "profile_export_aiperf.json"
SEARCH_HISTORY = "search_history.json"


@dataclass(frozen=True)
class NativeRecipe:
    """An AIPerf native BO search recipe we expose through the adapter.

    The adapter is a thin pass-through: AIPerf owns the optimisation. This
    record only declares which SLA flags the recipe accepts so the adapter can
    forward the InferenceX ``--sla-ms`` / ``--ttft-sla-ms`` values to the right
    AIPerf flags and fail early if a required one is missing.

    Attributes:
        objective: Human-readable objective (for the log line only).
        accepts_itl: Forward ``--sla-ms`` as ``--itl-sla-ms`` when set.
        accepts_ttft: Forward ``--ttft-sla-ms`` as ``--ttft-sla-ms`` when set.
        require_itl: ``--sla-ms`` is mandatory for this recipe.
        require_ttft: ``--ttft-sla-ms`` is mandatory for this recipe.
        require_any: At least one accepted SLA flag must be supplied.
    """

    objective: str
    accepts_itl: bool
    accepts_ttft: bool
    require_itl: bool = False
    require_ttft: bool = False
    require_any: bool = False


# Native AIPerf BO recipes we surface. Names match AIPerf's own recipe names
# (aiperf.search_recipes.builtins); the SLA stat defaults to p95 inside AIPerf's
# SLAFilter, matching the MaaS proposals (MEP-0001/0002).
SEARCH_RECIPES: dict[str, NativeRecipe] = {
    # Maximise total token throughput subject to a p95 ITL/TPOT SLA.
    "max-throughput-itl-sla": NativeRecipe(
        objective="max total_token_throughput",
        accepts_itl=True,
        accepts_ttft=False,
        require_itl=True,
    ),
    # Maximise total token throughput subject to a p95 TTFT SLA.
    "max-throughput-ttft-sla": NativeRecipe(
        objective="max total_token_throughput",
        accepts_itl=False,
        accepts_ttft=True,
        require_ttft=True,
    ),
    # Maximise sustainable concurrency subject to a p95 SLA. Composes whichever
    # of the ITL / TTFT filters are supplied (at least one required).
    "max-concurrency-under-sla": NativeRecipe(
        objective="max concurrency",
        accepts_itl=True,
        accepts_ttft=True,
        require_any=True,
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
        "p95_ttft_ms": artifact["time_to_first_token"]["p95"],
        "p99_ttft_ms": artifact["time_to_first_token"]["p99"],
        "mean_tpot_ms": artifact["inter_token_latency"]["avg"],
        "p95_tpot_ms": artifact["inter_token_latency"]["p95"],
        "p99_tpot_ms": artifact["inter_token_latency"]["p99"],
        "mean_itl_ms": artifact["inter_token_latency"]["avg"],
        "p95_itl_ms": artifact["inter_token_latency"]["p95"],
        "p99_itl_ms": artifact["inter_token_latency"]["p99"],
        "mean_e2el_ms": artifact["request_latency"]["avg"],
        "p95_e2el_ms": artifact["request_latency"]["p95"],
        "p99_e2el_ms": artifact["request_latency"]["p99"],
    }


def extract_max_concurrency(artifact: dict) -> int:
    """Read the concurrency AIPerf actually applied in the profiling phase."""
    for phase in artifact["input_config"]["phases"]:
        if phase.get("name") == "profiling":
            return int(phase["concurrency"])
    raise ValueError("AIPerf artifact is missing the profiling phase")


def common_aiperf_args(args: argparse.Namespace) -> list[str]:
    """Optional `aiperf profile` flags shared by the fixed and search paths."""
    cmd: list[str] = []
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
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
    if args.benchmark_duration is not None:
        cmd.extend(["--benchmark-duration", str(args.benchmark_duration)])
        if args.benchmark_grace_period is not None:
            cmd.extend(["--benchmark-grace-period", str(args.benchmark_grace_period)])
    return cmd


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
        "--artifact-dir",
        str(artifact_dir),
        *common_aiperf_args(args),
    ]
    if args.request_count is not None:
        cmd.extend(["--request-count", str(args.request_count)])
    subprocess.run(cmd, check=True)
    return json.loads((artifact_dir / PROFILE_EXPORT).read_text())


def build_search_command(args: argparse.Namespace, artifact_dir: Path) -> list[str]:
    """Assemble the native `aiperf profile --search-recipe ...` command."""
    recipe = SEARCH_RECIPES[args.search_recipe]
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
        "--search-recipe",
        args.search_recipe,
        "--concurrency-min",
        str(args.concurrency_min),
        "--concurrency-max",
        str(args.concurrency_max),
        "--artifact-dir",
        str(artifact_dir),
        *common_aiperf_args(args),
    ]
    if args.request_count is not None:
        cmd.extend(["--request-count", str(args.request_count)])
    # InferenceX's --sla-ms is the p95 ITL/TPOT threshold; forward it as AIPerf's
    # --itl-sla-ms (alias of --tpot-sla-ms). --ttft-sla-ms passes straight
    # through. AIPerf's SLAFilter applies these at p95 by default.
    if recipe.accepts_itl and args.sla_ms is not None:
        cmd.extend(["--itl-sla-ms", str(args.sla_ms)])
    if recipe.accepts_ttft and args.ttft_sla_ms is not None:
        cmd.extend(["--ttft-sla-ms", str(args.ttft_sla_ms)])
    if args.search_max_iterations is not None:
        cmd.extend(["--search-max-iterations", str(args.search_max_iterations)])
    return cmd


def winner_from_history(history: dict) -> tuple[int, int, bool]:
    """Read the winning point from search_history.json.

    ``best_trials`` is feasibility-first: for a single-objective recipe it holds
    the single argmax/argmin. ``variation_values`` keys are dotted parameter
    paths (e.g. ``phases.profiling.concurrency``); we pick the one whose leaf is
    ``concurrency``. ``iteration_idx`` is the BO iteration number — it, NOT the
    concurrency, identifies the winner's on-disk artifact directory
    (``search_iter_<iteration_idx:04d>/``; see ``winner_profile_export``).
    ``feasible_count == 0`` signals AIPerf fell back to the full (infeasible)
    pool because no probed point met the SLA.

    Returns:
        (winner_concurrency, iteration_idx, sla_met).
    """
    best = history.get("best_trials")
    if not best:
        raise ValueError("search_history.json has no best_trials")
    top = best[0]
    values = top["variation_values"]
    concurrency = None
    for key, value in values.items():
        if key == "concurrency" or key.endswith(".concurrency"):
            concurrency = int(value)
            break
    if concurrency is None:
        raise ValueError(f"no concurrency dimension in variation_values: {values}")
    if "iteration_idx" not in top:
        raise ValueError(f"best_trials[0] has no iteration_idx: {top}")
    iteration_idx = int(top["iteration_idx"])
    sla_met = bool(top.get("feasible", False)) and top.get("feasible_count", 0) > 0
    return concurrency, iteration_idx, sla_met


def winner_profile_export(artifact_dir: Path, iteration_idx: int) -> dict:
    """Load the profile export for the winning BO iteration.

    AIPerf's adaptive (BO) search lays out each iteration as
    ``<artifact>/search_iter_<NNNN>/profile_runs/run_<MMMM>/`` — the BO
    iteration is the cell identity, NOT the proposed concurrency (verified
    against aiperf 0.9.0: ``orchestrator.py`` builds
    ``base/<variation.label>/profile_runs/run_<trial+1:04d>`` and the search
    planners set ``variation.label = f"search_iter_{iteration_idx:04d}"``).
    There is one run per iteration here, so we glob ``run_*`` rather than
    hard-coding ``run_0001``. We prefer the single-run
    ``profile_export_aiperf.json`` (the nested-key schema ``build_result``
    expects), not the ``aggregate/`` rollup.
    """
    cell = artifact_dir / f"search_iter_{iteration_idx:04d}"
    runs = cell / "profile_runs"
    for candidate in sorted(runs.glob(f"run_*/{PROFILE_EXPORT}")):
        return json.loads(candidate.read_text())
    # Tolerate a flattened layout (no profile_runs wrapper) just in case.
    direct = cell / PROFILE_EXPORT
    if direct.exists():
        return json.loads(direct.read_text())
    raise FileNotFoundError(
        f"no {PROFILE_EXPORT} for winning BO iteration {iteration_idx} under {cell}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--request-count",
        type=int,
        help="Stop each (probed) run after this many requests. Mutually "
        "exclusive with --benchmark-duration; exactly one is required.",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=float,
        help="Measure each (probed) run for this many seconds instead of a "
        "fixed request count (AIPerf --benchmark-duration). Mutually exclusive "
        "with --request-count. In --search-recipe mode the duration applies "
        "per BO-probed concurrency, so every point gets an equal measurement "
        "window (request count would shrink the window as concurrency rises).",
    )
    parser.add_argument(
        "--benchmark-grace-period",
        type=float,
        help="Seconds to keep collecting in-flight responses after "
        "--benchmark-duration ends (AIPerf --benchmark-grace-period). Must "
        "exceed one request's decode time (~osl x target-ITL) or in-flight "
        "requests are truncated. Ignored unless --benchmark-duration is set.",
    )
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
        help="Delegate to this AIPerf native BO search recipe over a "
        "[--concurrency-min, --concurrency-max] range.",
    )
    parser.add_argument(
        "--concurrency-min",
        type=int,
        help="Lower bound of the BO concurrency search range (--search-recipe).",
    )
    parser.add_argument(
        "--concurrency-max",
        type=int,
        help="Upper bound of the BO concurrency search range (--search-recipe).",
    )
    parser.add_argument(
        "--search-max-iterations",
        type=int,
        help="Cap on BO iterations (trials) for --search-recipe.",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        help="p95 ITL/TPOT SLA threshold (ms); forwarded to AIPerf --itl-sla-ms.",
    )
    parser.add_argument(
        "--ttft-sla-ms",
        type=float,
        help="p95 TTFT SLA threshold (ms); forwarded to AIPerf --ttft-sla-ms.",
    )
    args = parser.parse_args()

    # Load terminator: exactly one of a fixed request count or a wall-clock
    # duration. AIPerf accepts either as the profiling-phase stop condition.
    if (args.request_count is None) == (args.benchmark_duration is None):
        parser.error(
            "exactly one of --request-count or --benchmark-duration is required"
        )

    if args.search_recipe:
        recipe = SEARCH_RECIPES[args.search_recipe]
        if args.concurrency_min is None or args.concurrency_max is None:
            parser.error(
                "--search-recipe requires --concurrency-min and --concurrency-max"
            )
        if args.concurrency_min >= args.concurrency_max:
            parser.error("--concurrency-min must be < --concurrency-max")
        if recipe.require_itl and args.sla_ms is None:
            parser.error(f"--search-recipe {args.search_recipe} requires --sla-ms")
        if recipe.require_ttft and args.ttft_sla_ms is None:
            parser.error(f"--search-recipe {args.search_recipe} requires --ttft-sla-ms")
        if recipe.require_any and args.sla_ms is None and args.ttft_sla_ms is None:
            parser.error(
                f"--search-recipe {args.search_recipe} requires at least one of "
                "--sla-ms / --ttft-sla-ms"
            )
        # request-count must cover the largest concurrency the BO may probe;
        # duration-mode has no such constraint (the window is time-bounded).
        if args.request_count is not None and args.request_count < args.concurrency_max:
            parser.error("--request-count must be >= --concurrency-max")
    else:
        if args.concurrency is None:
            parser.error("--concurrency is required unless --search-recipe is set")
        if args.request_count is not None and args.request_count < args.concurrency:
            parser.error("--request-count must be greater than or equal to --concurrency")

    return args


def run_fixed(args: argparse.Namespace) -> dict:
    """Run a single concurrency and return the intermediate result."""
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    artifact = run_aiperf(args, args.concurrency, artifact_dir)
    return build_result(artifact, extract_max_concurrency(artifact))


def run_search(args: argparse.Namespace) -> dict:
    """Delegate to AIPerf's native BO recipe and record the winning point."""
    artifact_dir = args.result_dir / f"{args.result_filename}_aiperf"
    cmd = build_search_command(args, artifact_dir)
    subprocess.run(cmd, check=True)

    history = json.loads((artifact_dir / SEARCH_HISTORY).read_text())
    winner_conc, winner_iter, sla_met = winner_from_history(history)
    artifact = winner_profile_export(artifact_dir, winner_iter)
    result = build_result(artifact, winner_conc)
    result["search_recipe"] = args.search_recipe
    result["sla_met"] = sla_met

    if sla_met:
        print(
            f"[aiperf-search] recipe={args.search_recipe} BO winner "
            f"concurrency={winner_conc} "
            f"total_token_throughput={result['total_token_throughput']} "
            f"p95_itl_ms={result['p95_itl_ms']} (SLA met)",
            file=sys.stderr,
        )
    else:
        print(
            f"[aiperf-search] WARNING: recipe={args.search_recipe} found no point "
            f"meeting the SLA; returning best-effort BO winner "
            f"concurrency={winner_conc}.",
            file=sys.stderr,
        )
    return result


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    result = run_search(args) if args.search_recipe else run_fixed(args)

    output_path = args.result_dir / f"{args.result_filename}.json"
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
