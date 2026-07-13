#!/usr/bin/env python3
"""Run the full smoke-test probe battery for one stack and report results.

Takes one matrix entry (as produced by generate_matrix.py) and runs
whichever probes are listed in its test_cases: metadata, tool-calling.
Writes a human-readable summary (Markdown, suitable for
$GITHUB_STEP_SUMMARY) and exits non-zero if any probe failed.

Throughput is a separate, standalone workflow (see
utils/throughput_test/ and design/throughput-test.md) -- it is NOT one of
this battery's probes. It moved out because it's a heavier check against a
shared production endpoint and deserves its own cadence/dataset/schema,
not a quick correctness gate bundled with metadata/tool-calling.

Usage:
    python3 utils/smoke_tests/run_smoke_test.py --matrix-entry '<json>'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_metrics import fetch_gpu_model  # noqa: E402
from probes import metadata, tool_calling  # noqa: E402
from result import ProbeResult  # noqa: E402


def run_probes(entry: dict) -> dict[str, ProbeResult]:
    results: dict[str, ProbeResult] = {}
    test_cases = entry["test_cases"]

    if "metadata" in test_cases:
        results["metadata"] = metadata.run(entry["version_url"], entry["expect"])

    if "tool-calling" in test_cases:
        results["tool-calling"] = tool_calling.run(
            entry["base_url"], entry["endpoint"], entry["model"]
        )

    return results


def render_summary(stack_name: str, results: dict[str, ProbeResult]) -> str:
    lines = [f"## Smoke test: `{stack_name}`", "", "| Probe | Result | Detail |", "|---|---|---|"]
    for probe_name, result in results.items():
        icon = "✅" if result.ok else "❌"
        lines.append(f"| {probe_name} | {icon} | {result.detail} |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-entry", required=True, help="One JSON matrix entry (see generate_matrix.py)"
    )
    parser.add_argument("--summary-file", default=None, help="Path to append the Markdown summary to")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to write the full raw results as JSON (all probe data), for "
        "upload as a build artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = json.loads(args.matrix_entry)

    results = run_probes(entry)
    summary = render_summary(entry["name"], results)

    print(summary)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(summary)

    if args.results_file:
        try:
            gpu_model = fetch_gpu_model(entry.get("gpu_metrics_url"))
        except Exception as exc:  # noqa: BLE001 -- enrichment, not a probe -- never fail the job over this
            print(f"::warning::[{entry['name']}] gpu_model lookup failed: {exc}", file=sys.stderr)
            gpu_model = None

        raw = {
            "stack": entry["name"],
            # Lets InferenceX-app file these into its own "live-check" tab,
            # separate from full sweep runs -- see design/smoke-test-matrix.md.
            "run_type": "live-check",
            # Snapshotted at test time, not looked up at ingest time -- see
            # the gpu-metrics discussion with InferenceX-app: gpu_metrics_url
            # reports live pod state, which may have moved/rescheduled by
            # the time ingest runs.
            "gpu_model": gpu_model,
            "probes": {
                name: {"ok": r.ok, "detail": r.detail, "data": r.data}
                for name, r in results.items()
            },
        }
        with open(args.results_file, "w") as f:
            json.dump(raw, f, indent=2)

    for probe_name, result in results.items():
        if not result.ok:
            print(f"::error::[{entry['name']}] {probe_name} failed: {result.detail}", file=sys.stderr)

    if any(not r.ok for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
