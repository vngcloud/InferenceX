#!/usr/bin/env python3
"""Run the full smoke-test probe battery for one stack and report results.

Takes one matrix entry (as produced by generate_matrix.py) and runs
whichever probes are listed in its test_cases: metadata, tool-calling,
throughput. Writes a human-readable summary (Markdown, suitable for
$GITHUB_STEP_SUMMARY) and exits non-zero if any probe failed.

Usage:
    python3 utils/smoke_tests/run_smoke_test.py --matrix-entry '<json>'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probes import metadata, tool_calling  # noqa: E402
from result import ProbeResult  # noqa: E402

try:
    from probes import throughput
except ImportError:
    throughput = None


def run_probes(entry: dict) -> dict[str, ProbeResult]:
    results: dict[str, ProbeResult] = {}
    test_cases = entry["test_cases"]

    if "metadata" in test_cases:
        results["metadata"] = metadata.run(entry["version_url"], entry["expect"])

    if "tool-calling" in test_cases:
        results["tool-calling"] = tool_calling.run(
            entry["base_url"], entry["endpoint"], entry["model"]
        )

    if "throughput" in test_cases:
        if throughput is None:
            results["throughput"] = ProbeResult(
                ok=False, detail="throughput probe not available (aiperf not installed)"
            )
        else:
            results["throughput"] = throughput.run(entry, entry["throughput"])

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

    for probe_name, result in results.items():
        if not result.ok:
            print(f"::error::[{entry['name']}] {probe_name} failed: {result.detail}", file=sys.stderr)

    if any(not r.ok for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
