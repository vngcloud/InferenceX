#!/usr/bin/env python3
"""Build the throughput-test job matrix from inference-cicd's live /discover
endpoint, cross-referenced against .github/configs/throughput-tests.yaml.

See design/throughput-test.md for the full design. Usage:

    python3 utils/throughput_test/generate_matrix.py \
        --config .github/configs/throughput-tests.yaml \
        [--stack sglang-vanilla] \
        [--discover-url http://.../discover]

Prints a JSON array to stdout, one entry per stack, suitable for a GitHub
Actions `strategy.matrix.include`.

Unlike smoke-test's matrix (which runs a default probe set against every
discovered stack), throughput testing is opt-in per stack: a discovered
stack with no entry in throughput-tests.yaml is silently skipped, not given
a default. Requesting a specific --stack that has no config entry fails
loudly instead, so a typo in a manual dispatch doesn't quietly no-op.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discover import DEFAULT_DISCOVER_URL, fetch_discover  # noqa: E402


def build_matrix(discover_payload: dict, config: dict, requested_stack: str | None) -> list[dict]:
    stacks = discover_payload.get("stacks", [])
    discovered_names = {s["name"] for s in stacks}

    if requested_stack is not None and requested_stack not in discovered_names:
        raise SystemExit(
            f"Stack '{requested_stack}' was requested but is not present in "
            f"/discover's response (registered stacks: {sorted(discovered_names)}). "
            "A deploy that isn't discoverable is itself a signal worth failing on."
        )

    matrix = []
    for stack in stacks:
        name = stack["name"]
        if requested_stack is not None and name != requested_stack:
            continue

        throughput_config = config.get(name)
        if throughput_config is None:
            if requested_stack == name:
                raise SystemExit(
                    f"'{name}' was requested but has no entry in "
                    "throughput-tests.yaml -- add one before requesting it directly."
                )
            print(
                f"::notice::'{name}' has no entry in throughput-tests.yaml -- "
                "skipping (throughput testing is opt-in per stack).",
                file=sys.stderr,
            )
            continue

        matrix.append(
            {
                "name": name,
                "base_url": stack["base_url"],
                "endpoint": stack["endpoint"],
                "version_url": stack["version_url"],
                "gpu_metrics_url": stack.get("gpu_metrics_url"),
                "model": stack["model"],
                "framework": stack.get("framework"),
                "precision": stack.get("precision"),
                "tp": stack.get("tp"),
                "throughput": throughput_config,
            }
        )

    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to throughput-tests.yaml")
    parser.add_argument(
        "--stack",
        default=None,
        help="Only build a matrix entry for this stack name (fails loudly if "
        "not registered in /discover or not configured in throughput-tests.yaml). "
        "Omit to cover every stack that has a throughput-tests.yaml entry.",
    )
    parser.add_argument("--discover-url", default=DEFAULT_DISCOVER_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    discover_payload = fetch_discover(args.discover_url)
    matrix = build_matrix(discover_payload, config, args.stack)

    if not matrix:
        raise SystemExit(
            "No stacks to test -- throughput-tests.yaml has no entries "
            "matching a discovered stack."
        )

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
