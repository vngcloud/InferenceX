#!/usr/bin/env python3
"""Build the smoke-test job matrix from inference-cicd's live /discover
endpoint, cross-referenced against .github/configs/smoke-tests.yaml.

See design/smoke-test-matrix.md for the full design. Usage:

    python3 utils/smoke_tests/generate_matrix.py \
        --config .github/configs/smoke-tests.yaml \
        [--stack sglang-vanilla] \
        [--discover-url http://.../discover]

Prints a JSON array to stdout, one entry per stack, suitable for a GitHub
Actions `strategy.matrix.include`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discover import DEFAULT_DISCOVER_URL, fetch_discover  # noqa: E402

DEFAULT_TEST_CASES = ["metadata", "tool-calling"]


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

        stack_config = config.get(name)
        if stack_config is None:
            print(
                f"::warning::'{name}' is discoverable via /discover but has no "
                f"entry in smoke-tests.yaml -- running default probe set "
                f"{DEFAULT_TEST_CASES}.",
                file=sys.stderr,
            )
            test_cases = DEFAULT_TEST_CASES
            expect = {}
        else:
            test_cases = stack_config.get("test-cases", DEFAULT_TEST_CASES)
            expect = stack_config.get("expect", {})

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
                "test_cases": test_cases,
                "expect": expect,
            }
        )

    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to smoke-tests.yaml")
    parser.add_argument(
        "--stack",
        default=None,
        help="Only build a matrix entry for this stack name (fails loudly if "
        "not registered in /discover). Omit to cover every discovered stack.",
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
            "No stacks to test -- /discover returned no matching entries."
        )

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
