#!/usr/bin/env python3
"""Metadata probe: fetch a stack's version_url and diff it against the
`expect` block from smoke-tests.yaml (if any field is declared).

This is a drift check, not just a reachability check -- it catches a
deploy that responds fine but is silently serving a different
model/framework/precision/tp than what's expected.

Usage:
    python3 utils/smoke_tests/probes/metadata.py \
        --version-url http://.../sglang-vanilla-version \
        --expect '{"framework": "sglang", "precision": "fp8", "tp": 2}'

Exits 0 and prints the live payload as JSON on success (no expected fields,
or all declared fields match). Exits 1 with a description of every
mismatch otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discover import fetch_version  # noqa: E402
from result import ProbeResult  # noqa: E402


def check_metadata(live: dict, expect: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty if all match)."""
    mismatches = []
    for field_name, expected_value in expect.items():
        live_value = live.get(field_name)
        if live_value != expected_value:
            mismatches.append(
                f"{field_name}: expected {expected_value!r}, live reports {live_value!r}"
            )
    return mismatches


def run(version_url: str, expect: dict) -> ProbeResult:
    live = fetch_version(version_url)
    mismatches = check_metadata(live, expect)
    if mismatches:
        return ProbeResult(
            ok=False,
            detail="metadata drift: " + "; ".join(mismatches),
            data=live,
        )
    return ProbeResult(ok=True, detail="metadata matches expectations", data=live)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-url", required=True)
    parser.add_argument(
        "--expect", default="{}", help="JSON object of fields to diff against"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.version_url, json.loads(args.expect))

    print(json.dumps(result.data, indent=2))
    if not result.ok:
        print(f"::error::{result.detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
